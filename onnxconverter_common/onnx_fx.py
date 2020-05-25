# -*- coding: utf-8 -*-
# The codegen script to build oopb opset functions

import os, sys, io, copy
import onnx
import onnxruntime as ort
import numpy as np

from onnx import helper
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE
from onnxconverter_common.registration import register_converter
from onnxconverter_common.topology import Topology, convert_topology, Scope
from onnxconverter_common.oopb import OnnxOperatorBuilder
from onnxconverter_common.container import ModelComponentContainer
from onnxconverter_common.data_types import DoubleTensorType, FloatTensorType, Int64TensorType, Int32TensorType, Int64Type, BooleanTensorType
from onnxconverter_common import onnx_ops


class _Ty:
    D = DoubleTensorType
    F = FloatTensorType
    I = Int64TensorType
    B = BooleanTensorType

    d = DoubleTensorType(shape=[])
    f = FloatTensorType(shape=[])
    i = Int64TensorType(shape=[])
    b = BooleanTensorType(shape=[])


def _get_python_function_arguments(f):
    '''
    Helper to get the parameter names and annotations of a Python function.
    '''
    # Note that we only return non-optional arguments (we assume that any optional args are not specified).
    # This allows to, e.g., accept max(a, b, *more, name='') as a binary function
    from inspect import getfullargspec
    param_specs = getfullargspec(f)
    annotations = param_specs.annotations
    arg_names = param_specs.args
    # "if this tuple has n elements, they correspond to the last n elements listed in args"
    defaults = param_specs.defaults
    if defaults:
        # we allow Function(functions with default arguments), but those args will always have default values since CNTK Functions do not support this
        arg_names = arg_names[:-len(defaults)]
    return (arg_names, annotations)


def _to_list(element_or_list):
    if element_or_list is None:
        return []

    return element_or_list if isinstance(
        element_or_list, (list, tuple)) else [element_or_list]


class Graph:

    def __init__(self, name, oxml, inputs, outputs):
        ox_graph = oxml.graph
        if name is None:
            name = ox_graph.name
        initializer_set = {
            initializer.name for initializer in ox_graph.initializer}
        model_inputs = [
            input.name for input in ox_graph.input if input.name not in initializer_set]
        model_outputs = [output.name for output in ox_graph.output]
        if inputs is None:
            inputs = model_inputs
        else:
            assert {input for input in inputs} == {
                input for input in model_inputs}, f"User-specified set of inputs ({', '.join(inputs)}) to {name} does not match actual set ({', '.join(model_inputs)})"
        if outputs is None:
            outputs = model_outputs
        else:
            assert {output for output in outputs} == {
                output for output in model_outputs}, f"User-specified set of outputs ({', '.join(outputs)}) to {name} does not match actual set ({', '.join(model_outputs)})"
        print(f"Graph: {name}({', '.join(inputs)}) -> {', '.join(outputs)}")
        self._name = name
        self._oxml = oxml
        self._inputs = inputs
        self._outputs = outputs
        self._sess = None

    @staticmethod
    def trace(*args, **kwargs):
        """
        This is the decorator. Example:
        @Graph.trace(outputs="logits")
        def model(source_sequence):
            ...
        """
        if len(args) > 0 and hasattr(args[0], '__call__'):  # first arg is function
            return Graph._to_Graph(args[0])
        else:
            return lambda f: Graph._to_Graph(f, *args, **kwargs)

    @staticmethod
    def _to_Graph(f, op_name=None, input_types=None, output_types=None, outputs=None, name=None):
        assert outputs is not None, "outputs has to be specified."
        input_types = _to_list(input_types)
        output_types = _to_list(output_types)
        outputs = _to_list(outputs)

        f_name = f.__name__
        arg_names, _ = _get_python_function_arguments(f)

        class _SimpleRawModelContainer(object):
            def __init__(self, inputs, outputs):
                self.input_names = inputs
                self.output_names = outputs
        raw_model = _SimpleRawModelContainer(arg_names, outputs)
        topo = Topology(raw_model)
        top_level = topo.declare_scope(f_name)

        inputs = None
        f_outputs = None

        def on_conversion(scope, operator, container):
            nonlocal inputs
            nonlocal f_outputs
            with OnnxOperatorBuilderX(container, scope).as_default(f_name) as ox:
                inputs = [ox.arg(arg_name) for arg_name in arg_names]
                f_outputs = f(*inputs)
                if outputs:
                    if isinstance(f_outputs, Tensor):
                        f_outputs = ox.identity([f_outputs], outputs=outputs)
                        n_outputs = 1
                    else:
                        f_outputs = [ox.identity([f_output], outputs=[output_name])
                                                 for f_output, output_name in zip(f_outputs, outputs)]
                        n_outputs = len(f_outputs)
                assert n_outputs == len(outputs), f"Function {f_name}() returned {n_outputs} but {len(outputs)} were declared"

        GRAPH_OPERATOR_NAME = f_name
        register_converter(GRAPH_OPERATOR_NAME, on_conversion, overwrite=True)
        top_level.declare_local_operator(GRAPH_OPERATOR_NAME)
        for i_ in raw_model.input_names:
            top_level.get_local_variable_or_declare_one(i_, FloatTensorType(shape=[1]) if not input_types  else input_types [arg_names.index(i_)])
        for o_ in raw_model.output_names:
            top_level.get_local_variable_or_declare_one(o_, FloatTensorType(shape=[1]) if not output_types else output_types[outputs.index(o_)])

        oxml = convert_topology(topo, f_name, "doc_string", target_opset=9, enable_optimizer=False)
        return Graph(name=f_name, oxml=oxml, inputs=arg_names, outputs=outputs)

    @staticmethod
    def _map_function_arguments(params, params_set, *args, **kwargs):
        '''
        Helper to determine the argument map for use with various call operations.
        Returns a dictionary from parameters to whatever arguments are passed.
        Accepted are both positional and keyword arguments.
        This mimics Python's argument interpretation, except that keyword arguments are not optional.
        '''
        # start with positional arguments
        arg_map = dict(zip(params, args))

        # now look up keyword arguments
        if len(kwargs) != 0:
            for name, arg in kwargs.items():  # keyword args are matched by name
                if name not in params_set:
                    raise TypeError("got an unexpected keyword argument '%s'" % name)
                if name in arg_map:
                    raise SyntaxError("got multiple values for argument '%s'" % name)
                arg_map[name] = arg  # add kw argument to dict
        assert len(arg_map) == len(params)

        return arg_map

    def _argument_map(self, *args, **kwargs):
        '''
        Determines the {placeholder: variable} map for use with various call operations
        Returns a dictionary from this function's placeholders to whatever arguments are passed.
        Accepted are both positional and keyword arguments.
        This mimics Python's argument interpretation, except that keyword arguments are not optional
        (there is no concept of default value).
        '''
        params = self._inputs
        if len(args) + len(kwargs) != len(params):
            raise TypeError("Graph invocation expected {} arguments, got {}".format(
                len(params), len(args) + len(kwargs)))
        params_set = {arg for arg in params}
        return Graph._map_function_arguments(params, params_set, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        # parse argument list and map to the function's input
        arg_map = self._argument_map(*args, **kwargs)
        # determine whether this is eval() or clone()
        is_symbolic = any(isinstance(arg, Tensor) for arg in arg_map.values())
        if is_symbolic:
            first_arg = next(iter(arg_map.values()))
            ox = first_arg.ox
            output_map = {output: None for output in self._outputs}
            # @TODO: outputs missing
            return ox.apply_invoke_inline(self._oxml.graph, arg_map, output_map)
        else:
            # evaluate with real values
            kwargs = { name : val for name, val in arg_map.items() }
            if self._sess == None:       # This requires an ORT session, which we create lazily and keep around for future calls.
                self._sess = ort.InferenceSession(self._oxml.SerializeToString())
            res = self._sess.run(None, kwargs)
            return res  # @TODO: of more than one, turn into a dict, or something
    
    def save(self, path):
        if self._oxml.opset_import[0].version < 7:    # @WORKAROUND: lower versions will crash onnxruntime upon load
            self._oxml.opset_import[0].version = 7
        #print("Model:")
        #print("--- opset_import:", self._oxml.opset_import[0])
        #print("--- inputs:", self._oxml.graph.input)
        #print("--- outputs:", self._oxml.graph.output)
        #print("--- nodes:", self._oxml.graph.node)
        onnx.save_model(self._oxml, path)
        if False:
            print("Saving as text: ", path + ".txt")
            with open(path + ".txt", "wt") as f:
                print(self._oxml, file=f)
            print("Done saving as text")

        # check the model
        #onnx.checker.check_model(self._oxml)
        try:
            import onnxruntime as _ort
            _ort.InferenceSession(self._oxml.SerializeToString())
        except Exception as e:
            print(e)
        print("{} saved!".format(path))

    @staticmethod
    def load(path, name=None, inputs=None, outputs=None):
        return Graph(name=name, oxml=onnx.load_model(path), inputs=inputs, outputs=outputs)


class Tensor(object):
    def __init__(self, tensor_name: str, ox):
        self.name = tensor_name
        self.ox = ox
    
    def _to_binary_tensor_args(self, other):  # convert self, other to [self, other], but if either is a number, convert that to a constant
        x, y = self, other
        if (isinstance(y, (int, float, bool, np.ndarray))):
            y = self.ox.constant(value=y)
        elif (isinstance(x, (int, float, bool, np.ndarray))):
            x = self.ox.constant(value=x)
        return [x, y]

    def __add__(self, other):
        return self.ox.add(self._to_binary_tensor_args(other))

    def __sub__(self, other):
        return self.ox.sub(self._to_binary_tensor_args(other))

    def __mul__(self, other):
        return self.ox.mul(self._to_binary_tensor_args(other))

    def __div__(self, other):
        return self.ox.div(self._to_binary_tensor_args(other))

    def __pow__(self, other):
        return self.ox.pow(self._to_binary_tensor_args(other))

    def __matmul__(self, other):
        return self.ox.matmul(self._to_binary_tensor_args(other))

    def __lt__(self, other):
        return self.ox.less(self._to_binary_tensor_args(other))

    def __le__(self, other):
        return self.ox.less_or_equal(self._to_binary_tensor_args(other))

    def __eq__(self, other):
       return self.ox.equal(self._to_binary_tensor_args(other))

    # def __ne__(self, other):
    #    return self.ox.matmul(self._to_binary_tensor_args(other))

    def __gt__(self, other):
        return self.ox.greater(self._to_binary_tensor_args(other))

    def __ge__(self, other):
        return self.ox.greater_or_equal(self._to_binary_tensor_args(other))

    def __neg__(self):
        return self.ox.neg([self])

    # def __not__(self):
    #    return self.ox.not([self])

    def __getitem__(self, indices):
        # normalize indices to tuples of slices
        # Formats encountered:
        #  - a single int
        #  - a tuple of (int or slice)
        if not isinstance(indices, (tuple, list)):  # single item: make it a tuple
            indices = (indices,)
        squeeze = [axis for axis, index in enumerate(indices) if isinstance(index, int)]  # which axes had a single index?
        indices = tuple(index if isinstance(index, slice) else slice(index, index+1 if index != -1 else None, 1) for index in indices)  # make all tuple items of type Slice
        bs, es, ss, ds = [], [], [], []
        INT_MAX = 2**63-1  # @TODO: Is this MAX_INT according to the ONNX spec?
        for axis, index in enumerate(indices):
            if not isinstance(index, slice):
                raise ValueError("Index expected")
            if index.start is None and index.stop is None:  # [:] can be skipped
                continue
            b, e, s = index.start, index.stop, index.step
            bs.append(b if b is not None else 0)
            es.append(e if e is not None else INT_MAX)
            ss.append(s if s is not None else 1)
            ds.append(axis)
        res = self.ox.slice(self, starts=bs, ends=es, axes=ds, steps=ss)
        if squeeze:  # single index means we must drop the axis
            res = self.ox.squeeze([res], axes=squeeze)
        return res

    _all_ops = [op_name[6:] for op_name in dir(onnx_ops) if op_name.startswith("apply_")] +  \
               ['shape', 'constant_of_shape', 'range', 'slice', 'equal']  # these are temporarily declared here

    def __getattribute__(self, attr):
        """
        A little hack that allows to call unary operators in a chaining fashion,
        e.g. x.shape() instead of ox.shape(x).
        """
        if attr in Tensor._all_ops:
            f = self.ox.__getattribute__(attr)
            def call_it(*args, **kwargs):
                assert len(args) == 0, "In chaining expressions, only keyword args are allowed"
                assert "inputs" not in kwargs, "Chaining expressions do not currently support additional inputs"
                return f(self, *args, **kwargs)
            return call_it
        else:
            return object.__getattribute__(self, attr)


class OnnxOperatorBuilderX(OnnxOperatorBuilder):
    def _output_names_to_tensors(self, outputs):
        if isinstance(outputs, str):
            return Tensor(outputs, self)
        else:
            return [self._output_names_to_tensors(output) for output in outputs]

    def _tensors_to_input_names(self, inputs):
        if isinstance(inputs, Tensor):
            return inputs.name
        else:
            return [self._tensors_to_input_names(input) for input in inputs]

    def apply_op(self, apply_func_or_op_type, inputs, name=None, outputs=None, **attrs):  # override!
        inputs = self._tensors_to_input_names(inputs)
        if isinstance(apply_func_or_op_type, str):
            return self._output_names_to_tensors(super().add_node(apply_func_or_op_type, inputs, name=name, outputs=outputs, **attrs))
        else:
            return self._output_names_to_tensors(super().apply_op(apply_func_or_op_type, inputs, name=name, outputs=outputs, **attrs))

    def apply_invoke_inline(self, ox_graph, input_map, output_map):
        input_map  = dict(input_map)
        output_map = dict(output_map)
        # input_map:  [name in graph] -> actual input Tensor
        # output_map: [name in graph] -> desired name for the result, or None
        f_name = "invoke_inline_" + ox_graph.name
        for graph_output in output_map.keys():  # @TODO: use proper comprehensions
            output_map[graph_output] = self._process_outputs(
                output_map[graph_output], name=f_name)[0]
        outputs = list(output_map.values())  # remember these; these are the outputs of this invocation
        if len(outputs) == 1:
            outputs = outputs[0]  # single output
        for graph_input in input_map.keys():
            input_map[graph_input] = self._process_inputs(
                [input_map[graph_input].name], name=f_name)[0]
        print(f_name, input_map, output_map)

        existing_node_names        = { item.name : item for item in self._container.nodes }
        existing_initializer_names = { item.name : item for item in self._container.initializers }
        existing_value_infos       = { item.name : item for item in self._container.value_info }

        # collect all outputs from the graph we are expanding, so that we can map them to unique names
        # @TODO: This will also map some code that may be shared later on. Leave that to the optimizer.
        node_map = dict()
        for node in ox_graph.node:
            if not node.input:  # leaves do not need to be mapped; they can just get uniq'ed
                continue
            for output in node.output:
                if output in output_map:  # this is an actual output that already has been mapped
                    continue
                uniq_name = onnx_ops._create_name_or_use_existing_one(self._scope, self._generate_name(output, None), None)
                output_map[output] = uniq_name
            uniq_node_name = onnx_ops._create_name_or_use_existing_one(self._scope, self._generate_name(node.name, None), None)
            node_map[output] = uniq_node_name

        def map_tensors(args, arg_map):
            for i in range(len(args)):
                if args[i] in arg_map:
                    print("Remapping", args[i], "to", arg_map[args[i]])
                    args[i] = arg_map[args[i]]

        for node in ox_graph.node:
            node = copy.deepcopy(node)            # since we patch, we must clone it first
            map_tensors(node.input,  input_map)   # patch the input references to the function arguments
            map_tensors(node.output, output_map)  # rename the outputs to unique ones
            map_tensors(node.input,  output_map)  # outputs may be inputs to other nodes in this graph
            if node.name in node_map:
                node.name = node_map[node.name]
            if node.name in existing_node_names:
                str_node  = str(node)
                str_other = str(existing_node_names[node.name])
                if str_node != str_other:  # must be the samem, otherwise we have inconsistent dups, e.g. in input models
                    print("Duplicate node name with inconsistent nodes:\n", node, "vs:\n", existing_node_names[node.name])
                    assert str_node == str_other
                continue
            self._container.nodes.append(node)
        for initializer in ox_graph.initializer:
            if initializer.name in existing_initializer_names:  # @TODO: check if they are the same
                print("Duplicate initializer name skipped:", initializer.name)
                continue
            if initializer.name in output_map:  # technically, the whole function could be a lonely initializer
                #print("Replacing:", initializer.name, initializer.shape)
                initializer = copy.deepcopy(initializer)
                initializer.name = output_map[initializer.name]
            # print(initializer.name)
            self._container.initializers.append(initializer)
        for value_info in ox_graph.value_info:
            if value_info.name in existing_value_infos:  # @TODO: check if they are the same
                print("Duplicate value_info name skipped:", value_info.name)
                continue
            # @TODO: Not sure what must be mapped, and how
            print(value_info)
            self._container.value_info.append(value_info)
        return self._output_names_to_tensors(outputs)
    
    def _value_to_tensor(self, value, name, atleast_1d=False):
        if isinstance(value, (int, float, bool)):
            ty = np.int64
            if isinstance(value, float):
                ty = np.float32
            elif isinstance(value, bool):
                ty = np.bool
            else:
                pass
            value = np.array(value).astype(ty)
            if atleast_1d:
                value = np.atleast_1d(value)  # e.g. constant_of_shape() needs this
        if isinstance(value, np.ndarray):
            l = value.flatten().tolist()
            value = helper.make_tensor(name, NP_TYPE_TO_TENSOR_TYPE[value.dtype], value.shape, l)
        return value

    def constant(self, name=None, value=None, outputs=None):   # override!
        if name is None:  # @BUGBUG: Somehow, constant() does not accept None...??
            name = onnx_ops._create_name_or_use_existing_one(self._scope, self._generate_name("constant", name), None)
        assert value is not None
        value = self._value_to_tensor(value, name)
        return self._output_names_to_tensors(super().constant(name, value, outputs=[name]))[0]  # strip an extra level of list()

    def arg(self, name):
        """
        Use this to create a function argument
        """
        return Tensor(name, self)

    # @TODO: make these proper ops
    def shape(self, inputs, name=None, outputs=None):
        return self.apply_op(lambda scope, input_name, output_name, container, operator_name=None:
                                onnx_ops._apply_unary_operation(scope, 'Shape', input_name, output_name, container, operator_name=operator_name),
                             inputs, name, outputs)
    
    def constant_of_shape(self, inputs, name=None, value=None):
        if name is None:  # @BUGBUG: Somehow, constant() does not accept None...??
            name = self._generate_name("constant_of_shape", name)
        assert value is not None
        if not isinstance(value, (int, float, bool)):
            raise ValueError("constant_of_shape requires 'value' to be a scalar")
        value = self._value_to_tensor(value, name, atleast_1d=True)
        def apply_constant_of_shape(scope, input_names, output_names, container, operator_name=None, output_seq=0, **attrs):
            name = onnx_ops._create_name_or_use_existing_one(scope, 'ConstantOfShape', operator_name)
            attrs['value'] = value
            container.add_node('ConstantOfShape', input_names, output_names, name=name, op_version=9, **attrs)
        return self.apply_op(apply_constant_of_shape, inputs, name, None)

    def range(self, inputs, name=None, outputs=None):
        return self.apply_op(lambda scope, input_name, output_name, container, operator_name=None:
                                onnx_ops._apply_unary_operation(scope, 'Range', input_name, output_name, container, operator_name=operator_name),
                             inputs, name, outputs)

    def slice(self, inputs, starts, ends, name=None, outputs=None, axes=None, steps=None):
        def apply_slice(scope, input_names, output_names, container, operator_name=None, output_seq=0, **attrs):
            name = onnx_ops._create_name_or_use_existing_one(scope, 'Slice', operator_name)
            attrs['starts'] = starts
            attrs['ends'] = ends
            if axes:
                attrs['axes'] = axes
            if steps and any(step != 1 for step in steps):
                attrs['steps'] = steps  # @BUGBUG: This does not seem to get recognized
            container.add_node('Slice', input_names, output_names, name=name, op_version=9, **attrs)
        return self.apply_op(apply_slice, inputs, name, None)

    def equal(self, inputs, name=None, outputs=None):
        def apply_equal(scope, input_names, output_name, container, operator_name=None):
            name = onnx_ops._create_name_or_use_existing_one(scope, 'Greater', operator_name)
            if container.target_opset < 7:
                op_version = 1
            elif container.target_opset < 9:
                op_version = 7
            else:
                op_version = 9
            container.add_node('Equal', input_names, output_name, name=name, op_version=op_version)
        return self.apply_op(apply_equal, inputs, name, outputs)

    def apply_tensor(self, func, inputs, output):
        func(self, inputs, outputs=[output])
        return output

    # @TODO?: Should we follow the conventions in the spec? loop(self, count, cond, inputs, body)
    def loop(self, count, cond, body, inputs, outputs=None, name=None):
        inputs = self._tensors_to_input_names(inputs)
        count = None if count is None else self._tensors_to_input_names(count)
        if cond is not None:
            cond = self._tensors_to_input_names(cond)
        else:
            cond = self._tensors_to_input_names(self.constant(value=np.array([True]), name="cf"))
        # need update the sub-graph since the loop will add more inputs
        sub_graph = body._oxml.graph
        #original_inputs = copy.deepcopy(sub_graph.input)
        #new_inputs = [onnx.helper.make_tensor_value_info('M', self.int64, shape=[1]),
        #              onnx.helper.make_tensor_value_info('c', self.bool, shape=[1])
        #            ] + list(original_inputs)
        #del sub_graph.input[:]
        #sub_graph.input.extend(new_inputs)
        # sg_output_names = [nm_.name for nm_ in sub_graph.output]
        # sub_graph.node.extend([
        #     onnx.helper.make_node('Constant', [], ['cond_o'], 'con_nd',
        #         value=self._value_to_tensor(np.array([True]).astype(np.bool), name='ts_xx')),
        #     #TODO: why do we need this??? ORT limitation?
        #     onnx.helper.make_node('Cast', [sg_output_names[0]], ['loop_out'], 'cast_nd',
        #         to=self.float)
        # ])
        # del sub_graph.output[:]
        # scan_outputs = []
        # for ot_ in sub_graph.output[1:]:
        #     shape = [dim_.dim_value for dim_ in ot_.type.tensor_type.shape.dim] if \
        #         hasattr(ot_, 'type') and hasattr(ot_.type.tensor_type.shape, 'dim') else []
        #     shape = [None] + shape if not shape else []
        #     scan_outputs.append(
        #         onnx.helper.make_tensor_value_info('so_' + ot_.name, ot_.type.tensor_type.elem_type, shape))

        return self._output_names_to_tensors(super().loop(count, cond, sub_graph,
                                                          inputs=inputs,
                                                          outputs=outputs,  # @TODO: unique output names
                                                          name=name))


# this works, and the exported graph is usable:

if True:
    @Graph.trace(outputs="s")
    def f(x,y):
        return x + y

    @Graph.trace(outputs="z")
    def g(x,y):
        return x.ox.abs(f(x, y) + 1.0)

    # g.save("c:/me/abssum.onnx")
    g.save("abssum.onnx")

    print(g([2.0], [-5.0]))

import sys
if len(sys.argv) > 1:
    text = input("Python process id: {} >".format(os.getpid()))  # or raw_input in python2

if True:
    @Graph.trace(outputs='range_res',
                 input_types  = [_Ty.I(shape=[])],
                 output_types = [_Ty.I(shape=['N'])])
    def onnx_range(len):
        ox = len.ox
        is_true = ox.constant(value=True)  # dummy condition, always True
        dummy_state_val = ox.constant(value=True)
        @Graph.trace(outputs=['c_o', 's_o', 'i_o'],
                     input_types  = [_Ty.I([1]), _Ty.b, _Ty.b],
                     output_types = [_Ty.b, _Ty.b, _Ty.i])
        def range_body(iteration_num, condition, dummy_state):
            """
            Loop body follows the requirements of ONNX Loop:

            "The graph run each iteration.
            It has 2+N inputs: (iteration_num, condition, loop carried dependencies...).
            It has 1+N+K outputs: (condition, loop carried dependencies..., scan_outputs...).
            Each scan_output is created by concatenating the value of the specified output value at the end of each iteration of the loop.
            It is an error if the dimensions or data type of these scan_outputs change across loop iterations."

            Inputs:
                iteration_num
                condition (dummy)
                dummy_state: loop-carried dependencies  --@BUGBUG: ORT requires at least one. Known and fixed in opset 11.

            Outputs:
                c_o: dummy condition, always True
                s_o: dummy loop-carried dependencies
                i_o: K scan outputs
            """
            iteration_num = iteration_num + 0  # @WORKAROUND: iteration_num is updated by the ONNX loop in-place; adding 0 creates a copy
            return is_true, dummy_state_val, iteration_num

        # "Final N loop carried dependency values then K scan_outputs"
        _, range_out = ox.loop(len, is_true, range_body, inputs=[dummy_state_val], outputs=['ds_o', 'range_out'])  # passing is_true for dummy_state
        return range_out

    onnx_range.save('range.onnx')
    print(onnx_range(np.array(16, dtype=np.int64)))


if True:  # old version that does only one step
    # path_stem = "c:/work/marian-dev/local/model/model.npz.best-ce-mean-words-debug-sin-uniq"
    path_stem = "C:/f/.odxcaches/_modeldata/model.npz.best-ce-mean-words-debug-sin-uniq"
    encode_source = Graph.load(f"{path_stem}.encode_source.onnx",
                            inputs=['data_0', 'data_0_mask', 'data_0_posrange'])  # define the order of arguments
    decode_first  = Graph.load(f"{path_stem}.decode_first.onnx",
                            inputs=['data_1_posrange', 'encoder_context_0', 'data_0_mask'],
                            outputs=['first_logits', 'first_decoder_state_0', 'first_decoder_state_1', 'first_decoder_state_2', 'first_decoder_state_3', 'first_decoder_state_4', 'first_decoder_state_5'])
    decode_next   = Graph.load(f"{path_stem}.decode_next.onnx",
                            inputs=['prev_word', 'data_1_posrange', 'encoder_context_0', 'data_0_mask',
                                    'decoder_state_0', 'decoder_state_1', 'decoder_state_2', 'decoder_state_3', 'decoder_state_4', 'decoder_state_5'],
                            outputs=['next_logits', 'next_decoder_state_0', 'next_decoder_state_1', 'next_decoder_state_2', 'next_decoder_state_3', 'next_decoder_state_4', 'next_decoder_state_5'])
    @Graph.trace(
        input_types =[ Int32TensorType(shape=['SOURCE_LENGTH']),
                       FloatTensorType(shape=['SOURCE_LENGTH', 1, 1]),
                       FloatTensorType(shape=['SOURCE_LENGTH', 1, 1])],
        output_types=[ Int64TensorType(shape=[1, 'SOURCE_LENGTH', 1, 1]) ],
        outputs="Y")
    def greedy_search(X):
        ox = X.ox
        data_0 = X
        seq_len = data_0.shape()[-1]
        data_0_mask = ox.constant_of_shape(seq_len, value=1.0)
        #data_0_index_range = seq_len.range()
        data_0_index_range = onnx_range(seq_len).cast(to=1)  # 1=float32
        max_len = seq_len * 3

        encoder_context_0 = encode_source(data_0=data_0, data_0_mask=data_0_mask,
                                        data_0_posrange=data_0_index_range)

        y_len_0 = ox.constant(value=0.0)
        logp, *out_decoder_states = decode_first(data_1_posrange=y_len_0,
                                                encoder_context_0=encoder_context_0, data_0_mask=data_0_mask)
        
        # # !!!! logp[:, :, :, unk_id] = -1e8  # suppress <unk>, like Marian
        y_t = logp[0,0].argmax(axis=-1)
        test_y_t = (y_t == 0)

        #Y = [y_t]
        #for t in range(2):
        #    logp, *out_decoder_states = decode_next(
        #        prev_word=y_t, data_1_posrange=y_len,
        #        encoder_context_0=encoder_context_0, data_0_mask=data_0_mask,
        #        decoder_state_0=out_decoder_states[0], decoder_state_1=out_decoder_states[1],
        #        decoder_state_2=out_decoder_states[2], decoder_state_3=out_decoder_states[3],
        #        decoder_state_4=out_decoder_states[4], decoder_state_5=out_decoder_states[5])
        #    y_t = logp[0,0].argmax(axis=-1)
        #    test_y_t = (y_t == 0)
        #    y_len = y_len + 1.0
        #
        #    Y.append(y_t)
        #
        #Y = ox.concat(Y, axis=1)

        @Graph.trace(outputs=['ty_t', 'y_t_o', 'ods_0', 'ods_1', 'ods_2', 'ods_3', 'ods_4', 'ods_5', 'y_t_o2'],
                     output_types=[       _Ty.b, _Ty.i] + [_Ty.f] * 6 + [_Ty.i],
                     input_types =[_Ty.i, _Ty.b, _Ty.i] + [_Ty.f] * 6)
        def loop_body(iteration_count, condition,  # these are not actually used inside
                      y_t,
                      out_decoder_states_0, out_decoder_states_1,
                      out_decoder_states_2, out_decoder_states_3,
                      out_decoder_states_4, out_decoder_states_5):
            """
            Loop body follows the requirements of ONNX Loop:

            "The graph run each iteration.
            It has 2+N inputs: (iteration_num, condition, loop carried dependencies...).
            It has 1+N+K outputs: (condition, loop carried dependencies..., scan_outputs...).
            Each scan_output is created by concatenating the value of the specified output value at the end of each iteration of the loop.
            It is an error if the dimensions or data type of these scan_outputs change across loop iterations."

            Inputs:
                iteration_num (not used by our function)
                test_y_t: condition (not used as an input)
                y_t, *out_decoder_states: N=7 loop-carried dependencies

            Outputs:
                test_y_t: condition
                y_t, *out_decoder_states: N=7 loop-carried dependencies (same as in the Inputs section)
                y_t: K=1 outputs
            """
            pos = iteration_count + 1
            data_1_posrange = pos.cast(to=1).unsqueeze(axes=[0, 1, 2])
            logp, *out_decoder_states = decode_next(
                prev_word=y_t, data_1_posrange=data_1_posrange,
                encoder_context_0=encoder_context_0, data_0_mask=data_0_mask,
                decoder_state_0=out_decoder_states_0, decoder_state_1=out_decoder_states_1,
                decoder_state_2=out_decoder_states_2, decoder_state_3=out_decoder_states_3,
                decoder_state_4=out_decoder_states_4, decoder_state_5=out_decoder_states_5)
            y_t = logp[0,0].argmax(axis=-1)
            test_y_t = (y_t == 0)
            return [test_y_t, y_t] + out_decoder_states + [y_t]

        # "Final N loop carried dependency values then K scan_outputs"
        ret_vals = ox.loop(max_len, test_y_t, loop_body,
                           inputs=[y_t] + out_decoder_states,
                           outputs=['gy_t_o', 'gods_0', 'gods_1', 'gods_2', 'gods_3', 'gods_4', 'gods_5', 'greedy_out'])
        y = ret_vals[-1]  # scan_output
        return y

    # greedy_search.save("c:/me/greedy.onnx")
    greedy_search.save("greedy.onnx")

    Y = greedy_search(np.array([530, 4, 0], dtype=np.int32))[0]
    print(Y.shape, Y)

    # @BUGBUG: This last one kills the model checker. The two above work.
    @Graph.trace(
        input_types =[ Int32TensorType(shape=['SOURCE_LENGTH']),
                    FloatTensorType(shape=['SOURCE_LENGTH', 1, 1]),
                    FloatTensorType(shape=['SOURCE_LENGTH', 1, 1])],
        output_types=[ FloatTensorType(shape=[1, 'SOURCE_LENGTH', 1, 512]) ],
        outputs="z")
    def h(a, b, c):
        return encode_source(a,b,c)

    model_path = "enc.onnx"
    print("Saving to:", model_path, flush=True)
    h.save(model_path)

    res = h(np.array([530, 4, 0]                , dtype=np.int32),
            np.array([[[1.0]], [[1.0]], [[1.0]]], dtype=np.float32),
            np.array([[[0.0]], [[1.0]], [[2.0]]], dtype=np.float32))

    print(res)
