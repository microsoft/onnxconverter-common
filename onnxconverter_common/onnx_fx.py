# -*- coding: utf-8 -*-
# The codegen script to build oopb opset functions

import onnx
import onnxruntime as ort
import numpy as np

from onnx import helper
from onnxconverter_common.registration import register_converter
from onnxconverter_common.topology import Topology, convert_topology, Scope
from onnxconverter_common.oopb import OnnxOperatorBuilder
from onnxconverter_common.container import ModelComponentContainer
from onnxconverter_common.data_types import DoubleTensorType, FloatTensorType, Int64TensorType, Int32TensorType

class ONNXFunction:
    functions = []

    def __init__(self, op_type, attributes):
        self.op_type = op_type
        self.attributes = attributes


def onnx_function(*op, **kwargs):
    def onnx_func(func):
        ONNXFunction.functions.append(ONNXFunction(op, kwargs))
        return func

    return onnx_func

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
    defaults = param_specs.defaults # "if this tuple has n elements, they correspond to the last n elements listed in args"
    if defaults:
        arg_names = arg_names[:-len(defaults)] # we allow Function(functions with default arguments), but those args will always have default values since CNTK Functions do not support this
    return (arg_names, annotations)

class Graph:
    def __init__(self, name, oxml, inputs, outputs):
        ox_graph = oxml.graph
        if name is None:
            name = ox_graph.name
        initializer_set = { initializer.name for initializer in ox_graph.initializer }
        model_inputs = [input.name for input in ox_graph.input if input.name not in initializer_set]
        model_outputs = [output.name for output in ox_graph.output]
        if inputs is None:
            inputs = model_inputs
        else:
            assert { input for input in inputs } == { input for input in model_inputs }, f"User-specified set of inputs ({', '.join(inputs)}) to {name} does not match actual set ({', '.join(model_inputs)})"
        if outputs is None:
            outputs = model_outputs
        else:
            assert { output for output in outputs } == { output for output in model_outputs }, f"User-specified set of outputs ({', '.join(outputs)}) to {name} does not match actual set ({', '.join(model_outputs)})"
        print(f"Graph: {name}({', '.join(inputs)}) -> {', '.join(outputs)}")
        self._name = name
        self._oxml = oxml
        self._inputs = inputs
        self._outputs = outputs

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
        if outputs is None:
            outputs = []

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
            with OnnxOperatorBuilderX(container, scope).as_default('node_bn') as ox:
                inputs = [ox.arg(arg_name) for arg_name in arg_names]
                f_outputs = f(*inputs)
                if outputs:
                    if isinstance(f_outputs, Tensor):
                        f_outputs = ox.identity([f_outputs], outputs=[outputs])
                    else:
                        f_outputs = [ox.identity([f_output], outputs=[output_name]) for f_output, output_name in zip(f_outputs, outputs)]

        GRAPH_OPERATOR_NAME = f_name
        register_converter(GRAPH_OPERATOR_NAME, on_conversion, overwrite=True)
        top_level.declare_local_operator(GRAPH_OPERATOR_NAME)
        for i_ in raw_model.input_names:
            top_level.get_local_variable_or_declare_one(i_, DoubleTensorType(shape=[1]) if not input_types  else input_types [arg_names.index(i_)])
        for o_ in raw_model.output_names:
            top_level.get_local_variable_or_declare_one(o_, DoubleTensorType(shape=[1]) if not output_types else output_types[outputs.index(o_)])

        oxml = convert_topology(topo, f_name, "doc_string", target_opset=8)
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
                param = params_set[name]
                if param in arg_map:
                    raise SyntaxError("got multiple values for argument '%s'" % name)
                arg_map[param] = arg  # add kw argument to dict
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
            raise TypeError("Graph invocation expected {} arguments, got {}".format(len(params), len(args) + len(kwargs)))
        params_set = { arg for arg in params }
        return Graph._map_function_arguments(params, params_set, *args, **kwargs) 

    def __call__(self, *args, **kwargs):
        # parse argument list and map to the function's input
        arg_map = self._argument_map(*args, **kwargs)
        # determine whether this is eval() or clone()
        is_symbolic = any(isinstance(arg, Tensor) for arg in arg_map.values())
        if is_symbolic:
            first_arg = next(iter(arg_map.values()))
            ox = first_arg.ox
            output_map = { output : None for output in self._outputs}
            return ox.apply_invoke_inline(self._oxml.graph, arg_map, output_map)  # @TODO: outputs missing
        else:
            # evaluate with real values
            return None
    
    def save(self, path):
        if self._oxml.opset_import[0].version < 7:
            self._oxml.opset_import[0].version = 7  # @WORKAROUND: lower versions will crash onnxruntime upon load
        #print(self._oxml.graph.node)
        onnx.checker.check_model(self._oxml)
        onnx.save_model(self._oxml, path)

    @staticmethod
    def load(path, name=None, inputs=None, outputs=None):
        return Graph(name=name, oxml=onnx.load_model(path), inputs=inputs, outputs=outputs)

class Tensor:
    def __init__(self, tensor_name: str, ox):
        self.name = tensor_name
        self.ox = ox

    def __add__(self, other):
        return self.ox.add([self, other])

    def __sub__(self, other):
        return self.ox.sub([self, other])

    def __mul__(self, other):
        return self.ox.mul([self, other])

    def __div__(self, other):
        return self.ox.div([self, other])

    def __pow__(self, other):
        return self.ox.pow([self, other])

    def __matmul__(self, other):
        return self.ox.matmul([self, other])

    def __lt__(self, other):
        return self.ox.less([self, other])

    def __le__(self, other):
        return self.ox.less_or_equal([self, other])

    #def __eq__(self, other):
    #    return self.ox.matmul([self, other])

    #def __ne__(self, other):
    #    return self.ox.matmul([self, other])

    def __gt__(self, other):
        return self.ox.greater([self, other])

    def __ge__(self, other):
        return self.ox.greater_or_equal([self, other])

    def __neg__(self):
        return self.ox.neg([self])

    #def __not__(self):
    #    return self.ox.not([self])


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

    def apply_op(self, apply_func, inputs, name=None, outputs=None, **attrs):  # override!
        inputs  = self._tensors_to_input_names(inputs)
        return self._output_names_to_tensors(super().apply_op(apply_func, inputs, name=name, outputs=outputs, **attrs))
    
    def apply_invoke_inline(self, ox_graph, input_map, output_map):
        # input_map:  [name in graph] -> actual input Tensor
        # output_map: [name in graph] -> desired name for the result, or None
        f_name = "invoke_inline_" + ox_graph.name
        for graph_output in output_map.keys():  # @TODO: use proper comprehensions
            output_map[graph_output] = self._process_outputs(output_map[graph_output], name=f_name)[0]
        for graph_input in input_map.keys():
            input_map[graph_input] = self._process_inputs([input_map[graph_input].name], name=f_name)[0]
        print(f_name, input_map, output_map)
        def map_tensors(args, arg_map):
            for i in range(len(args)):
                if args[i] in arg_map:
                    print("Replacing:", args[i])
                    args[i] = arg_map[args[i]]
        for node in ox_graph.node:
            map_tensors(node.input,  input_map)
            map_tensors(node.output, output_map)
            self._container.nodes.append(node)
        for initializer in ox_graph.initializer:
            if initializer.name in output_map:  # technically, the whole function could be a lonely initializer
                print("Replacing:", initializer.name, initializer.shape)
                initializer.name = output_map[initializer.name]
            #print(initializer.name)
            self._container.initializers.append(initializer)
        for value_info in ox_graph.value_info:
            # @TODO: Not sure what must be mapped, and how
            print(value_info)
            self._container.value_info.append(value_info)
        return self._output_names_to_tensors(output_map.values())

    def constant(self, name, value, outputs=None):   # override!
        return self._output_names_to_tensors(super().constant(name, value, outputs=[name]))  # not sure about the diff between name and outputs

    def arg(self, name):
        """
        Use this to create a function argument
        """
        return Tensor(name, self)

# this works, and the exported graph is usable:

if False:
    @Graph.trace(outputs="s")
    def f(x,y):
        return x + y

    @Graph.trace(outputs="z")
    def g(x,y):
        return x.ox.abs(f(x, y))

    g.save("c:/me/abssum.onnx")


path_stem = "c:/work/marian-dev/local/model/model.npz.best-ce-mean-words-debug-sin-proto"
encode_source = Graph.load(f"{path_stem}.encode_source.onnx",
                           inputs=['data_0', 'data_0_mask', 'data_0_posrange'])  # define the order of arguments
decode_first  = Graph.load(f"{path_stem}.decode_first.onnx",
                           inputs=['data_1_posrange', 'encoder_context_0', 'data_0_mask'],
                           outputs=['logits', 'out_decoder_state_0', 'out_decoder_state_1', 'out_decoder_state_2', 'out_decoder_state_3', 'out_decoder_state_4', 'out_decoder_state_5'])
decode_next   = Graph.load(f"{path_stem}.decode_next.onnx",
                           inputs=['prev_word', 'data_1_posrange', 'encoder_context_0', 'data_0_mask',
                                   'decoder_state_0', 'decoder_state_1', 'decoder_state_2', 'decoder_state_3', 'decoder_state_4', 'decoder_state_5'],
                           outputs=['logits', 'out_decoder_state_0', 'out_decoder_state_1', 'out_decoder_state_2', 'out_decoder_state_3', 'out_decoder_state_4', 'out_decoder_state_5'])


# @WORKAROUND: To make this work, must comment out the call to MergeCommonSequenceOptimizer():

@Graph.trace(
    input_types =[ Int32TensorType(shape=['SOURCE_LENGTH']),
                   FloatTensorType(shape=['SOURCE_LENGTH', 1, 1]),
                   FloatTensorType(shape=['SOURCE_LENGTH', 1, 1])],
    output_types=[ FloatTensorType(shape=[1, 'SOURCE_LENGTH', 1, 512]) ],
    #input_types =[ Int32TensorType(shape=[   None]),
    #               FloatTensorType(shape=[   None, 1, 1]),
    #               FloatTensorType(shape=[   None, 1, 1])],
    #output_types=[ FloatTensorType(shape=[1, None, 1, 512]) ],
    outputs="z")
def h(a, b, c):
    return encode_source(a,b,c)

model_path = "c:/me/enc.onnx"
print("Saving to:", model_path, flush=True)
h.save(model_path)

print("Loading from:", model_path, flush=True)
model = onnx.load(model_path)

#print("Serializing...", flush=True)
#model_str = str(model)
#print("Deleting raw_data...", flush=True)
#model_str = '\n'.join(line for line in model_str.split('\n') if "raw_data" not in line)
#print("Printing to:", model_path, flush=True)
#with open(model_path + ".txt", "wb") as f:
#    print(model_str, file=f)
#print("Done", flush=True)

print("Loading as session:", model_path, flush=True)
ort_sess = ort.InferenceSession(model_path)
output_vals = ort_sess.run(None, {
    'a' : np.array([530, 4, 0]                , dtype=np.int32),
    'b' : np.array([[[1.0]], [[1.0]], [[1.0]]], dtype=np.float32),
    'c' : np.array([[[0.0]], [[1.0]], [[2.0]]], dtype=np.float32)
})

print(output_vals)

print("done")


def greedy_graph(oopb, inputs, outputs):
    mul_node = oopb.mul(inputs)
    sub_node = oopb.sub([mul_node] + [np.array([1.0, 2.0])])
    add_const = helper.make_tensor('add_2_c', oopb.double, (2, 1), [3.0, 4.0])
    div1 = oopb.div([sub_node, oopb.constant('add_2', add_const)])
    oopb.add_node('Add',
                  [div1, ('add_3', oopb.double, np.array([3.0, 4.0]))],
                  outputs=outputs)


    # oopb.noop_unfold(attrs["encode_source"])
    # oopb.noop_unfold(attrs["decode_first"])
    # data_0 = inputs[0]
    # seq_len = oopb.shape(data_0)
    # data_0_mask = oopb.constant(1.0, shape=seq_len)
    # data_0_index_range = oopb.range(seq_len)
    # max_len = oopb.mul(seq_len, 3)

    # encoder_context_0, *_ = oopb.noop_unfold(attrs["encode_source"],
    #                                          data_0=data_0, data_0_mask=data_0_mask,
    #                                          data_0_posrange=data_0_index_range)

    # posrange = oopb.constant(np.arary([[[0]]], dtype=np.float))
    # logp, *out_decoder_states = oopb.noop_unfold(attrs["decode_first"],
    #                                              data_1_posrange=posrange,
    #                                              encoder_context_0=encoder_context_0, data_0_mask=data_0_mask)

    # # !!!! logp[:, :, :, unk_id] = -1e8  # suppress <unk>, like Marian
    # y0 = oopb.argmax(oopb.slice(logp, [0, 0], [1, 1], axis=[0, 1]))
    # test_y0 = oopb.equal(y0, [0])
    # y_len = oopb.constant([1])

    # def loop_body(y0, y_len, encoder_context_0, data_0_mask, out_decoder_states):
    #     data_1_posrange = oopb.unsqueeze(y_len, axes=[0, 1, 2])
    #     logp, *out_decoder_states = oopb.noop_unfold(attrs["decode_next"],
    #                                                  prev_word=[
    #                                                      y0], data_1_posrange=data_1_posrange,
    #                                                  encoder_context_0=encoder_context_0, data_0_mask=data_0_mask,
    #                                                  decoder_state_0=out_decoder_states[
    #         0], decoder_state_1=out_decoder_states[1],
    #         decoder_state_2=out_decoder_states[
    #         2], decoder_state_3=out_decoder_states[3],
    #         decoder_state_4=out_decoder_states[4], decoder_state_5=out_decoder_states[5])
    #     y0 = oopb.argmax(oopb.slice(logp, [0, 0], [1, 1], axis=[0, 1]))
    #     test_y0 = oopb.equal(y0, [0])
    #     y_len = oopb.add(y_len, [1])

    # y = oopb.loop(max_len, test_y0, loop_body,
    #               y0, y_len, encoder_context_0, data_0_mask, out_decoder_states)
    # oopb.identity(y, output=oopb.outputs)


class _SimpleRawModelContainer(object):
    def __init__(self, inputs, outputs):
        self.input_names = inputs
        self.output_names = outputs


def save_function(func, fname, opset, **kwargs):
    GRAPH_OPERATOR_NAME = '__test_graph__'
    inputs = ['input_0', 'input_1']
    outputs =  ["output_0"]
    raw_model = _SimpleRawModelContainer(inputs, outputs)

    def on_conversion(scope, operator, container):
        with OnnxOperatorBuilder(container, scope).as_default('node_bn') as oopb:
            greedy_graph(oopb, inputs, outputs)

    register_converter(GRAPH_OPERATOR_NAME, on_conversion, overwrite=True)
    topo = Topology(raw_model)
    top_level = topo.declare_scope('__root__')
    top_level.declare_local_operator(GRAPH_OPERATOR_NAME)
    for i_ in inputs:
        top_level.get_local_variable_or_declare_one(i_, DoubleTensorType(shape=[1]))

    for o_ in outputs:
        top_level.get_local_variable_or_declare_one(o_, DoubleTensorType(shape=[1]))

    oxml = convert_topology(topo, 'test', "doc_string", target_opset=8)
    onnx.save_model(oxml, 'fluency.onnx')

    oxml = convert_topology(topo, 'test', "doc_string", target_opset=opset, enable_optimizer=False)
    onnx.save_model(oxml, fname)
    import onnxruntime as ort
    ort.InferenceSession(fname)


#container = ModelComponentContainer(target_opset=8)
#scope = Scope('node_bn')
#ox = OnnxOperatorBuilderX(container, scope)
#@ox.graph(outputs="z")
#def f(x,y):
#    return ox.abs(x + y)
