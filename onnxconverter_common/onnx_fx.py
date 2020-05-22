# -*- coding: utf-8 -*-
# The codegen script to build oopb opset functions

import onnx
import numpy as np
from onnx import helper
from onnxconverter_common.registration import register_converter
from onnxconverter_common.topology import Topology, convert_topology, Scope
from onnxconverter_common.oopb import OnnxOperatorBuilder
from onnxconverter_common.container import ModelComponentContainer
from onnxconverter_common.data_types import DoubleTensorType, Int64TensorType

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
    # We override the constructors to implement an overload that constructs
    # an ONNX Graph from a Python function (@Graph).
    def __new__(cls, oopbx, *args, **kwargs):
        if len(args) > 0 and hasattr(args[0], '__call__') and not isinstance(args[0], Graph): # overload
            return Graph._to_Graph(oopbx, *args, **kwargs)

    def __init__(self, oopbx, *args, **kwargs):
        if len(args) > 0 and hasattr(args[0], '__call__') and not isinstance(args[0], Graph): # overload
            return
        self._init(oopbx, *args, **kwargs)

    def _init(oopbx, name, inputs, outputs):
        self.oopbx = oopbx
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        # need to add the inputs and outputs to the variable scope

    @staticmethod
    def _to_Graph(oopbx, f, op_name=None, outputs=None, name=None):
        f_name = f.__name__
        arg_names, _ = _get_python_function_arguments(f)
        inputs = [oopbx.arg(arg_name) for arg_name in arg_names]
        f_outputs = f(*inputs)
        if outputs is not None:
            if isinstance(f_outputs, Tensor):
                f_outputs = oopbx.identity([f_outputs], outputs=[outputs])
            else:
                f_outputs = [oopbx.identity([f_output], outputs=[output_name]) for f_output, output_name in zip(f_outputs, outputs)]
        return Graph(oopbx, name=f_name, inputs=inputs, outputs=f_outputs)

    @staticmethod
    def load(path):
        pass

class Tensor:
    def __init__(self, tensor_name: str, oopbx):
        self.name = tensor_name
        self.oopbx = oopbx

    def __add__(self, other):
        return self.oopbx.add([self, other])


class OnnxOperatorBuilderFX(OnnxOperatorBuilder):
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

    def constant(self, name, value, outputs=None):   # override!
        return self._output_names_to_tensors(super().constant(name, value, outputs=[name]))  # not sure about the diff between name and outputs

    def arg(self, name):   # NOT WORKING, use const
        """
        Use this to create a function argument
        """
        return Tensor(name, self)
        #t = helper.make_tensor('arg', self.double, (1,), [0.0])
        #return self.constant(name, t)[0]
    
    def graph(self, *args, **kwargs):
        """
        This is the decorator
        """
        if len(args) > 0 and hasattr(args[0], '__call__'):  # first arg is function
            return Graph(self, args[0])
        else:
            return lambda f: Graph(self, f, *args, **kwargs)


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
#oopbx = OnnxOperatorBuilderFX(container, scope)
#@oopbx.graph(outputs="z")
#def f(x,y):
#    return oopbx.abs(x + y)


def on_conversion(scope, operator, container):
    with OnnxOperatorBuilderFX(container, scope).as_default('node_bn') as oopbx:

        @oopbx.graph(outputs="z")
        def f(x,y):
            return oopbx.abs(x + y)

GRAPH_OPERATOR_NAME = '__test_graph__'
register_converter(GRAPH_OPERATOR_NAME, on_conversion, overwrite=True)
raw_model = _SimpleRawModelContainer(["x", "y"], ["z"])
topo = Topology(raw_model)
top_level = topo.declare_scope('__root__')
top_level.declare_local_operator(GRAPH_OPERATOR_NAME)
for i_ in raw_model.input_names:
    top_level.get_local_variable_or_declare_one(i_, DoubleTensorType(shape=[1]))
for o_ in raw_model.output_names:
    top_level.get_local_variable_or_declare_one(o_, DoubleTensorType(shape=[1]))

oxml = convert_topology(topo, 'test', "doc_string", target_opset=8)
onnx.save_model(oxml, 'c:/me/abssum.onnx')

print("done")
