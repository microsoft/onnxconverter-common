# -*- coding: utf-8 -*-
# The codegen script to build oopb opset functions

import onnx
import numpy as np
from onnx import helper
from onnxconverter_common.registration import register_converter
from onnxconverter_common.topology import Topology, convert_topology
from onnxconverter_common.oopb import OnnxOperatorBuilder


from .data_types import DoubleTensorType

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

    oxml = convert_topology(topo, 'test', "doc_string", target_opset=opset, enable_optimizer=False)
    onnx.save_model(oxml, fname)
    import onnxruntime as ort
    ort.InferenceSession(fname)
