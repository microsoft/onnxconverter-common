# -*- coding: utf-8 -*-
# The codegen script to build oopb opset functions

import numpy as np
from onnx import helper


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


def greedy_graph(oopb, inputs, outputs, opset, **attrs):
    # type: (OnnxOperatorBuilder, [], []) -> None
    data_0 = inputs[0]
    seq_len = oopb.shape(inputs)
    data_0_mask = oopb.constant(1.0, shape=seq_len)
    data_0_index_range = oopb.range(seq_len)
    max_len = oopb.mul(seq_len, 3)

    encoder_context_0, *_ = oopb.noop_unfold(attrs["encode_source"],
                data_0=data_0, data_0_mask=data_0_mask,
                data_0_posrange=data_0_index_range)

    posrange = oopb.constant(np.arary([[[0]]], dtype=np.float))
    logp, *out_decoder_states = oopb.noop_unfold(attrs["decode_first"],
                data_1_posrange=posrange,
                encoder_context_0=encoder_context_0, data_0_mask=data_0_mask)

    # !!!! logp[:, :, :, unk_id] = -1e8  # suppress <unk>, like Marian
    y0 = oopb.argmax(oopb.slice(logp, [0, 0], [1, 1], axis=[0, 1]))
    test_y0 = oopb.equal(y0, [0])
    y_len = oopb.constant([1])
    def loop_body(y0, y_len, encoder_context_0, data_0_mask, out_decoder_states):
        data_1_posrange = oopb.unsqueeze(y_len, axes=[0, 1, 2])
        logp, *out_decoder_states = oopb.noop_unfold(attrs["decode_next"],
            prev_word=[y0], data_1_posrange=data_1_posrange,
                        encoder_context_0=encoder_context_0, data_0_mask=data_0_mask,
                        decoder_state_0=out_decoder_states[
                            0], decoder_state_1=out_decoder_states[1],
                        decoder_state_2=out_decoder_states[
                            2], decoder_state_3=out_decoder_states[3],
                        decoder_state_4=out_decoder_states[4], decoder_state_5=out_decoder_states[5])
        y0 = oopb.argmax(oopb.slice(logp, [0, 0], [1, 1], axis=[0, 1]))                    
        test_y0 = oopb.equal(y0, [0])
        y_len = oopb.add(y_len, [1])

    y = oopb.loop(max_len, test_y0, loop_body,
        y0, y_len, encoder_context_0, data_0_mask, out_decoder_states)
    oopb.identity(y, output=oopb.outputs)


def save_function(func, fname, opset, **kwargs):
    pass
