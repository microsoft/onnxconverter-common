import numpy as np
from onnxconverter_common.onnx_fx import Graph
from onnxconverter_common.onnx_fx import GraphFunctionType as _Ty
import unittest


if False:
    class ONNXFunctionTest(unittest.TestCase):
        # this works, and the exported graph is usable:
        def test_core(self):
            @Graph.trace(outputs="s")
            def f(x, y):
                return x + y

            @Graph.trace(outputs="z")
            def g(x, y):
                return x.ox.abs(f(x, y) + 1.0)

            self.assertTrue(
                np.allclose(g([2.0], [-5.0]), np.array([2.0])))

        def test_loop(self):
            @Graph.trace(outputs=['y1', 'y2', 'y3', 'y4'],
                         input_types=[_Ty.I(shape=[1])],
                         output_types=[_Ty.F(shape=[None]), _Ty.F(shape=[None]), _Ty.F(shape=[None]), _Ty.F(shape=[None])])
            def loop_test(len):
                ox = len.ox
                s_len = ox.squeeze(len, axes=[0])
                is_true = ox.constant(value=True)

                @Graph.trace(outputs=['c_o', 'i_o', 'j_o', 'all_i', 'all_j'],
                             output_types=[_Ty.b, _Ty.f, _Ty.f, _Ty.f, _Ty.f],
                             input_types=[_Ty.I([1]), _Ty.b, _Ty.F(shape=[1]), _Ty.F(shape=[1])])
                def range_body(iter_n, cond, i, j):
                    return (is_true,
                            i + i.ox.constant(value=1.0), j + 2.0, i, j)

                one_c = ox.constant(value=-1.0)
                y1, y2, y3, y4 = ox.loop(s_len, is_true, range_body, inputs=[one_c, one_c],
                                         outputs=['y1_o', 'y2_o', 'y3_o', 'y4_o'])
                return y1, y2, y3, y4

            self.assertEqual(
                loop_test(np.array([16], dtype=np.int64))[2][4], 3.0)


if __name__ == '__main__':
    if False: # due to early build graph, the test cannot be disabled here.
        suite = unittest.TestLoader().loadTestsFromTestCase(ONNXFunctionTest)
        suite.debug()
    
    # --- range

    @Graph.trace(outputs='n',
                output_types = [_Ty.b],
                input_types = [_Ty.b])
    def func_not(x):
        return x.ox.not_(x)

    print(func_not(np.array([True])))

    @Graph.trace(outputs='range_res',
                 input_types=[_Ty.I(shape=[])],
                 output_types=[_Ty.I(shape=['N'])])
    def onnx_range(len):
        ox = len.ox
        is_true = ox.constant(value=True)  # dummy condition, always True
        dummy_state_val = ox.constant(value=True)

        @Graph.trace(outputs=['c_o', 's_o', 'i_o'],
                     input_types=[_Ty.I([1]), _Ty.b, _Ty.b],
                     output_types=[_Ty.b, _Ty.b, _Ty.i])
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
        _, range_out = ox.loop(len, is_true, range_body, inputs=[dummy_state_val],
                               outputs=['ds_o', 'range_out'])  # passing is_true for dummy_state
        return ox.squeeze(range_out, axes=[1])

    onnx_range.save('range.onnx')
    print(onnx_range(np.array(16, dtype=np.int64)))

    import sys
    if len(sys.argv) > 1:
        import os
        text = input("Python process id: {} > ".format(os.getpid()))  # or raw_input in python2

    # path_stem = "c:/work/marian-dev/local/model/model.npz.best-ce-mean-words-debug-sin-uniq"
    path_stem = "C:/f/.odxcaches/_modeldata/model.npz.best-ce-mean-words-debug-sin-uniq"
    encode_source = Graph.load(f"{path_stem}.encode_source.onnx",
                               inputs=['data_0', 'data_0_mask', 'data_0_posrange'])  # define the order of arguments
    decode_first = Graph.load(f"{path_stem}.decode_first.onnx",
                              inputs=['data_1_posrange', 'encoder_context_0', 'data_0_mask'],
                              outputs=['first_logits', 'first_decoder_state_0', 'first_decoder_state_1',
                                       'first_decoder_state_2', 'first_decoder_state_3', 'first_decoder_state_4',
                                       'first_decoder_state_5'])
    decode_next = Graph.load(f"{path_stem}.decode_next.onnx",
                             inputs=['prev_word', 'data_1_posrange', 'encoder_context_0', 'data_0_mask',
                                     'decoder_state_0', 'decoder_state_1', 'decoder_state_2', 'decoder_state_3',
                                     'decoder_state_4', 'decoder_state_5'],
                             outputs=['next_logits', 'next_decoder_state_0', 'next_decoder_state_1',
                                      'next_decoder_state_2', 'next_decoder_state_3', 'next_decoder_state_4',
                                      'next_decoder_state_5'])

    # --- greedy search for fixed length of 3

    if False:
        @Graph.trace(
            input_types =[_Ty.I(shape=['SOURCE_LENGTH'])],
            output_types=[_Ty.I(shape=[3])],
            outputs="Y")
        def greedy_search_fixed_length_3(X):
            ox = X.ox
            data_0 = X
            data_0_shape = data_0.shape()
            data_0_mask = ox.constant_of_shape(data_0_shape, value=1.0)
            seq_len = data_0_shape[-1]
            data_0_index_range = onnx_range(seq_len).cast(to=ox.float)
            data_0_index_range = ox.unsqueeze(data_0_index_range, axes=[1,2])  # [SOURCE_LENGTH, 1, 1]
            #max_len = seq_len * 3

            encoder_context_0 = encode_source(data_0=data_0, data_0_mask=data_0_mask,
                                              data_0_posrange=data_0_index_range)

            y_len_0 = ox.constant(value=0.0)
            logp, *out_decoder_states = decode_first(data_1_posrange=y_len_0,
                                                     encoder_context_0=encoder_context_0, data_0_mask=data_0_mask)

            # # !!!! logp[:, :, :, unk_id] = -1e8  # suppress <unk>, like Marian
            y_t = logp[0, 0, 0].argmax(axis=-1, keepdims=True)  # Concat fails with keepdims=False

            Y = [y_t]
            for iteration_count in range(1,3):
                pos = ox.constant(value=iteration_count) + 1
                data_1_posrange = pos.cast(to=1).unsqueeze(axes=[0, 1, 2])  # [1, 1, 1]
                logp, *out_decoder_states = decode_next(
                    prev_word=y_t, data_1_posrange=data_1_posrange,
                    encoder_context_0=encoder_context_0, data_0_mask=data_0_mask,
                    decoder_state_0=out_decoder_states[0], decoder_state_1=out_decoder_states[1],
                    decoder_state_2=out_decoder_states[2], decoder_state_3=out_decoder_states[3],
                    decoder_state_4=out_decoder_states[4], decoder_state_5=out_decoder_states[5])
                y_t = logp[0, 0, 0].argmax(axis=-1, keepdims=True)  # Concat fails with keepdims=False
                Y += [y_t]

            Y = ox.concat(Y, axis=0)  # note: y_t are rank-1 tensors, not scalars (ORT concat fails with scalars)
            return Y

        greedy_search_fixed_length_3.save("greedy3.onnx")

        Y = greedy_search_fixed_length_3(np.array([530, 4, 0], dtype=np.int64))
        # expected: [3421, 4, 0]
        print(Y.shape, Y)

    # --- full greedy search

    @Graph.trace(
        input_types=[_Ty.I(shape=['SOURCE_LENGTH']), _Ty.I([1])],
        output_types=[_Ty.I(shape=['TARGET_LENGTH'])],
        outputs="Y")
    def greedy_search(X, eos):
        ox = X.ox
        data_0 = X
        data_0_shape = data_0.shape()
        data_0_mask = ox.constant_of_shape(data_0_shape, value=1.0)
        seq_len = data_0_shape[-1]
        data_0_index_range = onnx_range(seq_len).cast(to=ox.float)
        data_0_index_range = ox.unsqueeze(data_0_index_range, axes=[1,2])
        max_len = seq_len * 3

        encoder_context_0 = encode_source(data_0=data_0, data_0_mask=data_0_mask,
                                          data_0_posrange=data_0_index_range)

        y_len_0 = ox.constant(value=0.0)
        logp, *out_decoder_states = decode_first(data_1_posrange=y_len_0,
                                                 encoder_context_0=encoder_context_0, data_0_mask=data_0_mask)

        # # !!!! logp[:, :, :, unk_id] = -1e8  # suppress <unk>, like Marian
        y_t = logp[0, 0, 0].argmax(axis=-1, keepdims=True)  # note: rank-1 tensor, not a scalar
        eos_token = eos + 0
        test_y_t = (y_t != eos_token)

        @Graph.trace(outputs=['ty_t', 'y_t_o', 'ods_0', 'ods_1', 'ods_2', 'ods_3', 'ods_4', 'ods_5', 'y_t_o2'],
                     output_types=[_Ty.b, _Ty.i] + [_Ty.f] * 6 + [_Ty.i],
                     input_types=[_Ty.I([1]), _Ty.b, _Ty.i] + [_Ty.f] * 6)
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
            y_t = logp[0, 0, 0].argmax(axis=-1, keepdims=True)
            test_y_t = (y_t != eos_token)
            return [test_y_t, y_t] + out_decoder_states + [y_t]

        # "Final N loop carried dependency values then K scan_outputs"
        ret_vals = ox.loop(max_len, test_y_t, loop_body,
                           inputs=[y_t] + out_decoder_states,
                           outputs=['gy_t_o', 'gods_0', 'gods_1', 'gods_2', 'gods_3', 'gods_4', 'gods_5', 'greedy_out'])
        y = ret_vals[-1]  # scan_output

        # we must prepend the very first token
        Y = ox.concat([ox.unsqueeze(y_t), y], axis=0)  # note: y_t are rank-1 tensors, not scalars (ORT concat fails with scalars)
        return ox.squeeze(Y, axes=[1])


    greedy_search.save("greedy.onnx")

    import time
    Y = greedy_search(np.array([274, 35, 52, 791, 59, 4060, 6, 2688, 2, 7744, 9, 2128, 7, 2, 4695, 9, 950, 2561, 3, 0], dtype=np.int64),
                      np.array([0], dtype=np.int64))
    print(Y.shape, Y)
    t_begin = time.time()
    np_input = np.array([274, 35, 52, 791, 59, 4060, 6, 2688, 2, 7744, 9, 2128, 7, 2, 4695, 9, 950, 2561, 3, 0], dtype=np.int64)
    for i in range(10):
        Y = greedy_search(np_input, np.array([0], dtype=np.int64))
    print("time elapsed: {}".format(time.time() - t_begin))

    # --- encoder only
    if False:
        # @BUGBUG: This last one kills the model checker. The two above work.
        @Graph.trace(
            input_types=[_Ty.I32(shape=['SOURCE_LENGTH']),
                         _Ty.F(shape=['SOURCE_LENGTH', 1, 1]),
                         _Ty.F(shape=['SOURCE_LENGTH', 1, 1])],
            output_types=[_Ty.F(shape=[1, 'SOURCE_LENGTH', 1, 512])],
            outputs="z")
        def h(a, b, c):
            return encode_source(a, b, c)


        model_path = "enc.onnx"
        print("Saving to:", model_path, flush=True)
        h.save(model_path)

        res = h(np.array([530, 4, 0], dtype=np.int32),
                np.array([[[1.0]], [[1.0]], [[1.0]]], dtype=np.float32),
                np.array([[[0.0]], [[1.0]], [[2.0]]], dtype=np.float32))

        print(res)
