import unittest
import os
import copy
import numpy as np
import onnx
import onnxruntime as _ort
import onnxmltools
import packaging.version as pv
from onnxconverter_common.onnx_fx import Graph, OnnxOperatorBuilderX
from onnxconverter_common.onnx_fx import GraphFunctionType as _Ty
from onnxconverter_common.onnx_ex import get_maximum_opset_supported
from onnxconverter_common.optimizer import optimize_onnx_model
from onnxconverter_common.float16 import convert_float_to_float16
from onnxconverter_common.float16 import convert_float_to_float16_old
from onnxconverter_common.float16 import convert_np_to_float16


def _ort_inference(mdl, inputs):
    sess = _ort.InferenceSession(mdl.SerializeToString())
    return sess.run(None, inputs)


Graph.inference_runtime = _ort_inference
Graph.opset = 9
onnx_function = Graph.trace

@unittest.skipIf(pv.Version(onnx.__version__) <= pv.Version('1.8.0'), "test for ONNX 1.8 and above")
@unittest.skipIf(get_maximum_opset_supported() < 9, "tests designed for ONNX opset 9 and greater")
class ONNXFloat16Test(unittest.TestCase):
    def test_float16(self):
        @onnx_function(outputs=['z'],
                       input_types=(_Ty.F([1, 1, 6, 1])),
                       output_types=[_Ty.f])
        def transpose_n_matmul(x):
            ox = x.ox  # type: OnnxOperatorBuilderX
            wm = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).astype(np.float32).reshape([2, 6])
            b = ox.constant(value=wm)
            a = ox.transpose(x, perm=[0, 1, 3, 2])
            c = ox.transpose(b, perm=[1, 0])
            d = ox.matmul([a, c])
            return ox.min(d)

        m1 = np.array([[2, 3], [4, 5], [6, 7]]).astype(np.float32).reshape([1, 1, 6, 1])
        expected = transpose_n_matmul(m1)
        model = transpose_n_matmul.to_model()
        actual = _ort_inference(model, {'x': m1})
        self.assertTrue(np.allclose(expected, actual))

        f16model = convert_float_to_float16(copy.deepcopy(model), is_io_fp32=False)
        actual = _ort_inference(f16model, {'x': m1.astype(np.float16)})
        self.assertTrue(np.allclose(expected, actual))

        f16model2 = convert_float_to_float16(copy.deepcopy(model), is_io_fp32=True)
        actual = _ort_inference(f16model2, {'x': m1})
        self.assertTrue(np.allclose(expected, actual))

    def test_float16_with_loop(self):
        @onnx_function(outputs=['y1', 'y2'],
                       input_types=[_Ty.F([None, None])],
                       output_types=[_Ty.F([None]), _Ty.F([None, None])])
        def loop_test(data):
            ox = data.ox
            shape = ox.shape(data)
            dim_0 = ox.gather([shape, ox.constant(value=0)], axis=0)
            dim_1 = ox.gather([shape, ox.constant(value=np.array([1], dtype=np.int64))], axis=0)
            zeros = ox.constant_of_shape(dim_1, value=0.0)
            is_true = ox.constant(value=True)

            @onnx_function(outputs=['c_o', 'total_o', 'scan_o'],
                           output_types=[_Ty.b, _Ty.F([None]), _Ty.F([None])],
                           input_types=[_Ty.I([1]), _Ty.b, _Ty.F([None])])
            def range_body(iter_n, cond, total):
                ox = iter_n.ox
                iter_scalar = ox.squeeze(iter_n, axes=[0])
                col = ox.gather([data, iter_scalar], axis=0)
                total = ox.add([total, col])
                return (is_true, total, total)

            final_total, scan_res = ox.loop(dim_0, is_true, range_body, inputs=[zeros],
                                   outputs=['final_total', 'scan_res'])
            return final_total, scan_res

        m1 = np.array([[2, 3], [4, 5], [6, 7]], dtype=np.float32)
        expected_res = loop_test(m1)

        model = loop_test.to_model()
        f16model1 = convert_float_to_float16(copy.deepcopy(model), is_io_fp32=False)
        actual_res = _ort_inference(f16model1, {'data': m1.astype(np.float16)})
        for expected, actual in zip(expected_res, actual_res):
            self.assertTrue(np.allclose(expected, actual))
            self.assertTrue(actual.dtype == np.float16)

        f16model2 = convert_float_to_float16(copy.deepcopy(model), is_io_fp32=True)
        actual_res2 = _ort_inference(f16model2, {'data': m1})
        for expected, actual2 in zip(expected_res, actual_res2):
            self.assertTrue(np.allclose(expected, actual2))
            self.assertTrue(actual2.dtype == np.float32)

    def test_convert_to_float16(self):
        model32_name = "image_classifier32.onnx"
        working_path = os.path.abspath(os.path.dirname(__file__))
        data_path = os.path.join(working_path, 'data')
        model_path = os.path.join(data_path, model32_name)
        onnx_model32 = onnxmltools.utils.load_model(model_path)
        input_x = np.random.rand(1, 3, 32, 32).astype(np.float32)
        output_32 = _ort_inference(onnx_model32, {'modelInput': input_x})

        onnx_model16_1 = convert_float_to_float16(onnx_model32, is_io_fp32=False)
        output_16_1 = _ort_inference(onnx_model16_1, {'modelInput': input_x.astype(np.float16)})
        self.assertTrue(np.allclose(output_16_1, output_32, atol=1e-2))

        onnx_model16_2 = convert_float_to_float16(onnx_model32, is_io_fp32=True)
        output_16_2 = _ort_inference(onnx_model16_2, {'modelInput': input_x})
        self.assertTrue(np.allclose(output_16_2, output_32, atol=1e-2))

    def test_convert_to_float16_with_truncated(self):
        np_array = np.array([1e-10, -2.0, 15, -1e-9, 65536.1, -100000])
        convert_np_to_float16(np_array)

    @unittest.skipIf(pv.Version(onnx.__version__) == pv.Version('1.9.0'), "ONNX 1.9 has different Optype behavior for Max operator")
    def test_convert_to_float16_with_subgraph(self):
        model32_name = "test_subgraph.onnx"
        working_path = os.path.abspath(os.path.dirname(__file__))
        data_path = os.path.join(working_path, 'data')
        model_path = os.path.join(data_path, model32_name)
        onnx_model32 = onnxmltools.utils.load_model(model_path)
        x = np.array([1.0], dtype=np.float32)
        y = np.array([2.0], dtype=np.float32)
        output_32 = _ort_inference(onnx_model32, {"x":x, "y":y})

        onnx_model16_1 = convert_float_to_float16(onnx_model32, is_io_fp32=True)
        actual = _ort_inference(onnx_model16_1, {"x":x, "y":y})
        self.assertTrue(np.allclose(actual, output_32, atol=1e-2))
        self.assertTrue(actual[0].dtype == np.float32)

        onnx_model16_2 = convert_float_to_float16(onnx_model32, is_io_fp32=False)
        actual = _ort_inference(onnx_model16_2, {"x": x.astype(np.float16), "y": y.astype(np.float16)})
        self.assertTrue(np.allclose(actual, output_32, atol=1e-2))
        self.assertTrue(actual[0].dtype == np.float16)


    # def test_auto_mixed_precision(self):
    #     @onnx_function(outputs=['z'],
    #                    input_types=(_Ty.F([1, 1, 6, 1])),
    #                    output_types=[_Ty.f])
    #     def transpose_n_matmul(x):
    #         ox = x.ox  # type: OnnxOperatorBuilderX
    #         wm = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).astype(np.float32).reshape([2, 6])
    #         b = ox.constant(value=wm)
    #         a = ox.transpose(x, perm=[0, 1, 3, 2])
    #         c = ox.transpose(b, perm=[1, 0])
    #         m = ox.matmul([a, c])
    #         m_large = ox.mul([m, ox.constant(value=np.array(100, np.float32))])
    #         m_xlarge = ox.mul([m_large, ox.constant(value=np.array(10, np.float32))])
    #         mr = ox.reshape([m_xlarge], desired_shape=[2])
    #         mr = ox.reshape([mr], desired_shape=[2])
    #         m_normal = ox.div([mr, ox.constant(value=np.array(999, np.float32))])
    #         return m_normal

    #     m1 = np.array([[2, 3], [4, 5], [6, 7]]).astype(np.float32).reshape([1, 1, 6, 1])
    #     expected = transpose_n_matmul(m1)
    #     model32 = transpose_n_matmul.to_model()
    #     onnx.save_model(model32, "d:/f32.onnx")
    #     model16_1 = convert_float_to_float16(model32, is_io_fp32=False)
    #     onnx.save_model(model16_1, "d:/f16_io16.onnx")
    #     actual = _ort_inference(model16_1, {"x": m1.astype(np.float16)})
    #     self.assertTrue(np.allclose(actual, expected, atol=1e-2))
    #     model16_2 = convert_float_to_float16(model32, is_io_fp32=True)
    #     onnx.save_model(model16_2, "d:/f16_io32.onnx")
    #     actual = _ort_inference(model16_2, {"x": m1})
    #     self.assertTrue(np.allclose(actual, expected, atol=1e-2))


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(ONNXFloat16Test)
    # suite.debug()
    unittest.TextTestRunner().run(suite)
