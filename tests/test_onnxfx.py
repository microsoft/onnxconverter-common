import unittest
import numpy as np
import onnxruntime as _ort
import packaging.version as pv
from onnxconverter_common.onnx_fx import Graph, OnnxOperatorBuilderX
from onnxconverter_common.onnx_fx import GraphFunctionType as _Ty
from onnxconverter_common.onnx_ex import get_maximum_opset_supported
from onnxconverter_common.optimizer import optimize_onnx_model


def _ort_inference(mdl, inputs):
    sess = _ort.InferenceSession(mdl.SerializeToString())
    return sess.run(None, inputs)


Graph.inference_runtime = _ort_inference
Graph.opset = 9
onnx_function = Graph.trace


@unittest.skipIf(get_maximum_opset_supported() < 9, "onnx_fx only supports ONNX opset 9 and greater")
class ONNXFunctionTest(unittest.TestCase):
    # this works, and the exported graph is usable:
    def test_core(self):
        @onnx_function
        def f(x, y):
            return x + y

        @onnx_function
        def g(x, y):
            return x.ox.abs(f(x, y) + 1.0)

        self.assertTrue(
            np.allclose(g([2.0], [-5.0]), np.array([2.0])))

    def test_loop(self):
        @onnx_function(outputs=['y1', 'y2', 'y3', 'y4'],
                       input_types=[_Ty.I([1])],
                       output_types=[_Ty.F([None]), _Ty.F([None]), _Ty.F([None, 1]), _Ty.F([None, 1])])
        def loop_test(len):
            ox = len.ox
            s_len = ox.squeeze(len, axes=[0])
            is_true = ox.constant(value=True)

            @onnx_function(outputs=['c_o', 'i_o', 'j_o', 'all_i', 'all_j'],
                           output_types=[_Ty.b, _Ty.f, _Ty.f, _Ty.f, _Ty.f],
                           input_types=[_Ty.I([1]), _Ty.b, _Ty.F([1]), _Ty.F([1])])
            def range_body(iter_n, cond, i, j):
                return (is_true,
                        i + i.ox.constant(value=1.0), j + 2.0, i, j)

            one_c = ox.constant(value=-1.0)
            y1, y2, y3, y4 = ox.loop(s_len, is_true, range_body, inputs=[one_c, one_c],
                                     outputs=['y1_o', 'y2_o', 'y3_o', 'y4_o'])
            return y1, y2, y3, y4

        self.assertEqual(
            loop_test(np.array([16], dtype=np.int64))[2][4], 3.0)

    @unittest.skipIf(pv.Version(_ort.__version__.split('-')[0]) < pv.Version("1.4.0"),
                     "onnxruntime fixed the issue in matmul since 1.4.0")
    def test_matmul_opt(self):
        @onnx_function(outputs=['z'],
                       input_types=(_Ty.F([1, 1, 6, 1])),
                       output_types=[_Ty.f])
        def transpose_n_matmul(x):
            ox = x.ox  # type: OnnxOperatorBuilderX
            wm = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).astype(np.float32).reshape([2, 6])
            b = ox.constant(value=wm)
            a = ox.transpose(x, perm=[0, 1, 3, 2])
            c = ox.transpose(b, perm=[1, 0])
            return ox.matmul([a, c])

        m1 = np.array([[2, 3], [4, 5], [6, 7]]).astype(np.float32).reshape([1, 1, 6, 1])
        expected = transpose_n_matmul(m1)
        opted = optimize_onnx_model(transpose_n_matmul.to_model())
        actual = _ort_inference(opted, {'x': m1})
        self.assertTrue(np.allclose(expected, actual), "The result mismatch")


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(ONNXFunctionTest)
    # suite.debug()
    unittest.TextTestRunner().run(suite)
