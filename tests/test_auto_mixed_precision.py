import unittest
import numpy as np
import onnxruntime as _ort
import onnx
import copy
from onnxconverter_common.onnx_fx import Graph, OnnxOperatorBuilderX
from onnxconverter_common.onnx_fx import GraphFunctionType as _Ty
from onnxconverter_common.onnx_ex import get_maximum_opset_supported
from onnxconverter_common.auto_mixed_precision import auto_convert_mixed_precision


def _ort_inference(mdl, inputs):
    sess = _ort.InferenceSession(mdl.SerializeToString())
    return sess.run(None, inputs)


Graph.inference_runtime = _ort_inference
Graph.opset = 9
onnx_function = Graph.trace

@unittest.skipIf(get_maximum_opset_supported() < 9, "tests designed for ONNX opset 9 and greater")
@unittest.skipIf(not hasattr(onnx, "shape_inference"), "shape inference is required")
class AutoFloat16Test(unittest.TestCase):
    def test_auto_mixed_precision(self):
        @onnx_function(outputs=['z'],
                       input_types=(_Ty.F([1, 1, 6, 1])),
                       output_types=[_Ty.f])
        def transpose_n_matmul(x):
            ox = x.ox  # type: OnnxOperatorBuilderX
            wm = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).astype(np.float32).reshape([2, 6])
            b = ox.constant(value=wm)
            a = ox.transpose(x, perm=[0, 1, 3, 2])
            c = ox.transpose(b, perm=[1, 0])
            m = ox.matmul([a, c])
            m_large = ox.mul([m, ox.constant(value=np.array(100, np.float32))])
            m_xlarge = ox.mul([m_large, ox.constant(value=np.array(10, np.float32))])
            mr = ox.reshape([m_xlarge], desired_shape=[2])
            mr = ox.reshape([mr], desired_shape=[2])
            m_normal = ox.div([mr, ox.constant(value=np.array(999, np.float32))])
            return m_normal

        m1 = np.array([[2, 3], [4, 5], [6, 7]]).astype(np.float32).reshape([1, 1, 6, 1])
        expected = transpose_n_matmul(m1)
        model = transpose_n_matmul.to_model()

        def validate_fn(res, fp16res):
            return np.allclose(res[0], fp16res[0], rtol=0.01)

        f16model = auto_convert_mixed_precision(copy.deepcopy(model), {'x': m1}, validate_fn, keep_io_types=True)

        actual = _ort_inference(f16model, {'x': m1})
        self.assertTrue(np.allclose(expected, actual, rtol=0.01))

        f16model2 = auto_convert_mixed_precision(copy.deepcopy(model), {'x': m1}, rtol=0.01, keep_io_types=False)

        actual = _ort_inference(f16model2, {'x': m1.astype(np.float16)})
        self.assertTrue(np.allclose(expected, actual, rtol=0.01))


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(AutoFloat16Test)
    # suite.debug()
    unittest.TextTestRunner().run(suite)
