import unittest
import numpy as np
import onnxruntime as _ort
import onnx
import os
import packaging.version as pv
from onnxconverter_common.onnx_fx import Graph
from onnxconverter_common.onnx_ex import get_maximum_opset_supported
from onnxconverter_common.auto_mixed_precision_model_path import auto_convert_mixed_precision_model_path


def _ort_inference(model_path, inputs):
    sess = _ort.InferenceSession(model_path)
    return sess.run(None, inputs)


Graph.inference_runtime = _ort_inference
Graph.opset = 9
onnx_function = Graph.trace

@unittest.skipIf(pv.Version(onnx.__version__) <= pv.Version('1.8.0'), "test for ONNX 1.8 and above")
@unittest.skipIf(get_maximum_opset_supported() < 9, "tests designed for ONNX opset 9 and greater")
@unittest.skipIf(not hasattr(onnx, "shape_inference"), "shape inference is required")
class AutoFloat16Test(unittest.TestCase):
    def test_auto_mixed_precision_model_path(self):
        model32_name = "image_classifier32.onnx"
        working_path = os.path.abspath(os.path.dirname(__file__))
        data_path = os.path.join(working_path, 'data')
        model32_path = os.path.join(data_path, model32_name)

        input_x = np.random.rand(1, 3, 32, 32).astype(np.float32)
        expected = _ort_inference(model32_path, {'modelInput': input_x})

        model16_name = "image_classifier16.onnx"
        model16_path = os.path.join(data_path, model16_name)
        auto_convert_mixed_precision_model_path(
            model32_path, {'modelInput': input_x},
            model16_path, ['CPUExecutionProvider'],
            rtol=1e-2, 
            keep_io_types=True)
        actual = _ort_inference(model16_path, {'modelInput': input_x.astype(np.float32)})
        self.assertTrue(np.allclose(expected, actual, rtol=0.01))


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(AutoFloat16Test)
    # suite.debug()
    unittest.TextTestRunner().run(suite)
