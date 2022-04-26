import unittest
import numpy as np
import os
import onnx
import onnxruntime as _ort
import sys

working_path = os.path.abspath(os.path.dirname(__file__))
tmp_path = os.path.join(working_path, 'temp')
data_path = os.path.join(working_path, 'data')

class Onnx2PyTests(unittest.TestCase):
    def tearDown(self):
        from onnxconverter_common.onnx2py import clear_directory
        for f in os.listdir(tmp_path):
            if f.endswith(".py"):
                os.remove(os.path.join(tmp_path, f))
                folder_path = os.path.join(tmp_path, f[:-3])
                if os.path.exists(folder_path):
                    clear_directory(folder_path)

    @unittest.skipIf(sys.version_info < (3, 6), "Requires onnx > 1.3.0")
    def test_onnx2py(self):
        from onnxconverter_common.onnx2py import convert
        model_name = 'test_model_1_no_opt'
        onnx_model = onnx.load(os.path.join(data_path, model_name + '.onnx'))
        sess1 = _ort.InferenceSession(onnx_model.SerializeToString())
        np.random.seed(42)
        data = np.random.random_sample(size=(1, 1, 512)).astype(np.float32)
        expected = sess1.run(["conv1d_1"], {"input_1": data})

        os.makedirs(tmp_path, exist_ok=True)
        out_path = os.path.join(tmp_path, model_name + '.py')
        convert(onnx_model, out_path)
        self.assertTrue(os.path.exists(out_path))

        sys.path.append(tmp_path)
        from test_model_1_no_opt import model

        sess2 = _ort.InferenceSession(model.SerializeToString())
        actual = sess2.run(["conv1d_1"], {"input_1": data})

        np.testing.assert_allclose(expected[0], actual[0])

if __name__ == '__main__':
    unittest.main()
