import unittest
import numpy as np

try:
    import onnxmltools
    import onnxmltools.convert.lightgbm as xmlt
except ImportError:
    onnxmltools = None

try:
    import lightgbm
except ImportError:
    lightgbm = None

from onnxconverter_common.data_types import FloatTensorType


class OnnxmltoolsTestCase(unittest.TestCase):

    @unittest.skipIf(onnxmltools is None or lightgbm is None,
                     reason="missing dependencies")
    def test_lightgbm(self):
        X = [[0, 1], [1, 1], [2, 0], [1, 2]]
        X = np.array(X, dtype=np.float32)
        y = [0, 1, 0, 1]
        model = lightgbm.LGBMClassifier(n_estimators=3, min_child_samples=1)
        model.fit(X, y)
        onx = xmlt.convert(
            model, 'dummy', initial_types=[('X', FloatTensorType([None, X.shape[1]]))],
            target_opset=9)
        assert "ir_version: 4" in str(onx).lower()


if __name__ == '__main__':
    unittest.main()
