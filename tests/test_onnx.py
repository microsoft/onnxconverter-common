import os
import onnx
import unittest
import packaging.version as pv
from onnxconverter_common.data_types import FloatTensorType
from onnxconverter_common import set_denotation


class TestTypes(unittest.TestCase):

    def test_to_onnx_type(self):
        dt = FloatTensorType((1, 5))
        assert str(dt) == 'FloatTensorType(shape=(1, 5))'
        onx = dt.to_onnx_type()
        assert "dim_value: 5" in str(onx)
        tt = onx.tensor_type
        assert "dim_value: 5" in str(tt)
        assert tt.elem_type == 1
        o = onx.sequence_type
        assert str(o) == ""

    @unittest.skipIf(pv.Version(onnx.__version__) < pv.Version('1.2.1'),
                     "not supported in this ONNX version")
    def test_set_denotation(self):
        this = os.path.dirname(__file__)
        onnx_file = os.path.join(this, "coreml_OneHotEncoder_BikeSharing.onnx")
        onnx_model = onnx.load_model(onnx_file)
        set_denotation(onnx_model, "1", "IMAGE", onnx.defs.onnx_opset_version(), dimension_denotation=["DATA_FEATURE"])
        self.assertEqual(onnx_model.graph.input[0].type.denotation, "IMAGE")
        self.assertEqual(onnx_model.graph.input[0].type.tensor_type.shape.dim[0].denotation, "DATA_FEATURE")


if __name__ == '__main__':
    unittest.main()
