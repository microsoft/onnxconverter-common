import unittest
from onnx import onnx_pb as onnx_proto
from onnxconverter_common.data_types import (
    BooleanTensorType,
    Complex64TensorType,
    Complex128TensorType,
    DoubleTensorType,
    DictionaryType,
    Int32TensorType,
    Int64Type,
    Int64TensorType,
    Int8TensorType,
    FloatType,
    FloatTensorType,
    SequenceType,
    StringType,
    StringTensorType,
    UInt8TensorType,
)


class TestDataTypes(unittest.TestCase):

    def common_test_tensor_type(self, data_type):
        dt = data_type((1, 5))
        self.assertIn(data_type.__name__, str(dt))
        assert dt._get_element_onnx_type() > 0
        assert isinstance(dt.to_onnx_type(), onnx_proto.TypeProto)

    def test_tensor_type(self):
        tensor_types = [
            BooleanTensorType,
            Complex64TensorType,
            Complex128TensorType,
            DoubleTensorType,
            Int32TensorType,
            Int64TensorType,
            Int8TensorType,
            FloatTensorType,
            StringTensorType,
            UInt8TensorType,
        ]
        seq_types = [
            DictionaryType,
            SequenceType,
        ]
        for dt in tensor_types:
            with self.subTest(data_type=dt):
                self.common_test_tensor_type(dt)

    def common_test_data_type(self, data_type):
        dt = data_type()
        self.assertIn(data_type.__name__, str(dt))
        assert isinstance(dt.to_onnx_type(), onnx_proto.TypeProto)

    def test_data_type(self):
        data_types = [
            Int64Type,
            FloatType,
            StringType,
        ]
        for dt in data_types:
            with self.subTest(data_type=dt):
                self.common_test_data_type(dt)

    def common_test_seq_type(self, data_type, dt):
        self.assertIn(data_type.__name__, str(dt))
        assert isinstance(dt.to_onnx_type(), onnx_proto.TypeProto)

    def test_sequence_type(self):
        self.common_test_seq_type(
            SequenceType, SequenceType(Int64Type()))
        self.common_test_seq_type(
            DictionaryType, DictionaryType(Int64Type(), Int64Type()))


if __name__ == '__main__':
    unittest.main()
