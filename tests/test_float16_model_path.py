import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, checker, helper, numpy_helper

from onnxconverter_common import convert_float_to_float16_model_path


class TestFloat16ModelPath(unittest.TestCase):
    @staticmethod
    def make_model():
        return helper.make_model(
            helper.make_graph(
                [helper.make_node("Add", ["X", "B"], ["Y"], name="add")],
                "float16_model_path",
                [helper.make_tensor_value_info("X", TensorProto.FLOAT, [3])],
                [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3])],
                [numpy_helper.from_array(np.array([1, 2, 3], dtype=np.float32), "B")],
            ),
            opset_imports=[helper.make_opsetid("", 13)],
            ir_version=8,
        )

    def test_convert_model_path(self):
        for external_data in (False, True):
            for keep_io_types in (False, True):
                with self.subTest(
                    external_data=external_data, keep_io_types=keep_io_types
                ), tempfile.TemporaryDirectory() as directory:
                    model_path = Path(directory) / "model.onnx"
                    model = self.make_model()
                    if external_data:
                        (Path(directory) / "weights").mkdir()
                        onnx.save_model(
                            model,
                            model_path,
                            save_as_external_data=True,
                            all_tensors_to_one_file=True,
                            location="weights/data.bin",
                            size_threshold=0,
                        )
                    else:
                        onnx.save_model(model, model_path)
                    original_files = {
                        path: path.read_bytes()
                        for path in Path(directory).rglob("*")
                        if path.is_file()
                    }

                    converted = convert_float_to_float16_model_path(
                        str(model_path), keep_io_types=keep_io_types
                    )

                    checker.check_model(converted)
                    dtype = np.float32 if keep_io_types else np.float16
                    inputs = np.array([-2, 0, 3], dtype=dtype)
                    output = ort.InferenceSession(
                        converted.SerializeToString(),
                        providers=["CPUExecutionProvider"],
                    ).run(None, {"X": inputs})[0]
                    np.testing.assert_array_equal(output, [-1, 2, 6])
                    self.assertEqual(output.dtype, dtype)
                    self.assertEqual(
                        converted.graph.initializer[0].data_type, TensorProto.FLOAT16
                    )
                    self.assertEqual(
                        {
                            path: path.read_bytes()
                            for path in Path(directory).rglob("*")
                            if path.is_file()
                        },
                        original_files,
                    )

    def test_cleanup_after_shape_inference_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "invalid.onnx"
            model_path.write_bytes(b"not an ONNX model")

            with self.assertRaises(checker.ValidationError):
                convert_float_to_float16_model_path(str(model_path))

            self.assertEqual(list(Path(directory).iterdir()), [model_path])
            self.assertEqual(model_path.read_bytes(), b"not an ONNX model")

    def test_cleanup_after_load_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.onnx"
            onnx.save_model(self.make_model(), model_path)

            with patch(
                "onnx.load", side_effect=OSError("failed to load inferred model")
            ):
                with self.assertRaisesRegex(OSError, "failed to load inferred model"):
                    convert_float_to_float16_model_path(str(model_path))

            self.assertEqual(list(Path(directory).iterdir()), [model_path])


if __name__ == "__main__":
    unittest.main()
