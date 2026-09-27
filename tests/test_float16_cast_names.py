import copy
import unittest

import numpy as np
import onnxruntime as ort
from onnx import TensorProto, checker, helper, numpy_helper

from onnxconverter_common import convert_float_to_float16


class TestFloat16CastNames(unittest.TestCase):
    def check_conversion(
        self,
        nodes,
        outputs,
        *,
        keep_io_types,
        block_list=None,
        initializers=(),
        extra_inputs=(),
    ):
        model = helper.make_model(
            helper.make_graph(
                nodes,
                "cast_names",
                [
                    helper.make_tensor_value_info("X", TensorProto.FLOAT, [3]),
                    *[
                        helper.make_tensor_value_info(name, TensorProto.FLOAT, [3])
                        for name in extra_inputs
                    ],
                ],
                [
                    helper.make_tensor_value_info(name, TensorProto.FLOAT, [3])
                    for name in outputs
                ],
                initializers,
            ),
            opset_imports=[helper.make_opsetid("", 13)],
            ir_version=8,
        )
        checker.check_model(model)
        inputs = {
            value.name: np.array([-2, 0, 3], dtype=np.float32)
            for value in model.graph.input
        }
        expected = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        ).run(None, inputs)

        converted = convert_float_to_float16(
            copy.deepcopy(model),
            keep_io_types=keep_io_types,
            op_block_list=block_list,
        )

        checker.check_model(converted)
        actual = ort.InferenceSession(
            converted.SerializeToString(), providers=["CPUExecutionProvider"]
        ).run(
            None,
            {
                name: value.astype(np.float32 if keep_io_types else np.float16)
                for name, value in inputs.items()
            },
        )
        for result, reference in zip(actual, expected):
            np.testing.assert_allclose(result, reference, rtol=1e-3, atol=1e-3)
            self.assertEqual(result.dtype, np.float32 if keep_io_types else np.float16)
        return converted

    def test_unnamed_input_consumers(self):
        self.check_conversion(
            [
                helper.make_node("Relu", ["X"], ["Y"]),
                helper.make_node("Sigmoid", ["X"], ["Z"]),
            ],
            ["Y", "Z"],
            keep_io_types=True,
        )

    def test_unnamed_blocked_nodes(self):
        for keep_io_types in (False, True):
            with self.subTest(keep_io_types=keep_io_types):
                self.check_conversion(
                    [
                        helper.make_node("Relu", ["X"], ["Y"]),
                        helper.make_node("Sigmoid", ["X"], ["Z"]),
                        helper.make_node("Add", ["Y", "Z"], ["W"], name="add"),
                    ],
                    ["W"],
                    keep_io_types=keep_io_types,
                    block_list=["Relu", "Sigmoid"],
                )

    def test_generated_names_do_not_replace_existing_values(self):
        for collision, blocked in (
            ("X_cast_to_relu", False),
            ("relu_cast_to_Y", False),
            ("relu_input_cast_0", True),
            ("relu_output_cast_0", True),
        ):
            for keep_io_types in (False, True):
                with self.subTest(collision=collision, keep_io_types=keep_io_types):
                    nodes = [
                        helper.make_node("Relu", ["X"], ["Y"], name="relu"),
                        helper.make_node("Add", ["X", collision], ["Z"], name="add"),
                    ]
                    if blocked:
                        nodes.append(
                            helper.make_node("Identity", ["Y"], ["R"], name="id")
                        )
                    self.check_conversion(
                        nodes,
                        ["R" if blocked else "Y", "Z"],
                        keep_io_types=keep_io_types,
                        block_list=["Relu"] if blocked else [],
                        initializers=[
                            numpy_helper.from_array(
                                np.array([1, 2, 3], dtype=np.float32), collision
                            )
                        ],
                    )

    def test_generated_names_do_not_duplicate_existing_node_names(self):
        for collision, blocked in (
            ("X_cast_to_relu", False),
            ("relu_cast_to_Y", False),
            ("relu_input_cast0", True),
            ("relu_output_cast0", True),
        ):
            with self.subTest(collision=collision):
                nodes = [
                    helper.make_node("Relu", ["X"], ["Y"], name="relu"),
                    helper.make_node("Sigmoid", ["X"], ["Z"], name=collision),
                ]
                if blocked:
                    nodes.append(helper.make_node("Identity", ["Y"], ["R"], name="id"))
                self.check_conversion(
                    nodes,
                    ["R" if blocked else "Y", "Z"],
                    keep_io_types=True,
                    block_list=["Relu"] if blocked else [],
                )

    def test_existing_graph_input_and_suffix_are_reserved(self):
        self.check_conversion(
            [
                helper.make_node("Relu", ["X"], ["Y"], name="relu"),
                helper.make_node(
                    "Add", ["X_cast_to_relu", "X_cast_to_relu_1"], ["Z"], name="add"
                ),
            ],
            ["Y", "Z"],
            keep_io_types=True,
            extra_inputs=["X_cast_to_relu", "X_cast_to_relu_1"],
        )

    def test_noncolliding_names_are_preserved(self):
        converted = self.check_conversion(
            [helper.make_node("Relu", ["X"], ["Y"], name="relu")],
            ["Y"],
            keep_io_types=True,
        )
        self.assertEqual(
            {node.name for node in converted.graph.node},
            {"relu", "X_cast_to_relu", "relu_cast_to_Y"},
        )

    def test_subgraph_values_are_reserved(self):
        branch = helper.make_graph(
            [helper.make_node("Relu", ["B"], ["X_cast_to_relu"], name="branch_relu")],
            "branch",
            [],
            [helper.make_tensor_value_info("X_cast_to_relu", TensorProto.FLOAT, [3])],
        )
        self.check_conversion(
            [
                helper.make_node("Relu", ["X"], ["Y"], name="relu"),
                helper.make_node(
                    "If",
                    ["condition"],
                    ["Z"],
                    name="if",
                    then_branch=branch,
                    else_branch=branch,
                ),
            ],
            ["Y", "Z"],
            keep_io_types=True,
            initializers=[
                numpy_helper.from_array(np.array(True), "condition"),
                numpy_helper.from_array(np.array([-1, 2, 3], dtype=np.float32), "B"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
