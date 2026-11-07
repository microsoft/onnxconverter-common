import copy
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator

from onnxconverter_common.optimizer import optimize_onnx_model


class OptimizerDomainTestCase(unittest.TestCase):
    @staticmethod
    def make_model(domains):
        functions = [
            helper.make_function(
                domain,
                "Transform",
                ["X"],
                ["Y"],
                [helper.make_node(op_type, ["X"], ["Y"])],
                [helper.make_opsetid("", 13)],
            )
            for domain, op_type in [("test.neg", "Neg"), ("test.abs", "Abs")]
        ]
        nodes = [helper.make_node("Relu", ["X"], ["shared"], name="shared")]
        nodes.extend(
            helper.make_node(
                "Transform", ["shared"], [f"branch_{i}"], domain=domain, name=f"op_{i}"
            )
            for i, domain in enumerate(domains)
        )
        nodes.append(helper.make_node("Add", ["branch_0", "branch_1"], ["Y"]))
        graph = helper.make_graph(
            nodes,
            "operator_domains",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, ["N", 3])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["N", 3])],
        )
        return helper.make_model(
            graph,
            functions=functions,
            opset_imports=[
                helper.make_opsetid("", 13),
                helper.make_opsetid("test.neg", 1),
                helper.make_opsetid("test.abs", 1),
            ],
        )

    def check_optimization(self, domains):
        model = self.make_model(domains)
        onnx.checker.check_model(model, full_check=True)
        data = np.array([[-2, 0, 3], [1, -4, 5]], dtype=np.float32)
        expected = ReferenceEvaluator(model).run(None, {"X": data})[0]
        optimized = optimize_onnx_model(copy.deepcopy(model))
        onnx.checker.check_model(optimized, full_check=True)
        inferred = onnx.shape_inference.infer_shapes(optimized, strict_mode=True)
        self.assertEqual(inferred.graph.input, model.graph.input)
        self.assertEqual(inferred.graph.output, model.graph.output)
        actual = ReferenceEvaluator(optimized).run(None, {"X": data})[0]
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.shape, expected.shape)
        np.testing.assert_array_equal(actual, expected)
        return [node for node in optimized.graph.node if node.op_type == "Transform"]

    def test_common_sequence_preserves_distinct_domains(self):
        for domains in [("test.neg", "test.abs"), ("test.abs", "test.neg")]:
            with self.subTest(domains=domains):
                nodes = self.check_optimization(domains)
                self.assertCountEqual([node.domain for node in nodes], domains)

    def test_common_sequence_merges_matching_domains(self):
        for domain in ["test.neg", "test.abs"]:
            with self.subTest(domain=domain):
                nodes = self.check_optimization([domain, domain])
                self.assertEqual(len(nodes), 1)
                self.assertEqual(nodes[0].domain, domain)


if __name__ == "__main__":
    unittest.main()
