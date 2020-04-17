import unittest
import numpy as np
from onnx import helper
from onnxconverter_common.oopb import OnnxOperatorBuilder
from onnxconverter_common.container import ModelComponentContainer
from onnxconverter_common.topology import Topology, convert_topology
from onnxconverter_common.registration import register_converter


class _SimpleRawModelContainer(object):
    def __init__(self, inputs, outputs):
        self.input_names = inputs
        self.output_names = outputs


GRAPH_NODES = 6


def build_graph(oopb, inputs, outputs):
    # type: (OnnxOperatorBuilder, [], []) -> None
    mul_node = oopb.mul(inputs)
    sub_node = oopb.sub([mul_node] + [np.array([1.0, 2.0])])
    gemm = oopb.gemm(sub_node)
    add_const = helper.make_tensor('add_2_c', oopb.float, (2, 1), [3.0, 4.0])
    div1 = oopb.div(gemm, oopb.constant('add_2', add_const))
    oopb.add_node('Add',
                  [div1, ('add_3', oopb.float, np.array([3.0, 4.0]))],
                  outputs=outputs)


def create_conversion_topology(input_names, output_names):
    GRAPH_OPERATOR_NAME = '__test_graph__'
    raw_model = _SimpleRawModelContainer(input_names, output_names)

    def on_conversion(scope, operator, container):
        with OnnxOperatorBuilder(container, scope).as_default('node_bn') as oopb:
            build_graph(oopb, container.inputs, container.outputs)

    register_converter(GRAPH_OPERATOR_NAME, on_conversion, overwrite=True)
    topo = Topology(raw_model)
    top_level = topo.declare_scope('__root__')
    top_level.declare_local_operator(GRAPH_OPERATOR_NAME)

    return topo


class OnnxOpTestCase(unittest.TestCase):
    def setUp(self):
        self.inputs = ('input_0', 'input_1')
        self.outputs = ('output0',)

    def test_op_only(self):
        topo = Topology(_SimpleRawModelContainer(self.inputs, self.outputs))
        scope = topo.declare_scope('__ROOT__')
        container = ModelComponentContainer(target_opset=7)

        with OnnxOperatorBuilder(container, scope).as_default('node_bn') as oopb:
            build_graph(oopb, self.inputs, self.outputs)

            self.assertEqual(len(container.nodes), GRAPH_NODES)
            self.assertEqual(len(container.initializers), 2)
            self.assertTrue(container.nodes[0].name.startswith('node_bn'))

    def test_whole_topo(self):
        topo = create_conversion_topology(self.inputs, self.outputs)
        # a fake conversion to check the scope data correctness.
        oxml = convert_topology(topo, 'test', "doc_string", target_opset=7)
        self.assertIsNotNone(oxml)
        self.assertEqual(len(oxml.graph.node), GRAPH_NODES)
