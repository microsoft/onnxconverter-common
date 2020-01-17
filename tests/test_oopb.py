import unittest
import numpy as np
from onnxconverter_common.oopb import OnnxOperatorBuilder
from onnxconverter_common.container import ModelComponentContainer, RawModelContainer
from onnxconverter_common.topology import Topology, convert_topology


class _SimpleRawModelContainer(RawModelContainer):
    def __init__(self):
        super(_SimpleRawModelContainer, self).__init__(None)

    @property
    def input_names(self):
        return ['input_0', 'input_1']

    @property
    def output_names(self):
        return ['m_output']


class OnnxOpTestCase(unittest.TestCase):
    def setUp(self):
        self.raw_model = _SimpleRawModelContainer()

    def test_apply_op(self):
        topo = Topology(self.raw_model)
        scope = topo.declare_scope('__ROOT__')
        container = ModelComponentContainer(target_opset=7)

        with OnnxOperatorBuilder(container, scope).as_default('node_bn') as oopb:
            mul_node = oopb.apply_mul(self.raw_model.input_names)
            sub_node = oopb.apply_sub([mul_node] + [np.array([1.0, 2.0])])
            output = oopb.add_node('Add',
                                   [sub_node, ('add_1', oopb.float, np.array([3.0, 4.0]))],
                                   outputs=self.raw_model.output_names)

        self.assertIsInstance(output, list)
        self.assertEqual(len(container.nodes), 3)
        self.assertEqual(len(container.initializers), 2)
        self.assertTrue(container.nodes[0].name.startswith('node_bn'))

        # a fake conversion to check the scope data correctness.
        oxml = convert_topology(topo, 'test', "doc_string", 7, None)
        self.assertIsNotNone(oxml)
        self.assertEqual(len(oxml.graph.node), 0)
