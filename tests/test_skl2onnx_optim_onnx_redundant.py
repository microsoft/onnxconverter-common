import sys
import unittest
from distutils.version import StrictVersion
import numpy
from numpy.testing import assert_almost_equal
try:
    import skl2onnx
    from skl2onnx.algebra.onnx_ops import (  # noqa
        OnnxAdd, OnnxMul, OnnxSub, OnnxIdentity
    )
    from skl2onnx.common.data_types import FloatTensorType
    from onnxconverter_common.optim import (
        onnx_statistics, onnx_remove_node_redundant, onnx_remove_node
    )
    from onnxruntime import InferenceSession
    skip = False
except ImportError:
    # python 2 or skl2onnx <= 1.5.0
    skip = True


class TestOptimOnnxRedundant(unittest.TestCase):

    @unittest.skipIf(skip or sys.version_info[0] == 2,
                     reason="skl2onnx only python 3")
    def test_onnx_remove_redundant(self):
        dtype = numpy.float32
        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))
        cop = OnnxAdd('X', numpy.array([1], dtype=dtype))
        cop2 = OnnxAdd('X', numpy.array([1], dtype=dtype))
        cop3 = OnnxAdd('X', numpy.array([2], dtype=dtype))
        cop4 = OnnxSub(OnnxMul(cop, cop3), cop2, output_names=['final'])
        model_def = cop4.to_onnx({'X': x})
        stats = onnx_statistics(model_def)
        c1 = model_def.SerializeToString()
        new_model = onnx_remove_node_redundant(model_def, max_hash_size=10)
        c2 = model_def.SerializeToString()
        self.assertEqual(c1, c2)
        stats2 = onnx_statistics(model_def)
        stats3 = onnx_statistics(new_model)
        self.assertEqual(stats['ninits'], 3)
        self.assertEqual(stats2['ninits'], 3)
        self.assertEqual(stats3['ninits'], 2)
        self.assertEqual(stats2['nnodes'], 5)
        self.assertEqual(stats3['nnodes'], 4)
        oinf1 = InferenceSession(model_def.SerializeToString())
        y1 = oinf1.run(None, {'X': x})

        oinf2 = InferenceSession(new_model.SerializeToString())
        y2 = oinf2.run(None, {'X': x})
        assert_almost_equal(y1[0], y2[0])

    @unittest.skipIf(skip or sys.version_info[0] == 2,
                     reason="skl2onnx only python 3")
    def test_onnx_remove_two_outputs(self):
        dtype = numpy.float32
        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))
        cop = OnnxAdd('X', numpy.array([1], dtype=dtype))
        cop2 = OnnxAdd('X', numpy.array(
            [1], dtype=dtype), output_names=['keep'])
        cop3 = OnnxAdd('X', numpy.array([2], dtype=dtype))
        cop4 = OnnxSub(OnnxMul(cop, cop3), cop2, output_names=['final'])
        model_def = cop4.to_onnx({'X': x},
                                 outputs=[('keep', FloatTensorType([None, 2])),
                                          ('final', FloatTensorType([None, 2]))])
        c1 = model_def.SerializeToString()
        self.assertEqual(len(model_def.graph.output), 2)
        c2 = model_def.SerializeToString()
        self.assertEqual(c1, c2)
        stats = onnx_statistics(model_def)
        new_model = onnx_remove_node_redundant(model_def, max_hash_size=10)
        stats2 = onnx_statistics(model_def)
        stats3 = onnx_statistics(new_model)
        self.assertEqual(stats['ninits'], 3)
        self.assertEqual(stats2['ninits'], 3)
        self.assertEqual(stats3['ninits'], 2)
        self.assertEqual(stats2['nnodes'], 5)
        self.assertEqual(stats3['nnodes'], 4)
        oinf1 = InferenceSession(model_def.SerializeToString())
        y1 = oinf1.run(None, {'X': x})

        oinf2 = InferenceSession(new_model.SerializeToString())
        y2 = oinf2.run(None, {'X': x})
        assert_almost_equal(y1[0], y2[0])
        assert_almost_equal(y1[1], y2[1])

    @unittest.skipIf(skip or sys.version_info[0] == 2,
                     reason="skl2onnx only python 3")
    def test_onnx_remove_redundant_subgraphs(self):
        from skl2onnx.algebra.complex_functions import onnx_squareform_pdist
        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))
        cop = OnnxAdd(OnnxIdentity('input'), 'input')
        cdist = onnx_squareform_pdist(cop, dtype=numpy.float32)
        cdist2 = onnx_squareform_pdist(cop, dtype=numpy.float32)
        cop2 = OnnxAdd(cdist, cdist2, output_names=['cdist'])

        model_def = cop2.to_onnx(
            {'input': FloatTensorType()},
            outputs=[('cdist', FloatTensorType())])
        c1 = model_def.SerializeToString()
        stats = onnx_statistics(model_def)
        c2 = model_def.SerializeToString()
        self.assertEqual(c1, c2)
        self.assertIn('subgraphs', stats)
        self.assertGreater(stats['subgraphs'], 1)
        self.assertGreater(stats['op_Identity'], 2)

        new_model = onnx_remove_node_redundant(model_def)
        stats2 = onnx_statistics(new_model)
        self.assertEqual(stats['subgraphs'], 2)
        self.assertEqual(stats2['subgraphs'], 1)

        new_model = onnx_remove_node_redundant(model_def)
        stats3 = onnx_statistics(new_model)
        self.assertEqual(stats2, stats3)

        new_model = onnx_remove_node(model_def)
        stats3 = onnx_statistics(new_model)
        self.assertLess(stats3['size'], stats2['size'])
        self.assertLess(stats3['nnodes'], stats2['nnodes'])
        self.assertLess(stats3['op_Identity'], stats2['op_Identity'])

        try:
            oinf1 = InferenceSession(model_def.SerializeToString())
        except RuntimeError as e:
            if 'NOT_IMPLEMENTED' in str(e):
                return
            if 'not placed on any Execution Provider' in str(e):
                return
            raise e
        oinf2 = InferenceSession(new_model.SerializeToString())
        
        try:
            y1 = oinf1.run(None, {'input': x})[0]
            y2 = oinf2.run(None, {'input': x})[0]
            assert_almost_equal(y1, y2)
        except RuntimeError as e:
            if "Subgraph must have the shape set for all outputs" in str(e):
                # onnxruntime 1.6
                return
            raise e


if __name__ == "__main__":
    unittest.main()
