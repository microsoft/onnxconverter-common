import unittest
import sys
import numpy
from numpy.testing import assert_almost_equal
try:
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsRegressor
    import skl2onnx
    from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
        OnnxIdentity, OnnxAdd
    )
    from skl2onnx.common.data_types import FloatTensorType
    from skl2onnx.algebra.complex_functions import onnx_cdist
    from skl2onnx import to_onnx
    from onnxconverter_common.optim import (
        onnx_statistics, onnx_remove_node_identity
    )
    from onnxruntime import InferenceSession
    skip = False
except ImportError:
    # python 2 or skl2onnx <= 1.5.0
    skip = True
try:
    from onnxruntime.capi.onnxruntime_pybind11_state import Fail as OrtFail
except ImportError:
    OrtFail = RuntimeError


class TestOptimOnnxIdentity(unittest.TestCase):

    @unittest.skipIf(skip or sys.version_info[0] == 2,
                     reason="skl2onnx only python 3")
    def test_onnx_remove_identities(self):
        from skl2onnx.algebra.complex_functions import onnx_squareform_pdist
        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))
        cop = OnnxAdd(OnnxIdentity('input'), 'input')
        cdist = onnx_squareform_pdist(cop, dtype=numpy.float32)
        cop2 = OnnxIdentity(cdist, output_names=['cdist'])

        model_def = cop2.to_onnx(
            {'input': FloatTensorType()},
            outputs=[('cdist', FloatTensorType())])
        stats = onnx_statistics(model_def)
        self.assertIn('subgraphs', stats)
        self.assertGreater(stats['subgraphs'], 0)
        self.assertGreater(stats['op_Identity'], 1)

        new_model = onnx_remove_node_identity(model_def)
        stats2 = onnx_statistics(new_model)
        self.assertEqual(stats['subgraphs'], stats2['subgraphs'])
        assert stats2['op_Identity'] <= 2

        try:
            oinf1 = InferenceSession(model_def.SerializeToString())
        except (RuntimeError, OrtFail) as e:
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
            assert stats2['op_Identity'] <= 1
        except (RuntimeError, OrtFail) as e:
            if "Subgraph must have the shape set for all outputs" in str(e):
                # onnxruntime 1.6
                return
            raise e

    @unittest.skipIf(skip or sys.version_info[0] == 2,
                     reason="skl2onnx only python 3")
    def test_onnx_remove_identities2(self):
        from skl2onnx.algebra.complex_functions import onnx_squareform_pdist
        x = numpy.array([1, 2, 4, 5, 5, 4]).astype(
            numpy.float32).reshape((3, 2))
        cop = OnnxIdentity('input')
        cdist = onnx_squareform_pdist(cop, dtype=numpy.float32)
        cop2 = OnnxIdentity(cdist, output_names=['cdist'])

        model_def = cop2.to_onnx(
            {'input': FloatTensorType()},
            outputs=[('cdist', FloatTensorType())])
        stats = onnx_statistics(model_def)
        self.assertIn('subgraphs', stats)
        self.assertGreater(stats['subgraphs'], 0)
        self.assertGreater(stats['op_Identity'], 1)

        new_model = onnx_remove_node_identity(model_def)
        stats2 = onnx_statistics(new_model)
        self.assertEqual(stats['subgraphs'], stats2['subgraphs'])
        assert stats2['op_Identity'] <= 2

        try:
            oinf1 = InferenceSession(model_def.SerializeToString())
        except (RuntimeError, OrtFail) as e:
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
            self.assertLesser(stats2['op_Identity'], 1)
        except (RuntimeError, OrtFail) as e:
            if "Subgraph must have the shape set for all outputs" in str(e):
                # onnxruntime 1.6
                return
            raise e

    @unittest.skipIf(skip or sys.version_info[0] == 2,
                     reason="skl2onnx only python 3")
    def test_onnx_example_cdist_in_euclidean(self):
        x2 = numpy.array([1.1, 2.1, 4.01, 5.01, 5.001, 4.001, 0, 0]).astype(
            numpy.float32).reshape((4, 2))
        cop = OnnxAdd('input', 'input')
        cop2 = OnnxIdentity(onnx_cdist(cop, x2, dtype=numpy.float32, metric='euclidean'),
                            output_names=['cdist'])

        model_def = cop2.to_onnx(
            inputs=[('input', FloatTensorType([None, None]))],
            outputs=[('cdist', FloatTensorType())])

        new_model = onnx_remove_node_identity(model_def)
        stats = onnx_statistics(model_def)
        stats2 = onnx_statistics(new_model)
        self.assertTrue(stats.get('op_Identity', 0) in (2, 3))
        self.assertEqual(stats2.get('op_Identity', 0), 1)


if __name__ == "__main__":
    unittest.main()
