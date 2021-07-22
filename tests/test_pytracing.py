import unittest
import numpy as np

from onnxconverter_common.pytracing import TracingObject

class TracingTests(unittest.TestCase):
    def test_tracing_simple(self):
        tracer = TracingObject("x")
        actual = repr(tracer.func_call([1, 'A', 3, tracer], tracer))
        expected = "x.func_call([1, 'A', 3, x], x)"
        self.assertEqual(actual, expected)

    def test_tracing_numpy(self):
        tracer = TracingObject("helper")
        x = np.array([1, 2, 3], dtype=np.int32)
        actual = repr(tracer.from_numpy(x))
        expected = "helper.from_numpy(np.array([1, 2, 3], dtype='int32'))"
        self.assertEqual(actual, expected)

if __name__ == '__main__':
    unittest.main()
