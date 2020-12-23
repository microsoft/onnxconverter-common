# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###########################################################################

import numpy as np


def indent(s):
    return "\n".join("    " + line for line in s.split("\n"))


class TracingObject:
    """
    Used by onnx2py to mock a module like numpy or onnx.helper and record calls on that module
    Ex:
        np = TracingObject("np")
        x = np.array(np.product([1, 2, 3]), np.int32)
        assert repr(x) == "np.array(np.product([1, 2, 3]), np.int32)"
    """
    def __init__(self, trace):
        self._trace = trace

    @staticmethod
    def from_repr(o):
        return TracingObject(TracingObject.get_repr(o))

    @staticmethod
    def get_repr(x):
        if isinstance(x, np.ndarray):
            return "np.array(%r, dtype=np.%s)" % (x.tolist(), x.dtype)
        if not isinstance(x, list):
            return repr(x)
        ls = [TracingObject.get_repr(o) for o in x]
        code = "[" + ", ".join(ls) + "]"
        if len(code) <= 200:
            return code
        return "[\n" + "".join(indent(s) + ",\n" for s in ls) + "]"

    def __getattr__(self, attr):
        return TracingObject(self._trace + "." + attr)

    def __call__(self, *args, **kwargs):
        arg_s = [TracingObject.get_repr(o) for o in args]
        arg_s += [k + "=" + TracingObject.get_repr(o) for k, o in kwargs.items()]
        trace = self._trace + "(" + ", ".join(arg_s) + ")"
        if len(trace) <= 200:
            return TracingObject(trace)
        return TracingObject(self._trace + "(\n" + "".join(indent(s) + ",\n" for s in arg_s) + ")")

    def __repr__(self):
        return self._trace
