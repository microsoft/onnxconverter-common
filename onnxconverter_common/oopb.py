# coding=utf-8
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################

import functools
import numpy as np
from onnx import onnx_pb as onnx_proto
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE
from . import onnx_ops


class _OperatorNameContext:
    def __init__(self, oopb, basename):
        self.basename = basename
        self.oopb = oopb

    def __enter__(self):
        assert self.oopb.basename is None, "The previous context doesn't quit"
        self.oopb.basename = self.basename
        return self.oopb

    def __exit__(self, type, value, traceback):
        self.oopb.basename = None


class OnnxOperatorBuilder:
    def __init__(self, container, scope):
        self._container = container
        self._scope = scope
        self.basename = None
        self.int32 = onnx_proto.TensorProto.INT32
        self.int64 = onnx_proto.TensorProto.INT64
        self.float = onnx_proto.TensorProto.FLOAT
        self.float16 = onnx_proto.TensorProto.FLOAT16
        self.double = onnx_proto.TensorProto.DOUBLE
        self.bool = onnx_proto.TensorProto.BOOL

        apply_operations = onnx_ops.__dict__
        for k_, m_ in apply_operations.items():
            if k_.startswith("apply_") and callable(m_):
                setattr(self, k_, functools.partial(self.apply_op, m_))

    def as_default(self, basename):
        return _OperatorNameContext(self, basename)

    def _process_inputs(self, inputs, name):
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        ox_inputs = []
        for i_ in inputs:
            ox_n = i_
            if isinstance(i_, np.ndarray):
                ox_n = self._scope.get_unique_variable_name(name + '_i')
                self._container.add_initializer(
                    ox_n,
                    NP_TYPE_TO_TENSOR_TYPE[i_.dtype],
                    i_.shape,
                    i_.flatten()
                )
            elif isinstance(i_, (tuple, list)):
                ox_n = self._scope.get_unique_variable_name(name + i_[0])
                self._container.add_initializer(
                    ox_n,
                    i_[1],
                    i_[2].shape,
                    i_[2].flatten()
                )
            elif isinstance(ox_n, str):
                pass
            else:
                raise ValueError('Unknown type for ONNX initializer: {}'.format(type(ox_n)))
            ox_inputs.append(ox_n)

        return ox_inputs

    def _generate_name(self, type_or_func, name):
        base_name = (self.basename if self.basename else '') + '_'
        if name is not None:
            long_name = base_name + name
        else:
            if isinstance(type_or_func, str):
                suffix = type_or_func.lower()
            else:
                suffix = type_or_func.__name__[len('apply_'):]
            long_name = base_name + suffix
        return long_name

    def add_node(self, op_type, inputs, name=None, outputs=None, op_domain='', op_version=None, **attrs):
        if op_version is None:
            op_version = self._container.target_opset
        name = self._generate_name(op_type, name)
        if outputs is None:
            ox_output = 1
        else:
            ox_output = outputs
        if isinstance(ox_output, int):
            ox_output = [self._scope.get_unique_variable_name(name + str(i_)) for i_ in range(ox_output)]
        elif isinstance(ox_output, (list, tuple)):
            pass
        else:
            raise ValueError('Unknown type for outputs: {}'.format(type(ox_output)))
        ox_inputs = self._process_inputs(inputs, name)
        self._container.add_node(op_type, ox_inputs, ox_output, op_domain, op_version,
                                 name=self._scope.get_unique_operator_name(name), **attrs)
        return ox_output[0] if outputs is None else ox_output

    def add_node_with_output(self, op_type, inputs, outputs, name, op_domain='', op_version=None, **attrs):
        if op_version is None:
            op_version = self._container.target_opset
        ox_inputs = self._process_inputs(inputs, name)
        self._container.add_node(op_type, ox_inputs, outputs, op_domain, op_version, name=name, **attrs)
        return outputs

    def apply_op(self, apply_func, inputs, name=None, outputs=None, **attrs):
        name = self._generate_name(apply_func, name)
        if outputs is None:
            ox_output = 1
        else:
            ox_output = outputs
        if isinstance(ox_output, int):
            ox_output = [self._scope.get_unique_variable_name(name + str(i_)) for i_ in range(ox_output)]
        elif isinstance(ox_output, (list, tuple)):
            pass
        else:
            raise ValueError('Unknown type for outputs: {}'.format(type(ox_output)))
        ox_inputs = self._process_inputs(inputs, name)
        apply_func(self._scope, ox_inputs, ox_output, self._container,
                   operator_name=self._scope.get_unique_operator_name(name), **attrs)
        return ox_output[0] if outputs is None else ox_output

    def apply_op_name(self, apply_func_name, inputs, name=None, outputs=None, **attrs):
        apply_operations = onnx_ops.__dict__
        apply_func = apply_operations[apply_func_name]
        assert apply_func is not None, "{} not implemented in onnx_ops.py.".format(apply_func_name)
        return self.apply_op(apply_func, inputs, name, outputs)
