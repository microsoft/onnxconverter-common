# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################

import numbers

import onnx
from onnx import onnx_pb as onnx_proto


class DataType:
    def __init__(self, shape=None, doc_string=""):
        self.shape = shape
        self.doc_string = doc_string

    def to_onnx_type(self):
        raise NotImplementedError()

    def __repr__(self):
        return f"{self.__class__.__name__}(shape={self.shape})"


class Int64Type(DataType):
    def __init__(self, doc_string=""):
        super().__init__([1, 1], doc_string)

    def to_onnx_type(self):
        onnx_type = onnx_proto.TypeProto()
        onnx_type.tensor_type.elem_type = onnx_proto.TensorProto.INT64
        s = onnx_type.tensor_type.shape.dim.add()
        s.dim_value = 1
        return onnx_type

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class FloatType(DataType):
    def __init__(self, doc_string=""):
        super().__init__([1, 1], doc_string)

    def to_onnx_type(self):
        onnx_type = onnx_proto.TypeProto()
        onnx_type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT
        s = onnx_type.tensor_type.shape.dim.add()
        s.dim_value = 1
        return onnx_type

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class StringType(DataType):
    def __init__(self, doc_string=""):
        super().__init__([1, 1], doc_string)

    def to_onnx_type(self):
        onnx_type = onnx_proto.TypeProto()
        onnx_type.tensor_type.elem_type = onnx_proto.TensorProto.STRING
        s = onnx_type.tensor_type.shape.dim.add()
        s.dim_value = 1
        return onnx_type

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class TensorType(DataType):
    def __init__(self, shape=None, doc_string="", denotation=None, channel_denotations=None):
        super().__init__([] if not shape else shape, doc_string)
        self.denotation = denotation
        self.channel_denotations = channel_denotations

    def _get_element_onnx_type(self):
        raise NotImplementedError()

    def to_onnx_type(self):
        onnx_type = onnx_proto.TypeProto()
        onnx_type.tensor_type.elem_type = self._get_element_onnx_type()
        for d in self.shape:
            s = onnx_type.tensor_type.shape.dim.add()
            if d is None:
                pass
            elif isinstance(d, numbers.Integral):
                s.dim_value = d
            elif isinstance(d, str):
                s.dim_param = d
            else:
                raise ValueError(
                    f"Unsupported dimension type: {type(d)}, see "
                    "https://github.com/onnx/onnx/blob/master/docs/IR.md#"
                    "input--output-data-types"
                )
        if getattr(onnx_type, "denotation", None) is not None:
            if self.denotation:
                onnx_type.denotation = self.denotation
            if self.channel_denotations:
                for d, denotation in zip(onnx_type.tensor_type.shape.dim, self.channel_denotations):
                    if denotation:
                        d.denotation = denotation
        return onnx_type


class Int32TensorType(TensorType):
    def __init__(self, shape=None, doc_string=""):
        super().__init__(shape, doc_string)

    def _get_element_onnx_type(self):
        return onnx_proto.TensorProto.INT32


class Int8TensorType(TensorType):
    def __init__(self, shape=None, doc_string=""):
        super().__init__(shape, doc_string)

    def _get_element_onnx_type(self):
        return onnx_proto.TensorProto.INT8


class UInt8TensorType(TensorType):
    def __init__(self, shape=None, doc_string=""):
        super().__init__(shape, doc_string)

    def _get_element_onnx_type(self):
        return onnx_proto.TensorProto.UINT8


class Int64TensorType(TensorType):
    def __init__(self, shape=None, doc_string=""):
        super().__init__(shape, doc_string)

    def _get_element_onnx_type(self):
        return onnx_proto.TensorProto.INT64


class BooleanTensorType(TensorType):
    def __init__(self, shape=None, doc_string=""):
        super().__init__(shape, doc_string)

    def _get_element_onnx_type(self):
        return onnx_proto.TensorProto.BOOL


class FloatTensorType(TensorType):
    def __init__(
        self,
        shape=None,
        color_space=None,
        doc_string="",
        denotation=None,
        channel_denotations=None,
    ):
        super().__init__(shape, doc_string, denotation, channel_denotations)
        self.color_space = color_space

    def _get_element_onnx_type(self):
        return onnx_proto.TensorProto.FLOAT


class DoubleTensorType(TensorType):
    def __init__(self, shape=None, color_space=None, doc_string=""):
        super().__init__(shape, doc_string)
        self.color_space = color_space

    def _get_element_onnx_type(self):
        return onnx_proto.TensorProto.DOUBLE


class Complex64TensorType(TensorType):
    def __init__(self, shape=None, color_space=None, doc_string=""):
        super().__init__(shape, doc_string)
        self.color_space = color_space

    def _get_element_onnx_type(self):
        return onnx_proto.TensorProto.COMPLEX64


class Complex128TensorType(TensorType):
    def __init__(self, shape=None, color_space=None, doc_string=""):
        super().__init__(shape, doc_string)
        self.color_space = color_space

    def _get_element_onnx_type(self):
        return onnx_proto.TensorProto.COMPLEX128


class StringTensorType(TensorType):
    def __init__(self, shape=None, doc_string=""):
        super().__init__(shape, doc_string)

    def _get_element_onnx_type(self):
        return onnx_proto.TensorProto.STRING


class DictionaryType(DataType):
    def __init__(self, key_type, value_type, shape=None, doc_string=""):
        super().__init__(shape, doc_string)
        self.key_type = key_type
        self.value_type = value_type

    def to_onnx_type(self):
        onnx_type = onnx_proto.TypeProto()
        try:
            if type(self.key_type) in [Int64Type, Int64TensorType]:
                onnx_type.map_type.key_type = onnx_proto.TensorProto.INT64
            elif type(self.key_type) in [StringType, StringTensorType]:
                onnx_type.map_type.key_type = onnx_proto.TensorProto.STRING
            onnx_type.map_type.value_type.CopyFrom(self.value_type.to_onnx_type())
        except AttributeError:
            msg = "ONNX was not compiled with flag ONNX-ML.\n{0}\n{1}"
            msg = msg.format(str(self), str(self.value_type.to_onnx_type()))
            info = [onnx.__version__, str(onnx_type)]
            msg += "\n".join(info)
            raise RuntimeError(msg)
        return onnx_type

    def __repr__(self):
        return f"DictionaryType(key_type={self.key_type}, value_type={self.value_type})"


class SequenceType(DataType):
    def __init__(self, element_type, shape=None, doc_string=""):
        super().__init__(shape, doc_string)
        self.element_type = element_type
        self.doc_string = doc_string

    def to_onnx_type(self):
        onnx_type = onnx_proto.TypeProto()
        try:
            onnx_type.sequence_type.elem_type.CopyFrom(self.element_type.to_onnx_type())
        except AttributeError:
            msg = "ONNX was not compiled with flag ONNX-ML.\n{0}\n{1}"
            msg = msg.format(str(self), str(self.element_type.to_onnx_type()))
            info = [onnx.__version__, str(onnx_type)]
            msg += "\n".join(info)
            raise RuntimeError(msg)
        except TypeError:
            raise RuntimeError(f"Unable to create SequenceType with element_type={self.element_type!r}")
        return onnx_type

    def __repr__(self):
        return f"SequenceType(element_type={self.element_type})"


def find_type_conversion(source_type, target_type):
    """
    Find the operator name for converting source_type into target_type
    """
    if isinstance(source_type, target_type):
        return "identity"
    if isinstance(target_type, FloatTensorType):
        return "imageToFloatTensor"
    raise ValueError(f"Unsupported type conversion from {source_type} to {target_type}")


def onnx_built_with_ml():
    """
    Tells if ONNX was built with flag ``ONNX-ML``.
    """
    seq = SequenceType(FloatTensorType())
    try:
        seq.to_onnx_type()
        return True
    except RuntimeError:
        return False
