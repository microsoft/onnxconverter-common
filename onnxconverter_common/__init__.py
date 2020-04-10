# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################

"""
The entry point to onnxconverter-common.
This framework performs optimization for ONNX models and
includes common utilities for ONNX converters.
"""
__version__ = "1.7.91"
__author__ = "Microsoft"
__producer__ = "OnnxMLTools"
__producer_version__ = __version__
__domain__ = "onnxconverter-common"
__model_version__ = 0

from .data_types import *
from .onnx_ops import *
from .container import *
from .registration import *
from .topology import *
from .interface import *
from .shape_calculator import *
from .tree_ensemble import *
from .utils import *
from .case_insensitive_dict import *
from .metadata_props import add_metadata_props, set_denotation
from .float16 import convert_tensor_float_to_float16
from .optimizer import optimize_onnx, optimize_onnx_graph, optimize_onnx_model
