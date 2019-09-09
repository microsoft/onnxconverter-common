# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ._onnx_optimisation_common import _apply_optimisation_on_graph
from .onnx_optimisation_identity import onnx_remove_node_identity
from .onnx_optimisation_redundant import onnx_remove_node_redundant


def onnx_remove_node(onnx_model, recursive=True, debug_info=None):
    """
    Removes as many nodes as possible without changing
    the outcome. It applies *onnx_remove_node_identity*,
    then *onnx_remove_node_redundant*.

    :param onnx_model: onnx model
    :param recursive: looks into subgraphs
    :param debug_info: debug information (private)
    :return: new onnx _model
    """
    if debug_info is None:
        debug_info = [str(type(onnx_model)).split('.')[-1].strip("'>")]
    else:
        debug_info = debug_info + \
            [str(type(onnx_model)).split('.')[-1].strip("'>")]

    if hasattr(onnx_model, 'graph'):
        return _apply_optimisation_on_graph(
            onnx_remove_node, onnx_model,
            recursive=recursive, debug_info=debug_info)

    graph = onnx_model
    graph = onnx_remove_node_identity(
        graph, recursive=recursive, debug_info=debug_info)
    graph = onnx_remove_node_redundant(
        graph, recursive=recursive, debug_info=debug_info)
    return graph
