# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import copy
import hashlib
from onnx.helper import make_graph
from ._onnx_optimisation_common import (  # pylint: disable=E0611
    _rename_node_input,
    _rename_node_output,
    _apply_optimisation_on_graph,
    _apply_remove_node_fct_node
)


def _hash_obj_content(obj, max_size=1000):
    """
    Hash the content of an object.
    """
    m = hashlib.sha256()
    if hasattr(obj, 'op_type'):
        # An operator.
        m.update(obj.op_type.encode('ascii'))
        m.update(len(obj.output).to_bytes(8, byteorder='big'))
        for i in obj.input:
            m.update(i.encode('ascii'))
        if hasattr(obj, 'attribute'):
            for att in obj.attribute:
                m.update(att.name.encode('ascii'))
                m.update(_hash_obj_content(att))
    else:
        # An initializer.
        obj = copy.deepcopy(obj)
        obj.name = ""
        obj.doc_string = ""
        m.update(obj.SerializeToString())

    content = m.digest()
    if len(content) > max_size:
        content = content[:max_size]
    return content


def onnx_remove_node_redundant(onnx_model, recursive=True, debug_info=None,
                               max_hash_size=1000):
    """
    Removes redundant part of the graph. A redundant part is
    a set of nodes which takes the same inputs and produces
    the same outputs. It first starts by looking into duplicated
    initializers, then looks into nodes taking the same inputs
    and sharing the same type and parameters.

    :param onnx_model: onnx model
    :param recursive: looks into subgraphs
    :param debug_info: debug information (private)
    :param max_hash_size: limit the size of a hash used to detect
        identical subgraphs
    :return: new onnx _model
    """
    if debug_info is None:
        debug_info = [str(type(onnx_model)).split('.')[-1].strip("'>")]
    else:
        debug_info = debug_info + \
            [str(type(onnx_model)).split('.')[-1].strip("'>")]

    if hasattr(onnx_model, 'graph'):
        return _apply_optimisation_on_graph(
            onnx_remove_node_redundant, onnx_model,
            recursive=recursive, debug_info=debug_info,
            max_hash_size=max_hash_size)

    def _enumerate_rename_list_nodes_inputs(nodes, rename):
        for i, node in enumerate(nodes):
            if node is None:
                yield False, i, None
                continue
            if any(set(node.input) & set(rename)):
                yield True, i, _rename_node_input(node, rename)
                continue
            yield False, i, node

    graph = onnx_model

    # Detects duplicated initializers.
    hashes = {}
    names = []
    rename = {}
    for init in graph.initializer:
        hs = _hash_obj_content(init, max_size=max_hash_size)
        if hs in hashes:
            # Already seen.
            rename[init.name] = hashes[hs]
        else:
            # New.
            hashes[hs] = init.name
            names.append(init.name)

    new_inits = [init for init in graph.initializer if init.name in set(names)]

    # Renames node inputs.
    new_nodes = []
    new_nodes = list(graph.node)
    new_nodes = list(
        _[2] for _ in _enumerate_rename_list_nodes_inputs(new_nodes, rename))

    # Detects duplicated operators.
    graph_outputs = set(o.name for o in graph.output)
    node_hashes = {}
    changed = 1
    replace = {}
    while changed > 0:
        changed = 0
        nnodes = len(new_nodes)
        for i in range(nnodes):
            if i in replace:
                # Already removed.
                continue
            node = new_nodes[i]
            hash = _hash_obj_content(node, max_size=max_hash_size)
            if hash in node_hashes:
                ni = node_hashes[hash]
                if ni == i:
                    continue
                replace[i] = ni
                changed += 1

                # Specifies what to rename.
                # One exception: the output is one of the graph output.
                rep = new_nodes[ni]
                for old, nn in zip(node.output, rep.output):
                    if old in graph_outputs:
                        rename[nn] = old
                        new_nodes[ni] = _rename_node_output(
                            new_nodes[ni], nn, old)
                    else:
                        rename[old] = nn

                # Renames inputs.
                new_new_nodes = []
                renew_index = set()
                for changed, ci, node in _enumerate_rename_list_nodes_inputs(new_nodes, rename):
                    if changed:
                        renew_index.add(ci)
                    new_new_nodes.append(node)
                new_nodes = new_new_nodes

                # Renews hashes.
                renew_hash = set(
                    k for k, v in node_hashes.items() if v in renew_index)
                for hs in renew_hash:
                    del node_hashes[hs]
                new_nodes[i] = None
            else:
                node_hashes[hash] = i

    if recursive:
        # Handles subgraphs.
        for i in range(len(new_nodes)):  # pylint: disable=C0200
            node = new_nodes[i]
            if node is None or not (node.attribute):  # pylint: disable=C0325
                continue
            new_nodes[i] = _apply_remove_node_fct_node(
                onnx_remove_node_redundant,
                node, recursive=True, debug_info=debug_info + [node.name])

    # Finally create the new graph.
    nodes = list(filter(lambda n: n is not None, new_nodes))
    graph = make_graph(nodes, onnx_model.name,
                       onnx_model.input, onnx_model.output,
                       new_inits)

    graph.value_info.extend(onnx_model.value_info)  # pylint: disable=E1101
    return graph
