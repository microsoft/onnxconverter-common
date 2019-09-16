# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from collections import Counter


def onnx_statistics(onnx_model, recursive=True):
    """
    Computes statistics on ONNX models.

    :param onnx_model: onnx model
    :param recursive: looks into subgraphs
    :return: dictionary
    """
    atts = ['doc_string', 'ir_version', 'metadata_props', 'domain',
            'model_version', 'producer_name', 'producer_version']

    def update(sts, st):
        for k, v in st.items():
            if k in ['size'] or k in atts:
                continue
            if k in sts:
                sts[k] += v
            else:
                sts[k] = v

    if hasattr(onnx_model, 'graph'):
        content = onnx_model.SerializeToString()
        nnodes = len(onnx_model.graph.node)
        ninits = len(onnx_model.graph.initializer)
        stats = {'size': len(content), 'nnodes': nnodes, 'ninits': ninits}
        for a in atts:
            v = getattr(onnx_model, a)
            if isinstance(v, str):
                li = None
            else:
                try:
                    li = list(v)
                except TypeError:
                    li = None
            if li is not None and len(li) == 0:
                continue
            stats[a] = v

        for opi in onnx_model.opset_import:
            stats[opi.domain] = opi.version

        graph = onnx_model.graph
    elif not hasattr(onnx_model, 'node'):
        # We're in a node.
        stats = {'nnodes': 1}
        if hasattr(onnx_model, 'attribute') and onnx_model.attribute:
            for att in onnx_model.attribute:
                if att.name == 'body':
                    st = onnx_statistics(att.g)
                    update(stats, st)
        return stats
    else:
        graph = onnx_model
        nnodes = len(graph.node)
        stats = {'nnodes': nnodes}

    # Number of identities
    counts = Counter(map(lambda obj: obj.op_type, graph.node))
    for op in ['Cast', 'Identity', 'ZipMap', 'Reshape']:
        if op in counts:
            stats['op_' + op] = counts[op]

    # Recursive
    if recursive:
        for node in graph.node:
            if not hasattr(node, 'attribute'):
                continue
            for att in node.attribute:
                if att.name != 'body':
                    continue
                substats = onnx_statistics(att.g, recursive=True)
                update(stats, {'subgraphs': 1})
                update(stats, substats)

    return stats
