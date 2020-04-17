# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################

import numpy as np
import onnx
from onnx import numpy_helper, mapping, helper


class OnnxGraphContext:
    def __init__(self, graph_proto):
        self.initializers = {ts_.name: ts_ for ts_ in graph_proto.initializer}
        # self.nodes = {nd_.name for nd_ in graph_proto.node}
        self.tensor_to_node = {}
        for nd_ in graph_proto.node:
            self.tensor_to_node.update({ts_: nd_ for ts_ in nd_.output})
        self.variables = {}

    def add_value_of_node(self, name, value):
        if name in self.variables:
            assert False, "The tensor({}) already was assigned!".format(name)
        else:
            self.variables[name] = value

    @staticmethod
    def get_attribute(node, attr_name, default_value=None):
        found = [attr for attr in node.attribute if attr.name == attr_name]
        if found:
            return helper.get_attribute_value(found[0])
        return default_value

    def calculate(self, node):
        func_name = '_On' + node.op_type
        func = type(self).__dict__.get(func_name, None)
        if func is None:
            return False

        inputs = []
        for ts_ in node.input:
            if ts_ in self.initializers:
                inputs.append(numpy_helper.to_array(self.initializers[ts_]))
            elif ts_ in self.variables:
                inputs.append(self.variables[ts_])
            else:
                return None

        output_values = func(self, node, inputs)
        for idx_, ots_ in enumerate(node.output):
            self.add_value_of_node(ots_, output_values[idx_])

        return output_values

    def _OnIdentity(self, node, inputs):
        return inputs

    def _OnConst(self, node, inputs):
        return [OnnxGraphContext.get_attribute(node, 'value')]

    def _OnAdd(self, node, inputs):
        return [np.add(inputs[0], inputs[1])]

    def _OnCast(self, node, inputs):
        np_dtype = mapping.TENSOR_TYPE_TO_NP_TYPE[OnnxGraphContext.get_attribute(node, 'to')]
        casted = inputs[0].astype(np_dtype)
        return [casted]

    def _OnGather(self, node, inputs):
        data_val = inputs[0]
        idx_val = inputs[1]
        axis = OnnxGraphContext.get_attribute(node, 'axis')
        retval = np.take(data_val, idx_val, axis).astype(inputs[0].dtype)
        return [retval]

    def _OnGreater(self, node, inputs):
        return [np.greater(inputs[0], inputs[1])]

    def _OnLess(self, node, inputs):
        return [np.less(inputs[0], inputs[1])]

    def _OnMul(self, node, inputs):
        return [np.multiply(inputs[0], inputs[1])]

    def _OnNot(self, node, inputs):
        return [np.logical_not(inputs[0])]

    def _OnTranspose(self, node, inputs):
        perm_attr = OnnxGraphContext.get_attribute(node, 'perm')
        retval = inputs[0].transpose(perm_attr)
        return [retval]

    def _OnRange(self, node, inputs):
        retval = np.asarray(list(range(inputs[0].astype(np.int32).item(0),
                                       inputs[1].astype(np.int32).item(0),
                                       inputs[2].astype(np.int32).item(0))))
        return [retval]

    def _OnSlice(self, node, inputs):
        data_val = inputs[0]
        rank = len(inputs[0].shape)
        starts = inputs[1].tolist()
        ends = inputs[2].tolist()
        axes = inputs[3].tolist() if len(node.input) > 3 else list(range(0, rank))
        steps = inputs[4].tolist() if len(node.input) > 4 else [1] * rank
        exp_all = list()
        for axis_ in range(0, rank):
            if axis_ in axes:
                idx = axes.index(axis_)
                cur_starts = starts[idx]
                cur_ends = ends[idx]
                cur_step = steps[idx]
            else:
                cur_starts = 0
                cur_ends = np.iinfo(np.int64).max
                cur_step = 1

            cur_starts = max(cur_starts, -inputs[0].shape[axis_])
            cur_ends = min(cur_ends, inputs[0].shape[axis_])
            exp = slice(cur_starts, cur_ends, cur_step)
            exp_all.append(exp)

        retval = data_val[tuple(exp_all)]
        return [retval]

    def _OnSqrt(self, node, inputs):
        return [np.sqrt(inputs[0].tolist(), dtype=inputs[0].dtype)]

    def _OnSub(self, node, inputs):
        return [np.subtract(inputs[0], inputs[1])]

    def _OnUnsqueeze(self, node, inputs):
        axes = OnnxGraphContext.get_attribute(node, 'axes')
        shape_in = inputs[0].shape
        dims_out = len(shape_in) + len(axes)
        shape_in = iter(shape_in)
        shape_out = [None] * dims_out
        for idx_ in axes:
            shape_out[idx_] = 1
        for ind, val in enumerate(shape_out):
            if val is None:
                shape_out[ind] = next(shape_in)

        retval = inputs[0].reshape(shape_out)
        return [retval]

    def _OnReshape(self, node, inputs):
        retval = inputs[0].reshape(inputs[1])
        return [retval]


def _fix_unamed_node(graph):
    # type: (onnx.GraphProto)->onnx.GraphProto
    node_id = [1]

    def _ensure_node_named(node, incre_id):
        if node.name:
            return node
        name = node.op_type.lower() + "_{}".format(incre_id[0])
        incre_id[0] += 1
        node.name = name
        return node

    named_nodes = [_ensure_node_named(nd_, node_id) for nd_ in graph.node]

    if node_id[0] == 1:
        return graph

    del graph.node[:]
    graph.node.extend(named_nodes)
    return graph


def reserve_node_for_embedded_graph(graph):
    # type: (onnx.GraphProto)->frozenset
    fixed_graph = _fix_unamed_node(graph)
    ginputs = []
    for nd_ in fixed_graph.node:
        if nd_.op_type in ['Loop', 'Scan']:
            inner_graph = OnnxGraphContext.get_attribute(nd_, 'body')
            inner_inputs = frozenset([i_.name for i_ in inner_graph.input])
            for sub_nd_ in inner_graph.node:
                ginputs.extend([i_ for i_ in sub_nd_.input if i_ not in inner_inputs])
    return fixed_graph, frozenset(ginputs)


def _dfs_calc(graph, node, reserved_names, node_status):
    # type: (OnnxGraphContext, onnx.NodeProto, frozenset, dict) -> int
    if node.name in node_status:
        return node_status[node.name]

    if len(node.input) == 0:
        assert node.op_type == 'Constant', "Assume only a constant node hasn't any inputs"
        if all(o_ not in reserved_names for o_ in node.output):
            graph.calculate(node)
        return 0
    else:
        calc_status = [0] * len(node.input)
        for idx_, ts_ in enumerate(node.input):
            if ts_ in graph.initializers:
                calc_status[idx_] = -1 if ts_ in reserved_names else 0
            elif ts_ not in graph.tensor_to_node:  # input of graph
                calc_status[idx_] = -1
            else:
                calc_status[idx_] = _dfs_calc(graph, graph.tensor_to_node[ts_], reserved_names, node_status)

        status_up = max(calc_status)
        status_low = min(calc_status)
        if status_low < 0:
            status = - max(-status_low, abs(status_up)) - 1
        else:
            status = status_up + 1

        node_status[node.name] = status
        if status > 0:
            if any(o_ in reserved_names for o_ in node.output):
                status = -status
            else:
                graph.calculate(node)
        return status


def _remove_unused_initializers(nodes, initializers, reversed_names):
    nodes_input_set = set()
    for nd_ in nodes:
        nodes_input_set.update(n_ for n_ in nd_.input)

    return [intlz_ for intlz_ in initializers if intlz_.name in nodes_input_set or intlz_.name in reversed_names]


def const_folding_optimizer(graph):
    # type: (onnx.GraphProto)->onnx.GraphProto
    fixed_graph, reserved_names = reserve_node_for_embedded_graph(graph)
    opt_graph = OnnxGraphContext(fixed_graph)
    node_status = {}
    for ts_ in graph.output:
        _dfs_calc(opt_graph, opt_graph.tensor_to_node[ts_.name], reserved_names, node_status)

    graph.initializer.extend([numpy_helper.from_array(ts_, nm_) for nm_, ts_ in opt_graph.variables.items()])
    new_nodes = [nd_ for nd_ in graph.node if nd_.name in node_status]
    new_nodes = [nd_ for nd_ in new_nodes if nd_.output[0] not in opt_graph.variables]

    def node_key(nd_):
        return abs(node_status[nd_.name])

    new_nodes.sort(key=node_key)
    pruned_initilizers = _remove_unused_initializers(new_nodes, graph.initializer, reserved_names)
    del graph.node[:]
    graph.node.extend(new_nodes)
    del graph.initializer[:]
    graph.initializer.extend(pruned_initilizers)
    return graph
