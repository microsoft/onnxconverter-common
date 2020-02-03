import onnx
from onnx import numpy_helper, mapping, helper


class OnnxGraph:
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
        return [OnnxGraph.get_attribute(node, 'value')]

    def _OnCast(self, node, inputs):
        np_dtype = mapping.TENSOR_TYPE_TO_NP_TYPE[OnnxGraph.get_attribute(node, 'to')]
        casted = inputs[0].astype(np_dtype)
        return [casted]

    def _OnTranspose(self, node, inputs):
        perm_attr = OnnxGraph.get_attribute(node, 'perm')
        retval = inputs[0].transpose(perm_attr)
        return [retval]

    def _OnUnsqueeze(self, node, inputs):
        axes = OnnxGraph.get_attribute(node, 'axes')
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


def _dfs_calc(graph, node, node_status):
    # type: (OnnxGraph, onnx.NodeProto, dict) -> int
    if node.name in node_status:
        return node_status[node.name]

    if len(node.input) == 0:
        assert node.op_type == 'Constant', "Assume only a constant node hasn't any inputs"
        graph.calculate(node)
        return 0
    else:
        calc_status = [0] * len(node.input)
        for idx_, ts_ in enumerate(node.input):
            if ts_ in graph.initializers:
                calc_status[idx_] = 0
            elif ts_ not in graph.tensor_to_node:  # input of graph
                calc_status[idx_] = -1
            else:
                calc_status[idx_] = _dfs_calc(graph, graph.tensor_to_node[ts_], node_status)

        status_up = max(calc_status)
        status_low = min(calc_status)
        if status_low < 0:
            status = - max(-status_low, abs(status_up)) - 1
        else:
            status = status_up + 1

        node_status[node.name] = status
        if status > 0:
            graph.calculate(node)
        return status


def const_folding_optimizer(graph):
    # type: (onnx.GraphProto)->onnx.GraphProto
    opt_graph = OnnxGraph(graph)
    node_status = {}
    for ts_ in graph.output:
        _dfs_calc(opt_graph, opt_graph.tensor_to_node[ts_.name], node_status)

    graph.initializer.extend([numpy_helper.from_array(ts_, nm_) for nm_, ts_ in opt_graph.variables.items()])
    new_nodes = [nd_ for nd_ in graph.node if nd_.name in node_status]
    new_nodes = [nd_ for nd_ in new_nodes if nd_.output[0] not in opt_graph.variables]

    def node_key(nd_):
        return abs(node_status[nd_.name])

    new_nodes.sort(key=node_key)

    del graph.node[:]
    graph.node.extend(new_nodes)
    return graph
