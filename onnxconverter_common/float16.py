# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###########################################################################

import itertools
import numpy as np
import onnx
import packaging.version as pv
import warnings
from onnx import helper, numpy_helper
from onnx import onnx_pb as onnx_proto


FLOAT32 = 1
FLOAT16 = 10


def _npfloat16_to_int(np_list):
    '''
    Convert numpy float16 to python int.

    :param np_list: numpy float16 list
    :return int_list: python int list
    '''
    return [int(bin(_.view('H'))[2:].zfill(16), 2) for _ in np_list]


def convert_np_to_float16(np_array, min_positive_val=1e-7, max_finite_val=1e4):
    '''
    Convert float32 numpy array to float16 without changing sign or finiteness.
    Positive values less than min_positive_val are mapped to min_positive_val.
    Positive finite values greater than max_finite_val are mapped to max_finite_val.
    Similar for negative values. NaN, 0, inf, and -inf are unchanged.
    '''
    def between(a, b, c):
        return np.logical_and(a < b, b < c)

    if (np_array[np.where(np_array > 0)].shape[0] > 0):
        pos_max = np_array[np.where(np_array > 0)].max()
        pos_min = np_array[np.where(np_array > 0)].min()

        if (pos_max >= max_finite_val):
            warnings.warn("the float32 number {} will be truncated to {}".format(pos_max, max_finite_val))

        if (pos_min <= min_positive_val):
            warnings.warn("the float32 number {} will be truncated to {}".format(pos_min, min_positive_val))

    if (np_array[np.where(np_array < 0)].shape[0] > 0):
        neg_max = np_array[np.where(np_array < 0)].max()
        neg_min = np_array[np.where(np_array < 0)].min()

        if (neg_min <= -max_finite_val):
            warnings.warn("the float32 number {} will be truncated to {}".format(neg_min, -max_finite_val))

        if (neg_max >= -min_positive_val):
            warnings.warn("the float32 number {} will be truncated to {}".format(neg_max, -min_positive_val))

    np_array = np.where(between(0, np_array, min_positive_val), min_positive_val, np_array)
    np_array = np.where(between(-min_positive_val, np_array, 0), -min_positive_val, np_array)
    np_array = np.where(between(max_finite_val, np_array, float('inf')), max_finite_val, np_array)
    np_array = np.where(between(float('-inf'), np_array, -max_finite_val), -max_finite_val, np_array)
    return np.float16(np_array)


def convert_tensor_float_to_float16(tensor, min_positive_val=1e-7, max_finite_val=1e4):
    '''
    Convert tensor float to float16.

    :param tensor: TensorProto object
    :return tensor_float16: converted TensorProto object

    Example:

    ::

        from onnxmltools.utils.float16_converter import convert_tensor_float_to_float16
        new_tensor = convert_tensor_float_to_float16(tensor)

    '''
    if not isinstance(tensor, onnx_proto.TensorProto):
        raise ValueError('Expected input type is an ONNX TensorProto but got %s' % type(tensor))

    if tensor.data_type == onnx_proto.TensorProto.FLOAT:
        tensor.data_type = onnx_proto.TensorProto.FLOAT16
        # convert float_data (float type) to float16 and write to int32_data
        if tensor.float_data:
            float16_data = convert_np_to_float16(np.array(tensor.float_data), min_positive_val, max_finite_val)
            int_list = _npfloat16_to_int(float16_data)
            tensor.int32_data[:] = int_list
            tensor.float_data[:] = []
        # convert raw_data (bytes type)
        if tensor.raw_data:
            # convert n.raw_data to float
            float32_list = np.fromstring(tensor.raw_data, dtype='float32')
            # convert float to float16
            float16_list = convert_np_to_float16(float32_list, min_positive_val, max_finite_val)
            # convert float16 to bytes and write back to raw_data
            tensor.raw_data = float16_list.tostring()
    return tensor


def make_value_info_from_tensor(tensor):
    shape = numpy_helper.to_array(tensor).shape
    return helper.make_tensor_value_info(tensor.name, tensor.data_type, shape)


DEFAULT_OP_BLOCK_LIST = ['ArrayFeatureExtractor', 'Binarizer', 'CastMap', 'CategoryMapper', 'DictVectorizer',
                         'FeatureVectorizer', 'Imputer', 'LabelEncoder', 'LinearClassifier', 'LinearRegressor',
                         'Normalizer', 'OneHotEncoder', 'RandomUniformLike', 'SVMClassifier', 'SVMRegressor', 'Scaler',
                         'TreeEnsembleClassifier', 'TreeEnsembleRegressor', 'ZipMap', 'NonMaxSuppression', 'TopK',
                         #'RoiAlign', 'Resize', 'Range', 'CumSum', 'Min', 'Max', 'Upsample']
                         'RoiAlign', 'Resize', 'Range', 'CumSum', 'Min', 'Upsample']


def initial_checking(model, disable_shape_infer):
    func_infer_shape = None
    if not disable_shape_infer and pv.Version(onnx.__version__) >= pv.Version('1.2'):
        try:
            from onnx.shape_inference import infer_shapes
            func_infer_shape = infer_shapes
        finally:
            pass

    if not isinstance(model, onnx_proto.ModelProto):
        raise ValueError('Expected model type is an ONNX ModelProto but got %s' % type(model))

    if func_infer_shape is not None:
        model = func_infer_shape(model)

    return model, func_infer_shape

# new implementation by Xiaowu to fix a lot of bug due to ort changed
def convert_float_to_float16(model, min_positive_val=1e-7, max_finite_val=1e4,
                             is_io_fp32=False, disable_shape_infer=False,
                             op_block_list=None, node_block_list=None):

    # create blocklists
    if op_block_list is None:
        op_block_list = DEFAULT_OP_BLOCK_LIST
    if node_block_list is None:
        node_block_list = []
    op_block_list = set(op_block_list)
    node_block_list = set(node_block_list)

    global_input_name_dict = {}  # key: input name, value: new output name after Cast node
    # basic checking, including shape inference
    model, func_infer_shape = initial_checking(model, disable_shape_infer)
    graph_stack = [model.graph]
    
    is_top_level = True
    while graph_stack:
        next_level = []
        for curr_graph in graph_stack:

            process_graph_input(curr_graph, is_top_level, is_io_fp32, global_input_name_dict, op_block_list, node_block_list)    
            process_initializers(curr_graph, min_positive_val, max_finite_val)
            process_tensor_in_node(curr_graph, op_block_list, node_block_list, min_positive_val, max_finite_val)
            
            value_info_block_list = process_node_in_block_list(curr_graph, is_io_fp32, global_input_name_dict, op_block_list, node_block_list)
            process_value_info(curr_graph, value_info_block_list)
            process_tensor_in_node(curr_graph, op_block_list, node_block_list, min_positive_val, max_finite_val)

            # This is for the model can shwo all value_info (shape and size) in Netron
            sub_graph_list = get_next_level_graph(curr_graph, op_block_list, node_block_list)
            next_level.extend(sub_graph_list)

            process_graph_output(curr_graph, is_top_level, is_io_fp32)

            if not is_top_level:
                process_node_input_output(curr_graph, global_input_name_dict)

            is_top_level = False  # Going to process sub-graph
        
        graph_stack = next_level

    if func_infer_shape is not None:
        model = func_infer_shape(model)

    return model

# Change the input/output of the node to the new output name after Cast node
def process_node_input_output(graph: onnx_proto.GraphProto, global_input_name_dict: dict):
    for node in graph.node:
        for i, input_name in enumerate(node.input):
            if input_name in global_input_name_dict:
                node.input[i] = global_input_name_dict[input_name]
        for i, output_name in enumerate(node.output):
            if output_name in global_input_name_dict:
                node.output[i] = global_input_name_dict[output_name]


# 处理输入
def process_graph_input(graph: onnx_proto.GraphProto, is_top_level: bool, is_io_fp32: bool, global_input_name_dict: dict, op_block_list: list, node_block_list: list):
    # The input dtype is float32, need to cast to fp16
    if is_top_level and is_io_fp32:
        for n_input in graph.input:  # n_input is ValueInfoProto
            if n_input.type.tensor_type.elem_type == onnx_proto.TensorProto.FLOAT:
                downstream_nodes = find_donwstream_node_by_input(graph, n_input.name)
                for i, d_n in enumerate(downstream_nodes):
                    # Add cast to 16 for very next node
                    # but if the very next node needs fp32, don't do that
                    if not(d_n.op_type in op_block_list or d_n.name in node_block_list):
                        cast_node_output_name = insert_cast_node_between(graph, n_input, d_n, FLOAT16, id=i)              
                        # Sometimes sub-graph will use the global input directly
                        # But sub-graph not in the downstream_nodes
                        # So we need to remember the new output name of the input
                        if cast_node_output_name is not None:
                            global_input_name_dict[n_input.name] = cast_node_output_name
    else:  # Change the input dtype to fp16 without any cast
        for graph_input in graph.input:
            if graph_input.type.tensor_type.elem_type == onnx_proto.TensorProto.FLOAT:
                graph_input.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT16


# 在黑名单中的 op 前面要加 cast32, 后面加 cast16
def process_node_in_block_list(graph: onnx_proto.GraphProto, is_io_fp32, global_input_name_dict: dict, op_block_list, node_block_list):
    value_info_block_list = set()
    for node in graph.node:
        if node.op_type in op_block_list or node.name in node_block_list:
            for i, input_name in enumerate(node.input):                
                upstream_node = find_upstream_node_by_output(graph, input_name)
                if upstream_node is None:  # Should be the graph input
                    if not is_io_fp32:
                        value_info = find_value_info_by_name(graph, input_name)
                        cast_node_output_name = insert_cast_node_between(graph, value_info, node, FLOAT32)
                        value_info_block_list.add(cast_node_output_name)
                        if cast_node_output_name is not None:
                            global_input_name_dict[input_name] = cast_node_output_name
                    continue
                # 检查上游有没有 cast32
                if upstream_node.op_type == 'Cast':
                    if upstream_node.attribute[0].i != FLOAT32:
                        upstream_node.attribute[0].i = FLOAT32
                else:  # 加 cast 32
                    # 如果 value_info type 是 float16，需要加 cast 32
                    cast_node_output_name = insert_cast_node_between(graph, upstream_node, node, FLOAT32)
                    if cast_node_output_name is not None:
                        global_input_name_dict[input_name] = cast_node_output_name
                        value_info_block_list.add(cast_node_output_name)
            # 检查下游有没有 cast16
            for i, output_name in enumerate(node.output):
                value_info_block_list.add(output_name)
                downstream_nodes = find_donwstream_node_by_input(graph, output_name)
                for d_n in downstream_nodes:
                    if d_n.op_type == 'Cast':
                        if d_n.attribute[0].i != FLOAT16:
                            d_n.attribute[0].i = FLOAT16
                    else:  # 加 cast 16
                        cast_node_output_name = insert_cast_node_between(graph, node, d_n, FLOAT16)
                        if cast_node_output_name is not None:
                            global_input_name_dict[output_name] = cast_node_output_name
    return value_info_block_list


def process_tensor_in_node(graph: onnx_proto.GraphProto, op_block_list, node_block_list, min_positive_val, max_finite_val):
    for node in graph.node:
        if node.op_type in op_block_list or node.name in node_block_list:
            continue
        for attr in node.attribute:
            # 一个 tensor
            # attr.t = convert_tensor_float_to_float16(attr.t, min_positive_val, max_finite_val)
            attr.t.CopyFrom(convert_tensor_float_to_float16(attr.t, min_positive_val, max_finite_val))
            # 多个 tensor
            for t in attr.tensors:
                t.CopyFrom(convert_tensor_float_to_float16(t, min_positive_val, max_finite_val))
                

def process_value_info(graph: onnx_proto.GraphProto, value_info_block_list):
    for value_info in graph.value_info:
        if value_info.name in value_info_block_list:
            continue
        else:
            if value_info.type.tensor_type.elem_type == onnx_proto.TensorProto.FLOAT:
                value_info.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT16


def process_initializers(graph: onnx_proto.GraphProto, min_positive_val, max_finite_val):
    for initializer in graph.initializer:
        if initializer.data_type == onnx_proto.TensorProto.FLOAT:
            convert_tensor_float_to_float16(initializer, min_positive_val, max_finite_val)


def process_graph_output(graph: onnx_proto.GraphProto, is_top_level: bool, is_io_fp32: bool):
    if is_top_level and is_io_fp32:  # the output dtype is float32, need to cast to fp16
        for i, n_output in enumerate(graph.output):
            if n_output.type.tensor_type.elem_type == onnx_proto.TensorProto.FLOAT:
                upstream_node = find_upstream_node_by_output(graph, n_output.name)
                if upstream_node is not None:
                    insert_cast_node_between(graph, upstream_node, n_output, FLOAT32, id=i)
    else:  # change the output dtype to fp16 in tensor
        for graph_output in graph.output:
            if graph_output.type.tensor_type.elem_type == onnx_proto.TensorProto.FLOAT:
                graph_output.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT16


def get_next_level_graph(graph: onnx_proto.GraphProto, op_block_list, node_block_list):
    sub_graph_list = []
    for node in graph.node:
        if node.op_type in op_block_list or node.name in node_block_list:
            continue
        for attr in node.attribute:
            # 处理 sub-graph
            if len(attr.g.node) > 0:
                sub_graph_list.append(attr.g) 
                print("ssssss")
            for g in attr.graphs:
                if len(g.node) > 0:
                    sub_graph_list.append(g)
    return sub_graph_list


# 在两个node之间插入一个 cast node
def insert_cast_node_between(graph: onnx_proto.GraphProto, upstream_node, downstream_node, to_type, id=0):
    # Insert cast node between two nodes
    if isinstance(upstream_node, onnx_proto.NodeProto) and isinstance(downstream_node, onnx_proto.NodeProto):
        for output_name in upstream_node.output:
            for i, input_name in enumerate(downstream_node.input):
                if output_name == input_name:
                    value_info = find_value_info_by_name(graph, input_name)
                    if value_info is not None and value_info.type.tensor_type.elem_type == onnx_proto.TensorProto.FLOAT:
                        cast_node_name = downstream_node.name + "_input_cast" + str(i)
                        cast_node_output_name = downstream_node.name + "_input_cast_" + str(i)
                        add_cast_node(graph, [output_name], [cast_node_output_name], cast_node_name, to_type)
                        downstream_node.input[i] = cast_node_output_name
                        return cast_node_output_name  # this will be put into value_info_block_list
                    else:
                        return None
    # Insert cast node between graph input(x) and node
    elif isinstance(upstream_node, onnx_proto.ValueInfoProto) and isinstance(downstream_node, onnx_proto.NodeProto):
        if upstream_node.type.tensor_type.elem_type in [onnx_proto.TensorProto.FLOAT, onnx_proto.TensorProto.FLOAT16]:
            cast_node_name = "graph_input_cast_" + upstream_node.name + str(id)
            cast_node_output_name = "graph_input_cast_" + upstream_node.name + "_" + str(id)
            add_cast_node(graph, [upstream_node.name], [cast_node_output_name], cast_node_name, to_type)
            for i, input_name in enumerate(downstream_node.input):
                if input_name == upstream_node.name:
                    downstream_node.input[i] = cast_node_output_name
            return cast_node_output_name
    # Insert cast node between node and graph output(z)
    elif isinstance(upstream_node, onnx_proto.NodeProto) and isinstance(downstream_node, onnx_proto.ValueInfoProto):
        if downstream_node.type.tensor_type.elem_type == onnx_proto.TensorProto.FLOAT:
            cast_node_name = "graph_output_cast_" + downstream_node.name + str(id)
            cast_node_input_name = "graph_output_cast_" + downstream_node.name + "_" + str(id)
            add_cast_node(graph, [cast_node_input_name], [downstream_node.name], cast_node_name, to_type)
            upstream_node.output[0] = cast_node_input_name
            return cast_node_input_name
    else:
        raise ValueError("upstream_node and downstream_node should be NodeProto or ValueInfoProto")

# 这里的 model 应该改成 graph，for sub-graph
def add_cast_node(graph: onnx_proto.GraphProto, inputs, outputs, node_name, to_type):
    new_node = [helper.make_node('Cast', inputs, outputs, to=to_type, name=node_name)]
    graph.node.extend(new_node)


def find_value_info_by_name(graph, name):
    for value_info in graph.value_info:
        if value_info.name == name:
            return value_info
    for value_info in graph.input:
        if value_info.name == name:
            return value_info
    for value_info in graph.output:
        if value_info.name == name:
            return value_info
    return None


# 通过 output name 找到所有的上游 node, 应该只有一个上有node
def find_upstream_node_by_output(graph: onnx_proto.GraphProto, output):
    nodes = []
    for node in graph.node:
        if output in node.output:
            nodes.append(node)
    assert len(nodes) <= 1
    if len(nodes) == 0:
        return None
    else:
        return nodes[0]


# 通过 input name 找到所有的下游 node
def find_donwstream_node_by_input(graph: onnx_proto.GraphProto, input):
    nodes = []
    for node in graph.node:
        if input in node.input:
            nodes.append(node)
    return nodes


def basic_info(model):
    print("---- node ----")
    for node in model.graph.node:
        print("-- node --")
        print(node)

    print("---- value info ----")
    for value_info in model.graph.value_info:
        print("-- value_info --")
        print(value_info)

    print("---- input ----")
    for input in model.graph.input:
        print("-- input --")
        print(input)
    
    print("---- output ----")
    for output in model.graph.output:
        print("-- output --")
        print(output)

    print("---- initializer ----")
    for initializer in model.graph.initializer:
        print("-- initializer --")
        print(initializer)


def convert_float_to_float16_old(model, min_positive_val=1e-7, max_finite_val=1e4,
                             keep_io_types=False, disable_shape_infer=False,
                             op_block_list=None, node_block_list=None):
    '''
    Convert tensor float type in the ONNX ModelProto input to tensor float16.

    :param model: ONNX ModelProto object
    :param disable_shape_infer: Type/shape information is needed for conversion to work.
                                Set to True only if the model already has type/shape information for all tensors.
    :return: converted ONNX ModelProto object

    Examples:

    ::

        Example 1: Convert ONNX ModelProto object:
        from onnxmltools.utils.float16_converter import convert_float_to_float16
        new_onnx_model = convert_float_to_float16(onnx_model)

        Example 2: Convert ONNX model binary file:
        from onnxmltools.utils.float16_converter import convert_float_to_float16
        from onnxmltools.utils import load_model, save_model
        onnx_model = load_model('model.onnx')
        new_onnx_model = convert_float_to_float16(onnx_model)
        save_model(new_onnx_model, 'new_model.onnx')

    '''
    func_infer_shape = None
    if not disable_shape_infer and pv.Version(onnx.__version__) >= pv.Version('1.2'):
        try:
            from onnx.shape_inference import infer_shapes
            func_infer_shape = infer_shapes
        finally:
            pass

    if not isinstance(model, onnx_proto.ModelProto):
        raise ValueError('Expected model type is an ONNX ModelProto but got %s' % type(model))

    # create blocklists
    if op_block_list is None:
        op_block_list = DEFAULT_OP_BLOCK_LIST
    if node_block_list is None:
        node_block_list = []
    op_block_list = set(op_block_list)
    node_block_list = set(node_block_list)
    # create a queue for BFS
    queue = []
    value_info_list = []
    node_list = []
    # key = node, value = graph, used to distinguish global with sub-graph
    node_dict = {}
    # type inference on input model
    if func_infer_shape is not None:
        model = func_infer_shape(model)
    queue.append(model)
    name_mapping = {}
    graph_io_to_skip = set()
    io_casts = set()
    if keep_io_types:  # the input dtype is float32, output dtype is float32
        for i, n in enumerate(model.graph.input):
            if n.type.tensor_type.elem_type == onnx_proto.TensorProto.FLOAT:
                output_name = 'graph_input_cast_' + str(i)
                name_mapping[n.name] = output_name
                graph_io_to_skip.add(n.name)

                node_name = 'graph_input_cast' + str(i)
                new_value_info = model.graph.value_info.add()
                new_value_info.CopyFrom(n)
                new_value_info.name = output_name
                new_value_info.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT16
                # add Cast node (from tensor(float) to tensor(float16) after graph input
                new_node = [helper.make_node('Cast', [n.name], [output_name], to=10, name=node_name)]
                model.graph.node.extend(new_node)
                value_info_list.append(new_value_info)
                io_casts.add(node_name)

        for i, n in enumerate(model.graph.output):
            if n.type.tensor_type.elem_type == onnx_proto.TensorProto.FLOAT:
                input_name = 'graph_output_cast_' + str(i)
                name_mapping[n.name] = input_name
                graph_io_to_skip.add(n.name)

                node_name = 'graph_output_cast' + str(i)
                # add Cast node (from tensor(float16) to tensor(float) before graph output
                new_value_info = model.graph.value_info.add()
                new_value_info.CopyFrom(n)
                new_value_info.name = input_name
                new_value_info.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT16
                new_node = [helper.make_node('Cast', [input_name], [n.name], to=1, name=node_name)]
                model.graph.node.extend(new_node)
                value_info_list.append(new_value_info)
                io_casts.add(node_name)

    while queue:
        next_level = []
        for q in queue:
            # if q is model, push q.graph (GraphProto)
            if isinstance(q, onnx_proto.ModelProto):
                next_level.append(q.graph)
            # if q is model.graph, push q.node.attribute (AttributeProto)
            if isinstance(q, onnx_proto.GraphProto):
                for n in q.node:
                    # if n is in the block list (doesn't support float16), no conversion for the node,
                    # and save the node for further processing
                    if n.name in io_casts:
                        continue
                    for i in range(len(n.input)):
                        if n.input[i] in name_mapping:
                            n.input[i] = name_mapping[n.input[i]]
                    for i in range(len(n.output)):
                        if n.output[i] in name_mapping:
                            n.output[i] = name_mapping[n.output[i]]
                    # don't add the attr into next_level for the node in node_keep_data_type_list
                    # so it will not be converted to float16
                    if n.op_type in op_block_list or n.name in node_block_list:
                        node_list.append(n)
                        node_dict[n.name] = q
                    else:
                        if n.op_type == 'Cast':
                            for attr in n.attribute:
                                if attr.name == 'to' and attr.i == 1:  # float32
                                    attr.i = 10  # float16. bug: if this cast is degined for next op(need cast to fp32), why force changing to fp16?
                                    break
                        for attr in n.attribute:
                            next_level.append(attr)
            # if q is model.graph.node.attribute, push q.g and q.graphs (GraphProto)
            # and process node.attribute.t and node.attribute.tensors (TensorProto)
            if isinstance(q, onnx_proto.AttributeProto):
                next_level.append(q.g)
                for n in q.graphs:
                    next_level.append(n)
                q.t.CopyFrom(convert_tensor_float_to_float16(q.t, min_positive_val, max_finite_val))
                for n in q.tensors:
                    n = convert_tensor_float_to_float16(n, min_positive_val, max_finite_val)
            # if q is graph, process graph.initializer(TensorProto), input, output and value_info (ValueInfoProto)
            if isinstance(q, onnx_proto.GraphProto):
                for n in q.initializer:  # TensorProto type
                    if n.data_type == onnx_proto.TensorProto.FLOAT:
                        n = convert_tensor_float_to_float16(n, min_positive_val, max_finite_val)
                        value_info_list.append(make_value_info_from_tensor(n))
                # for all ValueInfoProto with tensor(float) type in input, output and value_info, convert them to
                # tensor(float16) except map and seq(map). And save them in value_info_list for further processing
                for n in itertools.chain(q.input, q.output, q.value_info):
                    if n.type.tensor_type.elem_type == onnx_proto.TensorProto.FLOAT:
                        if n.name not in graph_io_to_skip:
                            n.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT16
                            value_info_list.append(n)
        queue = next_level

    # process the nodes in block list that doesn't support tensor(float16)
    for node in node_list:
        # if input's name is in the value_info_list meaning input is tensor(float16) type,
        # insert a float16 to float Cast node before the node,
        # change current node's input name and create new value_info for the new name
        for i in range(len(node.input)):
            input = node.input[i]
            for value_info in value_info_list:
                if input == value_info.name:
                    # create new value_info for current node's new input name
                    graph = node_dict[node.name]  # get the correct graph instead of the global graph
                    new_value_info = graph.value_info.add()
                    new_value_info.CopyFrom(value_info)
                    output_name = node.name + '_input_cast_' + str(i)
                    new_value_info.name = output_name
                    new_value_info.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT
                    # add Cast node (from tensor(float16) to tensor(float) before current node
                    node_name = node.name + '_input_cast' + str(i)
                    new_node = [helper.make_node('Cast', [input], [output_name], to=1, name=node_name)]
                    graph.node.extend(new_node)
                    # change current node's input name
                    node.input[i] = output_name
                    break
        # if output's name is in the value_info_list meaning output is tensor(float16) type, insert a float to
        # float16 Cast node after the node, change current node's output name and create new value_info for the new name
        for i in range(len(node.output)):
            output = node.output[i]
            for value_info in value_info_list:
                if output == value_info.name:
                    # create new value_info for current node's new output
                    graph = node_dict[node.name]  # get the correct graph instead of the global graph
                    new_value_info = graph.value_info.add()
                    new_value_info.CopyFrom(value_info)
                    input_name = node.name + '_output_cast_' + str(i)
                    new_value_info.name = input_name
                    new_value_info.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT
                    # add Cast node (from tensor(float) to tensor(float16) after current node
                    node_name = node.name + '_output_cast' + str(i)
                    new_node = [helper.make_node('Cast', [input_name], [output], to=10, name=node_name)]
                    graph.node.extend(new_node)
                    # change current node's input name
                    node.output[i] = input_name
                    break

    sort_topology(model.graph)
    remove_unnecessary_cast_node(model.graph)
    return model

# cast 原来是 to32 的，不要轻易改成 to16，看看后面的 op 是否需要 to32，如果需要，就不要改成 to16
# 反之，某个 op 前面有 cast 的，就不要再加新的 cast
# op 内部的 tensor 需要转成 16
# 保持 graph/sub-graph 不乱


def convert_float_to_float16_model_path(model_path, min_positive_val=1e-7, max_finite_val=1e4, keep_io_types=False):
    '''
    Convert tensor float type in the ONNX Model to tensor float16.
    *It is to fix an issue that infer_shapes func cannot be used to infer >2GB models.
    *But this function can be applied to all model sizes.
    :param model_path: ONNX Model path
    :return: converted ONNX ModelProto object
    Examples
    ::
        #Convert to ONNX ModelProto object and save model binary file:
        from onnxmltools.utils.float16_converter import convert_float_to_float16_model_path
        new_onnx_model = convert_float_to_float16_model_path('model.onnx')
        onnx.save(new_onnx_model, 'new_model.onnx')
    '''

    disable_shape_infer = False
    if pv.Version(onnx.__version__) >= pv.Version('1.8'):
        try:
            # infer_shapes_path can be applied to all model sizes
            from onnx.shape_inference import infer_shapes_path
            import tempfile
            import os
            # shape_infer_model_path should be in the same folder of model_path
            with tempfile.NamedTemporaryFile(dir=os.path.dirname(model_path)) as tmpfile:
                shape_infer_model_path = tmpfile.name
                infer_shapes_path(model_path, shape_infer_model_path)
                model = onnx.load(shape_infer_model_path)
                disable_shape_infer = True
        finally:
            pass
    if not disable_shape_infer:
        model = onnx.load(model_path)
    return convert_float_to_float16(model, min_positive_val, max_finite_val, keep_io_types, disable_shape_infer)


def sort_graph_node(graph_proto):
    # find the "first" node in Nodes that its input is not any node's output
    def find_first_node(output2node_dict):
        for node in org_nodes:
            is_not_first_node = any(item in output2node_dict for item in node.input)
            if not is_not_first_node:
                return node
        return None

    # remove the node from output2node_dict using output as key
    def remove_first_node_from_dict2(first_node):
        for output in first_node.output:
            if output in output2node_dict:
                del output2node_dict[output]

    org_nodes = graph_proto.node
    # create a dict to store output as key and node as value
    output2node_dict = {}
    for node in org_nodes:
        for output in node.output:
            output2node_dict[output] = node

    # save the final node after sorted
    sorted_node = []
    # traverse the Nodes to find the first node
    while (len(output2node_dict) > 0):
        first_node = find_first_node(output2node_dict)
        sorted_node.append(first_node)
        remove_first_node_from_dict2(first_node)
        # del node from original nodes list to avoid duplicate traverse
        org_nodes.remove(first_node)

    for new_node in sorted_node:
        graph_proto.node.extend([new_node])


# The input graph should be mode.graph
# Recursevly sort the topology for each sub-graph
def sort_topology(graph_proto):
    assert (isinstance(graph_proto, onnx_proto.GraphProto))
    sort_graph_node(graph_proto)  # sort global graph
    for node in graph_proto.node:
        for attr in node.attribute:
            if isinstance(attr.g, onnx_proto.GraphProto) and len(attr.g.node) > 0:
                sort_topology(attr.g)  # sort sub-graph
            for g in attr.graphs:
                if isinstance(g, onnx_proto.GraphProto):
                    sort_topology(g)  # sort sub-graph


def remove_unnecessary_cast_node(graph_proto):
    # 1. find all cast nodes in the graph
    cast_node_list = []
    input_name_to_cast_node_dict = {}
    output_name_to_cast_node_dict = {}
    # using name as key to point to a node. because node cannot be key
    name_to_node_dict = {}  
    for node in graph_proto.node:
        if node.op_type == 'Cast':
            if node.name not in ["graph_input_cast0", "graph_output_cast0"]:
                cast_node_list.append(node)
            
            name_to_node_dict[node.name] = node
            for input_name in node.input:
                input_name_to_cast_node_dict[input_name] = node
            for output_name in node.output:
                output_name_to_cast_node_dict[output_name] = node

    # 2. find upstream and downstream node of the cast node
    cast_node_upstream_dict = {}  # mapping cast node(name) to its upstream node
    cast_node_downstream_dict = {}   # mapping cast node(name) to its downstream node
    for current_node in graph_proto.node:
        # find the downstream node(s)
        for input_name in current_node.input:
            if input_name in output_name_to_cast_node_dict:
                # found the downstream node of the cast node, might be multiple
                cast_node = output_name_to_cast_node_dict[input_name]
                if cast_node.name not in cast_node_downstream_dict:
                    cast_node_downstream_dict[cast_node.name] = current_node
                else:  # already exists one downstream node, make it a list
                    existing_downstream_nodes = cast_node_downstream_dict[cast_node.name]
                    if isinstance(existing_downstream_nodes, list):
                        existing_downstream_nodes.append(current_node)
                    else:  # make a list
                        existing_downstream_nodes = [existing_downstream_nodes, current_node]
                        cast_node_downstream_dict[cast_node.name] = existing_downstream_nodes
        # find the upstream node
        for output_name in current_node.output:
            if output_name in input_name_to_cast_node_dict:
                # found the upstream node of the cast node, should be unique
                cast_node = input_name_to_cast_node_dict[output_name]
                cast_node_upstream_dict[cast_node.name] = current_node

    # 3. remove the cast node which upstream is 'Constant'
    for cast_node_name, upstream_node in cast_node_upstream_dict.items():
        cast_node = name_to_node_dict[cast_node_name]
        if upstream_node.op_type == 'Constant':
            cast_node_list.remove(cast_node)

    # 4. find the cast(to16) node which downstream is Cast(to32)
    remove_candidate = []
    for cast_node_name, downstream_node in cast_node_downstream_dict.items():
        cast_node = name_to_node_dict[cast_node_name]
        if isinstance(downstream_node, list):
            for dn in downstream_node:
                if dn.op_type == 'Cast' and \
                    dn.attribute[0].i == 32 and \
                    cast_node.attribute[0].i == 16 and \
                    dn in cast_node_list and \
                    cast_node in cast_node_list:
                    remove_candidate.append((cast_node, dn))
        else:
            if downstream_node.op_type == 'Cast' and \
                cast_node.attribute[0].i == 10 and \
                downstream_node.attribute[0].i == 1 and \
                downstream_node in cast_node_list and \
                cast_node in cast_node_list:
                remove_candidate.append((cast_node, downstream_node))

    # 5. change the connection of "upstream->cast16->cast32->downstream" to "upstream->downstream"
    for cast_node_pair in remove_candidate:
        first_cast_node = cast_node_pair[0]
        second_cast_node = cast_node_pair[1]
        upstream_node = cast_node_upstream_dict[first_cast_node.name]
        downstream_node = cast_node_downstream_dict[second_cast_node.name]
        # find the upstream node's output to first_cast_node
        out = None
        for output_name in upstream_node.output:
            if output_name == first_cast_node.input[0]:
                out = output_name
                break
        # find the downstream node's input as second_cast_node's output
        for i, input_name in enumerate(downstream_node.input):
            for output_name in second_cast_node.output:
                if input_name == output_name:
                    # change the input as the upstream node's output
                    downstream_node.input[i] = out

    # 6. remove the cast node pair
    for cast_node_pair in remove_candidate:
        graph_proto.node.remove(cast_node_pair[0])
        graph_proto.node.remove(cast_node_pair[1])
