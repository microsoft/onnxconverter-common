# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###########################################################################

"""
Converts onnx model into model.py file for easy editing. Resulting model.py file uses onnx.helper library to
recreate the original onnx model. Constant tensors with more than 10 elements are saved into .npy
files in location model/const#_tensor_name.npy

Example usage:
python -m onnxconverter_common.onnx2py my_model.onnx my_model.py
"""

import sys
import onnx
import collections
from collections import OrderedDict
from onnx import helper, numpy_helper, TensorProto
import numpy as np
import os

from .pytracing import TracingObject

needed_types = set()
const_dir = None
const_counter = None

np_traced = TracingObject("np")
helper_traced = TracingObject("helper")
numpy_helper_traced = TracingObject("numpy_helper")
TensorProtoTraced = TracingObject("TensorProto")
os_traced = TracingObject("os")


def convert_tensor_type(i):
    return getattr(TensorProtoTraced, TensorProto.DataType.Name(i))


def convert_field(field):
    global needed_types
    if isinstance(field, (int, str, float, bytes)):
        return field
    elif isinstance(field, onnx.GraphProto):
        return convert_graph(field)
    elif isinstance(field, onnx.ModelProto):
        return convert_model(field)
    elif isinstance(field, onnx.NodeProto):
        return convert_node(field)
    elif isinstance(field, onnx.TensorProto):
        return convert_tensor(field)
    elif isinstance(field, onnx.ValueInfoProto):
        return convert_value_info(field)
    elif isinstance(field, onnx.OperatorSetIdProto):
        return convert_operatorsetid(field)
    elif isinstance(field, collections.abc.Iterable):
        return list(convert_field(x) for x in field)
    else:
        # Missing handler needs to be added
        t = str(type(field))
        needed_types.add(t)
        return field


def convert_value_info(val_info):
    name = val_info.name
    elem_type = convert_tensor_type(val_info.type.tensor_type.elem_type)
    kwargs = OrderedDict()

    def convert_shape_dim(d):
        if d.HasField("dim_value"):
            return d.dim_value
        if d.HasField("dim_param"):
            return d.dim_param
        return None

    def convert_shape_denotation(d):
        if d.HasField("denotation"):
            return d.denotation
        return None

    kwargs["shape"] = [convert_shape_dim(d) for d in val_info.type.tensor_type.shape.dim]
    if any(d.HasField("denotation") for d in val_info.type.tensor_type.shape.dim):
        kwargs["shape_denotation"] = [convert_shape_denotation(d) for d in val_info.type.tensor_type.shape.dim]

    if val_info.HasField("doc_string"):
        kwargs["doc_string"].doc_string

    return helper_traced.make_tensor_value_info(name, elem_type, **kwargs)


def convert_operatorsetid(opsetid):
    domain = opsetid.domain
    version = opsetid.version
    return helper_traced.make_operatorsetid(domain, version)


def convert_tensor(tensor):
    global const_dir, const_counter
    np_data = numpy_helper.to_array(tensor)
    if np.product(np_data.shape) <= 10:
        return numpy_helper_traced.from_array(np_data, name=tensor.name)
    os.makedirs(const_dir, exist_ok=True)
    name = "const" + str(const_counter)
    if tensor.name:
        name = name + "_" + tensor.name
    for c in '~"#%&*:<>?/\\{|}':
        name = name.replace(c, '_')
    const_path = "%s/%s.npy" % (const_dir, name)
    np.save(const_path, np_data)
    rel_path = TracingObject("os.path.join(DATA_DIR, '%s.npy')" % name)
    const_counter += 1
    np_dtype = getattr(np_traced, str(np_data.dtype))
    np_shape = list(np_data.shape)
    return numpy_helper_traced.from_array(np_traced.load(rel_path).astype(np_dtype).reshape(np_shape), name=tensor.name)


def convert_node(node):
    fields = OrderedDict((f[0].name, f[1]) for f in node.ListFields())
    attributes = fields.pop("attribute", [])
    attrs = OrderedDict((a.name, convert_field(helper.get_attribute_value(a))) for a in attributes)
    fields = OrderedDict((f, convert_field(v)) for f, v in fields.items())
    op_type = fields.pop("op_type")
    if op_type == "Cast" and "to" in attrs:
        attrs["to"] = convert_tensor_type(attrs["to"])
    inputs = fields.pop("input", [])
    outputs = fields.pop("output", [])
    return helper_traced.make_node(op_type, inputs=inputs, outputs=outputs, **fields, **attrs)


def convert_graph(graph):
    fields = OrderedDict((f[0].name, convert_field(f[1])) for f in graph.ListFields())
    nodes = fields.pop("node", [])
    name = fields.pop("name")
    inputs = fields.pop("input", [])
    outputs = fields.pop("output", [])
    return helper_traced.make_graph(name=name, inputs=inputs, outputs=outputs, **fields, nodes=nodes)


def convert_model(model):
    fields = OrderedDict((f[0].name, convert_field(f[1])) for f in model.ListFields())
    graph = fields.pop("graph")
    opset_imports = fields.pop("opset_import", [])
    return helper_traced.make_model(opset_imports=opset_imports, **fields, graph=graph)


def clear_directory(path):
    for f in os.listdir(path):
        if f.endswith(".npy"):
            os.remove(os.path.join(path, f))
    try:
        # Delete if empty
        os.rmdir(path)
    except OSError:
        pass


class MissingHandlerException(Exception):
    pass


def convert(model, out_path):
    global needed_types, const_dir, const_counter
    needed_types = set()
    if out_path.endswith(".py"):
        out_path = out_path[:-3]
    if os.path.exists(out_path):
        clear_directory(out_path)
    const_dir = out_path
    const_dir_name = os.path.basename(out_path)
    const_counter = 0

    model_trace = convert_model(model)
    code = "from onnx import helper, numpy_helper, TensorProto\n"
    code += "import onnx\n"
    code += "import numpy as np\n"
    code += "import sys\n"
    if os.path.exists(const_dir):
        code += "import os\n"
        code += "\nDATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), %r)\n" % const_dir_name
    code += "\n" + "model = " + repr(model_trace) + "\n"
    code += "\nif __name__ == '__main__' and len(sys.argv) == 2:\n"
    code += "    _, out_path = sys.argv\n"
    code += "    onnx.save(model, out_path)\n"
    with open(out_path + ".py", "wt") as file:
        file.write(code)
    if needed_types:
        raise MissingHandlerException("Missing handler for types: %s" % list(needed_types))


def main():
    _, in_path, out_path = sys.argv
    if not out_path.endswith(".py"):
        out_path = out_path + ".py"

    model = onnx.load(in_path)
    try:
        convert(model, out_path)
    except MissingHandlerException as e:
        print("ERROR:", e)

    print("Model saved to", out_path)
    print("Run '%s output.onnx' to generate ONNX file" % out_path)
    print("Import the model with 'from %s import model'" % os.path.basename(out_path[:-3]))


if __name__ == '__main__':
    main()
