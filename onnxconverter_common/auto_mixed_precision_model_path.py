
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###########################################################################

"""
This tool converts converts a model to mixed precision (float32->float16) while excluding nodes as needed to maintain
a certain accuracy.

Example usage:

    from onnxconverter_common import auto_mixed_precision
    import onnx

    # Could also use rtol/atol attributes directly instead of this
    def validate(res1, res2):
        for r1, r2 in zip(res1, res2):
            if not np.allclose(r1, r2, rtol=0.01, atol=0.001):
                return False
        return True

    auto_convert_mixed_precision_model_path(
        source_model_path, test_data, 
        target_model_path, location=None,
        validate_fn=None, rtol=None, atol=None,
        keep_io_types=False, providers=None)
    
    The target fp16 model will be saved as %target_model_path.
    It is better to specify a unique output folder.

"""

import copy
import numpy as np
import onnxruntime as ort
import onnx
import os
import shutil
import uuid
from onnxconverter_common import float16
from onnx import helper, mapping
from onnx import ModelProto
from onnx import shape_inference
from .auto_mixed_precision import SegmentList


def auto_convert_mixed_precision_model_path(source_model_path, input_feed,
                                            target_model_path, location=None,
                                            validate_fn=None, rtol=None, atol=None,
                                            keep_io_types=False, providers=None):
    """
    Automatically converts a model to mixed precision, excluding the minimum number of nodes required to
    ensure valudate_fn returns True and/or results are equal according to rtol/atol
    this version support model_path as input, which the model could > 2GB
    """
    
    print("**** source %s, target %s, location %s" %(source_model_path, target_model_path, location))
    
    if isinstance(source_model_path, ModelProto):
        raise TypeError('auto_convert_mixed_precision_model_path only accepts model Path (String),'
                        'you can use auto_convert_mixed_precision for the ModelProto.')

    if rtol is None and atol is not None:
        rtol = 1e-5

    if atol is None and rtol is not None:
        atol = 1e-8

    if rtol is None and validate_fn is None:
        raise ValueError("Argument `validate_fn` and `rtol` cannot both be `None`.")
    
    if location is None:
        location = "fp16_tensor.data"
    
    tmp_model32_path, tmp_model32_tensor_name = generate_temp_filename(target_model_path)

    kwargs = {
        "tmp_model32_path": tmp_model32_path,
        "tmp_model32_tensor_name": tmp_model32_tensor_name,
        "source_model_path": source_model_path,
        "input_feed": input_feed,
        "target_model_path": target_model_path,
        "location": location,
        "validate_fn": validate_fn,
        "rtol": rtol,
        "atol": atol,
        "keep_io_types": keep_io_types,
        "providers": providers
        }

    print("**** copy source model to temp folder, change to external data, then check ****")  
    model_32, output_32 = copy_fp32_model(**kwargs)

    print("**** convert to initial fp16 model, then check ****")
    node_names = [n.name for n in model_32.graph.node if n.op_type not in ["Loop", "If", "Scan"]]
    print_node_block_list(node_names)
    kwargs["model_32"] = model_32
    kwargs["res1"] = output_32
    kwargs["node_block_list"] = node_names
    kwargs["return_model"] = False
    run_attempt(**kwargs)

    print("Sanity checks passed. Starting autoconvert.")
    final_nodes = try_to_convert_to_valid_fp16_model(**kwargs)

    print("**** final convert ****")
    kwargs["node_block_list"] = final_nodes
    kwargs["return_model"] = True
    valid, model = run_attempt(**kwargs)
    if not valid:
        raise ValueError("validation failed for final fp16 model")
    print("Final model validated successfully.")
    
    clean_output_folder(**kwargs)
    
    return model


def try_to_convert_to_valid_fp16_model(**kwargs):
    print(" **** try_to_convert_to_valid_fp16_mode ****")
    node_names = kwargs.get('node_block_list')
    
    segments = SegmentList(node_names)
    i = 0
    while segments.get_largest() is not None:
        seg = segments.get_largest()
        nodes_to_try = segments.get_nodes(seg)
        i += 1
        print("Running attempt %d excluding conversion of %s nodes" % (i, len(nodes_to_try)))
        kwargs["node_block_list"] = nodes_to_try
        if run_attempt(**kwargs):
            seg.good = True
            print("Attempt succeeded.")
        else:
            print("Attempt failed.")
            if seg.size == 1:
                seg.bad = True
            else:
                seg.split()
        print("segments=", segments)
    print("**** Done! these nodes will keep float32 type:", segments.get_nodes())
    
    return segments.get_nodes()


def generate_temp_filename(target_model_path):
    target_model_folder = os.path.dirname(target_model_path)
    if not os.path.exists(target_model_folder):
        os.mkdir(target_model_folder)
    tensor_filename = str(uuid.uuid1())
    onnx_filename = os.path.join(target_model_folder, tensor_filename + ".onnx")
    return onnx_filename, tensor_filename + ".data"


def copy_fp32_model(**kwargs):
    print("****copy_fp32_model****")

    source_model_path = kwargs.get('source_model_path')
    input_feed = kwargs.get('input_feed')
    providers = kwargs.get('providers')
    tmp_model32_path = kwargs.get("tmp_model32_path")
    tmp_model32_tensor_name = kwargs.get("tmp_model32_tensor_name")

    model_32 = onnx.load(source_model_path)
    save_model(True, model_32, tmp_model32_path, location=tmp_model32_tensor_name)
    
    print("infer_shape_path for", tmp_model32_path, tmp_model32_tensor_name)
    shape_inference.infer_shapes_path(tmp_model32_path)
    model_32 = onnx.load(tmp_model32_path)

    print("**** run fp32 inference")
    output_32 = inference(tmp_model32_path, input_feed, providers=providers)
    print("****", output_32)

    kwargs["res1"] = output_32
    kwargs["res2"] = output_32
    if not validate(**kwargs):
      raise ValueError("validation failed for fp32 model")

    return model_32, output_32

def validate(**kwargs):
    print("****validate****")

    validate_fn = kwargs.get("validate_fn")
    rtol = kwargs.get("rtol")
    res1 = kwargs.get("res1")
    res2 = kwargs.get("res2")
    if validate_fn is not None and not validate_fn(res1, res2):
        return False
    if rtol is not None:
        for r1, r2 in zip(res1, res2):
            if not np.allclose(r1, r2, rtol):
                return False
    return True


def run_attempt(**kwargs):
    print("****run_attempt****")

    model_32 = kwargs.get("model_32")
    keep_io_types = kwargs.get("keep_io_types")
    return_model = kwargs.get("return_model")
    target_model_path = kwargs.get("target_model_path")
    node_block_list = kwargs.get("node_block_list")
    input_feed = kwargs.get("input_feed")
    providers = kwargs.get("providers")
    tmp_model32_tensor_name = kwargs.get("tmp_model32_tensor_name")

    print_node_block_list(node_block_list)
    # convert to fp16
    model_16 = float16.convert_float_to_float16(
        copy.deepcopy(model_32), node_block_list=node_block_list,
        keep_io_types=keep_io_types, disable_shape_infer=True)
    # need to save model to model_path here.....
    if not return_model:
        location = tmp_model32_tensor_name
    else:
        location = kwargs.get("location")  # using the speficified external data file name
    save_model(True, model_16, target_model_path, location=location)     
    # inference
    output_16 = inference(target_model_path, input_feed, providers=providers)
    kwargs["res2"] = output_16
    result = validate(**kwargs)
    print("validate result = ", result)
    if return_model:
        return result, model_16
    else:
        return result


def inference(model_path, input_feed, providers=None):
    print(" **** inference ****")
    sess = ort.InferenceSession(model_path, None, providers=providers)
    output = sess.run(None, input_feed)
    return output


def save_model(need_to_save_model, model, model_path, location=None):
    if need_to_save_model:
        print(" **** save model: model path is: %s external file name is: %s****", model_path, location)
        onnx.save(model, model_path, save_as_external_data=True, location=location)
        print("**** save model complete ****")


def print_node_block_list(node_block_list, max_len=128):
        print("node block list =")
        if (len(node_block_list) < max_len):
            print(node_block_list)
        else:
            tmp_list = node_block_list[0:64] + ['......'] + node_block_list[-64:]
            print(tmp_list)


def clean_output_folder(**kwargs):
    tmp_model32_path = kwargs.get("tmp_model32_path")
    tmp_model32_tensor_name = kwargs.get("tmp_model32_tensor_name")
    os.remove(tmp_model32_path)
    tensor_path = os.path.join(os.path.dirname(tmp_model32_path), tmp_model32_tensor_name)
    os.remove(tensor_path)


def add_missing_dtypes_using_ort_model_path(source_model_path, model, input_feed, 
                                            outputs_per_iter=100, providers=None):
    outputs = [out for node in model.graph.node for out in node.output]
    graph_io = [inp.name for inp in model.graph.input] + [out.name for out in model.graph.output]
    value_info_names = [info.name for info in model.graph.value_info]
    skip = set(graph_io + value_info_names)
    outputs = [out for out in outputs if out not in skip]
    print("Adding missing dtypes for %s outputs" % len(outputs))
    out_to_dtype = {}
    i = 0
    while i < len(outputs):
        outs = outputs[i:i + outputs_per_iter]
        vals = get_tensor_values_using_ort_model_path(source_model_path, model, input_feed, outs, providers=providers, location=location)
        for out, val in zip(outs, vals):
            out_to_dtype[out] = mapping.NP_TYPE_TO_TENSOR_TYPE[val.dtype]
        i += outputs_per_iter
    need_to_save_model = False
    for out, dtype in out_to_dtype.items():
        model.graph.value_info.append(helper.make_tensor_value_info(out, dtype, shape=None))
        need_to_save_model = True

    return model, need_to_save_model


# need input both model_path (for big model inference) and model (Proto, in memory, for manipulation)
def get_tensor_values_using_ort_model_path(target_model_path, model, input_feed, output_names=None,
                                           sess_options=None, providers=None, location=None):
    print("Inference Model : ", target_model_path)
    if output_names is None:
        sess = ort.InferenceSession(target_model_path, sess_options, providers=providers)
        return sess.run(None, input_feed)

    need_to_save_model = False
    original_outputs = list(model.graph.output)
    while len(model.graph.output) > 0:
        model.graph.output.pop()
    for n in output_names:
        out = model.graph.output.add()
        out.name = n
        need_to_save_model = True

    # if model changed, need to save model to model_path here.....
    save_model(need_to_save_model, model, target_model_path, location)

    sess = ort.InferenceSession(target_model_path, sess_options, providers=providers)
    try:
        return sess.run(output_names, input_feed)
    finally:
        need_to_save_model = False
        while len(model.graph.output) > 0:
            model.graph.output.pop()
        for orig_out in original_outputs:
            out = model.graph.output.add()
            out.CopyFrom(orig_out)
            need_to_save_model = True

        # if model changed, need to save model to model_path here.....
        save_model(need_to_save_model, model, target_model_path, location)
