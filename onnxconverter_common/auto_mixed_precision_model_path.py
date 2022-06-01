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

    model_fp16 = auto_convert_mixed_precision_model_path(model_path, test_data, validate, keep_io_types=True)
    onnx.save(model_fp16, "ouptut_path")

"""

import copy
import onnxruntime as ort
import onnx
import numpy as np
from onnxconverter_common import float16
from onnx import helper, mapping
from onnx import ModelProto
from onnx import shape_inference
from auto_mixed_precision import SegmentList, NodeSegment


def auto_convert_mixed_precision_model_path(source_model_path, input_feed,
                                            validate_fn=None, rtol=None, atol=None,
                                            keep_io_types=False, providers=None):
    """
    Automatically converts a model to mixed precision, excluding the minimum number of nodes required to
    ensure valudate_fn returns True and/or results are equal according to rtol/atol
    this version support model_path as input, which the model could > 2GB
    """
    if isinstance(source_model_path, ModelProto):
        raise TypeError('auto_convert_mixed_precision_model_path only accepts model Path (String),'
                        'you can use auto_convert_mixed_precision for the ModelProto.')

    if rtol is None and atol is not None:
        rtol = 1e-5

    if atol is None and rtol is not None:
        atol = 1e-8

    if rtol is None and validate_fn is None:
        raise ValueError("Argument `validate_fn` and `rtol` cannot both be `None`.")

    def validate(res1, res2):
        if validate_fn is not None and not validate_fn(res1, res2):
            return False
        if rtol is not None:
            for r1, r2 in zip(res1, res2):
                if not np.allclose(r1, r2, rtol, atol):
                    return False
        return True

    model_path = "a.tmp"
    shape_inference.infer_shapes_path(source_model_path, model_path)
    model0 = onnx.load(model_path)
    model0 = add_missing_dtypes_using_ort_model_path(model_path, model0, input_feed, providers=providers)
    res0 = get_tensor_values_using_ort_model_path(model_path, model0, input_feed, providers=providers)
    if not keep_io_types:
        input_feed = {k: v.astype(np.float16) if v.dtype == np.float32 else v for k, v in input_feed.items()}
    if not validate(res0, res0):
        raise ValueError("validation failed for original fp32 model")
    node_names = [n.name for n in model0.graph.node if n.op_type not in ["Loop", "If", "Scan"]]

    def run_attempt(node_block_list, return_model=False):
        print("node block list")
        print(node_block_list)
        model = float16.convert_float_to_float16(copy.deepcopy(model0), node_block_list=node_block_list,
                                                 keep_io_types=keep_io_types, disable_shape_infer=True)

        # need to save model to model_path here.....
        print("******** save model 000 ********")
        onnx.save(model, model_path, save_as_external_data=True)
        print("******** save complete ********")

        res1 = get_tensor_values_using_ort_model_path(model_path, model, input_feed, providers=providers)
        if return_model:
            return validate(res0, res1), model
        else:
            valid = validate(res0, res1)
            print(valid)
            return valid

    if not run_attempt(node_names):
        raise ValueError("validation failed for model with all nodes in node_block_list")
    print("Sanity checks passed. Starting autoconvert.")
    segments = SegmentList(node_names)
    i = 0
    while segments.get_largest() is not None:
        seg = segments.get_largest()
        nodes_to_try = segments.get_nodes(seg)
        i += 1
        print("Running attempt %d excluding conversion of %s nodes" % (i, len(nodes_to_try)))
        if run_attempt(nodes_to_try):
            seg.good = True
            print("Attempt succeeded.")
        else:
            print("Attempt failed.")
            if seg.size == 1:
                seg.bad = True
            else:
                seg.split()
        print("segments=")
        print(segments)
    print("Done:", segments.get_nodes())
    valid, model = run_attempt(segments.get_nodes(), return_model=True)
    if not valid:
        raise ValueError("validation failed for final fp16 model")
    print("Final model validated successfully.")
    return model


def add_missing_dtypes_using_ort_model_path(model_path, model, input_feed, outputs_per_iter=100, providers=None):
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
        vals = get_tensor_values_using_ort_model_path(model_path, model, input_feed, outs, providers=providers)
        for out, val in zip(outs, vals):
            out_to_dtype[out] = mapping.NP_TYPE_TO_TENSOR_TYPE[val.dtype]
        i += outputs_per_iter
    need_to_save_model = False
    for out, dtype in out_to_dtype.items():
        model.graph.value_info.append(helper.make_tensor_value_info(out, dtype, shape=None))
        need_to_save_model = True

    # if model changed, need to save model to model_path here.....
    if need_to_save_model:
        print("******** save model 111********")
        onnx.save(model, model_path, save_as_external_data=True)
        print("******** save complete ********")

    return model


# need input both model_path (for big model inference) and model (Proto, in memory, for manipulation)
def get_tensor_values_using_ort_model_path(model_path, model, input_feed, output_names=None,
                                           sess_options=None, providers=None):
    if output_names is None:
        sess = ort.InferenceSession(model_path, sess_options, providers=providers)
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
    if need_to_save_model:
        print("******** save model 222********")
        onnx.save(model, model_path, save_as_external_data=True)
        print("******** save complete ********")

    sess = ort.InferenceSession(model_path, sess_options, providers=providers)
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
        if need_to_save_model:
            print("******** save model 333********")
            onnx.save(model, model_path, save_as_external_data=True)
            print("******** save complete ********")
