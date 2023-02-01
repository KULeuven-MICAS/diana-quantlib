from pathlib import Path
from typing import Optional
from quantlib.backends.base.onnxannotator import ONNXAnnotator
from quantlib.backends.base.onnxexporter import ONNXExporter
from quantlib.backends.dory.onnxexporter import DORYExporter

import onnxruntime as rt

import torch
from torch import nn
import os
import onnx


class DianaAnnotator(ONNXAnnotator):
    def __init__(self, backend_name: str):

        super(DianaAnnotator, self).__init__(backend_name)

    def _annotate(self, network: nn.Module, onnxproto: onnx.ModelProto):

        # Define backend-specific supported ONNX nodes. The nodes belonging to
        # different node classes will be annotated using a class-specific logic.
        dory_onnxnode_op_types = {
            "linear": {"Conv", "Gemm"},
            "mul": {"Mul"},
            "add": {"Add"},
            "clip": {"Clip"},
            "div": {"Div"},
        }

        for n in onnxproto.graph.node:

            op_type = n.op_type
            annotations = []

            if op_type in dory_onnxnode_op_types["linear"]:
                op_name = n.input[1].rsplit(".", 1)[0]
                pytorch_module = network.get_submodule(op_name)
                if isinstance(
                    pytorch_module,
                    (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d),
                ):
                    weight_bits = (
                        8
                        if not pytorch_module.is_analog == torch.Tensor([1])
                        else 2
                    )  # check register buffer inside conv layer to know if it's analog or not
                    bias_bits = 32
                    annotations.append(
                        onnx.helper.make_attribute(
                            key="weight_bits", value=weight_bits
                        )
                    )
                    annotations.append(
                        onnx.helper.make_attribute(
                            key="bias_bits", value=bias_bits
                        )
                    )
                    if isinstance(pytorch_module, nn.Conv2d):
                        annotations.append(
                            onnx.helper.make_attribute(
                                key="weight_data_layout", value="OIHW"
                            )
                        )
                        annotations.append(
                            onnx.helper.make_attribute(
                                key="input_data_layout", value="NCHW"
                            )
                        )

            elif (
                op_type in dory_onnxnode_op_types["mul"]
            ):  # only for analog core. still don't have specs
                mul_bits = self._requantisation_bits
                annotations.append(
                    onnx.helper.make_attribute(key="mult_bits", value=mul_bits)
                )

            elif op_type in dory_onnxnode_op_types["add"]:
                add_bits = 8
                annotations.append(
                    onnx.helper.make_attribute(key="add_bits", value=add_bits)
                )

            else:  # the backend does not require special handling for this node type
                pass

            # flush attributes to the ONNX node
            n.attribute.extend(annotations)


class DianaExporter(DORYExporter):
    def __init__(self):
        annotator = DianaAnnotator("DORY")
        ONNXExporter.__init__(self, annotator=annotator)

    def export(
        self,
        network: nn.Module,
        input_shape: torch.Size,
        path: os.PathLike,
        name: Optional[str] = None,
        opset_version: int = 10,
    ) -> None:
        onnxname = name if name is not None else network._get_name()
        onnxfilename = (
            onnxname + "_QL_NOANNOTATION.onnx"
        )  # TODO: should the name hint at whether the network is FP, FQ, or TQ? How can we derive this information (user-provided vs. inferred from the network)?
        onnxfilepath = os.path.join(path, onnxfilename)

        # export the network (https://pytorch.org/docs/master/onnx.html#torch.onnx.export)
        torch.onnx.export(
            network,
            torch.randn(input_shape),  # a dummy input to trace the `nn.Module`
            onnxfilepath,
            export_params=True,
            do_constant_folding=True,
            opset_version=opset_version,
        )

        options = rt.SessionOptions()

        # Set graph optimization level
        options.graph_optimization_level = (
            rt.GraphOptimizationLevel.ORT_ENABLE_BASIC
        )
        # To enable model serialization after graph optimization set this
        data_folder = path + "/" + onnxfilename
        options.optimized_model_filepath = data_folder
        # generate optimized graph from non-annotated model

        session = rt.InferenceSession(
            str(path) + f"/{onnxfilename}",
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        # read optimized graph and annotate it with backend-specific information
        self._annotator.annotate(network, data_folder)
