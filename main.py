import argparse
import logging
import sys
from typing import Tuple, List

import onnx
import torch
from onnx import ModelProto, ValueInfoProto
from torch import nn

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger('obsidian')
logger.setLevel(logging.INFO)


# noinspection PyMethodMayBeStatic
class YoloTensorParser(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.transpose(1, 2)
        boxes = x[:, :, :4]
        scores, classes = torch.max(x[:, :, 4:], 2, keepdim=True)
        classes = classes.float()
        return boxes, scores, classes


def load_onnx(p_model: str) -> Tuple[ModelProto, ValueInfoProto]:
    logger.info(f'Loading model from: {p_model}')

    model: ModelProto = onnx.load_model(p_model)
    onnx.checker.check_model(model)

    output_nodes = [node for node in model.graph.output]

    if len(output_nodes) != 1:
        raise IOError('Expected exactly 1 output layer')

    layer_out: ValueInfoProto = output_nodes[0]

    logger.info('Model loaded')
    return model, layer_out


def create_tail_onnx(dims: Tuple[int], opset_ver: int, p_out: str):
    model = YoloTensorParser()
    onnx_input_im = torch.zeros(*dims)

    logger.info(f'Exporting tensor parser to: {p_out}')

    torch.onnx.export(
        model,
        onnx_input_im,
        p_out,
        verbose=False,
        opset_version=opset_ver,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['boxes', 'scores', 'classes'],
        dynamic_axes=None
    )


def fuse_models(m1: ModelProto, m2: ModelProto, io_map: Tuple[str, str], p_out: str):
    logger.info('Fusing models...')
    fused: ModelProto = onnx.compose.merge_models(
        m1=m1,
        m2=m2,
        io_map=[io_map]
    )

    onnx.save_model(fused, p_out)


def main(argv: List[str]):
    parser = argparse.ArgumentParser('Obsidian')

    parser.add_argument('-i', '--input', help='Path to input onnx model', type=str, required=True)
    parser.add_argument('-t', '--tmp-out', help='Path to intermediate onnx file (the tensor parser)', type=str,
                        required=True)
    parser.add_argument('-o', '--output', help='Path to output onnx file', type=str, required=True)

    args = parser.parse_args(argv)

    logger.info(f'Running with: {args}')

    model, layer_out = load_onnx(args.input)
    output_shape = tuple(dim.dim_value for dim in layer_out.type.tensor_type.shape.dim)

    logger.info(f'Model output has shape: {output_shape}')

    create_tail_onnx(output_shape, model.opset_import[0].version, args.tmp_out)
    tail: ModelProto = onnx.load_model(args.tmp_out)

    fuse_models(model, tail, (layer_out.name, 'input'), args.output)


if __name__ == "__main__":
    main(sys.argv[1:])
