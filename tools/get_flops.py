# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table

from mmocr.apis import init_detector


def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, device='cpu')
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy

    w = 640
    h = 640

    input_shape = (3, h, w)

    print('input shape is ', input_shape)

    batch = torch.ones(()).new_empty((1, *input_shape),
                                     dtype=next(model.parameters()).dtype,
                                     device=next(model.parameters()).device)
    flops = FlopCountAnalysis(model, batch)

    flops_data = flop_count_table(flops)

    print(flops_data)


if __name__ == '__main__':
    main()

# pip install fvcore
# python tools/get_flops.py configs/textdet/dbnet/dbnet_r18_fpnc_1200e_icdar2015.py
