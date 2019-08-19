# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np
import torch
from torch.nn import functional as F


def logit(x):
    return np.log(x/(1-x+1e-08)+1e-08)


def sigmoid_np(x):
    return 1/(1+np.exp(-x))


def center_crop(x, height, width):
    crop_h = torch.FloatTensor([x.size()[2]]).sub(height).div(-2)
    crop_w = torch.FloatTensor([x.size()[3]]).sub(width).div(-2)

    # fixed indexing for PyTorch 0.4
    return F.pad(x, [int(crop_w.ceil()[0]), int(crop_w.floor()[0]), int(crop_h.ceil()[0]), int(crop_h.floor()[0])])


def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)


def interp_surgery(lay):
    """
    Set parameters s.t. deconvolutional layers compute bilinear interpolation
    Only for deconvolution without groups
    """
    m, k, h, w = lay.weight.data.size()
    if m != k:
        print('input + output channels need to be the same')
        raise ValueError
    if h != w:
        print('filters need to be square')
        raise ValueError
    filt = upsample_filt(h)

    for i in range(m):
        lay.weight[i, i, :, :].data.copy_(torch.from_numpy(filt))

    return lay.weight.data

