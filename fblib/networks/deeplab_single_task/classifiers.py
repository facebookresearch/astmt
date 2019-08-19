# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch.nn as nn
from torch.nn import functional as F
import torch

affine_par = True


class AtrousSpatialPyramidPoolingModule(nn.Module):
    """
    Atrous Spatial Pyramid Pooling Module (DeepLab-v3+)
    """
    def __init__(self, dilation_series=[6, 12, 18], depth=256, in_f=2048, groupnorm=False, sync_bnorm=False,
                 cardinality=1, exist_decoder=True):
        super(AtrousSpatialPyramidPoolingModule, self).__init__()

        padding_series = dilation_series
        self.conv2d_list = nn.ModuleList()
        self.bnorm = nn.BatchNorm2d

        if not groupnorm:
            NormModule = self.bnorm
            kwargs = {"num_features": depth, "affine": affine_par}
        else:
            NormModule = nn.GroupNorm
            kwargs = {"num_groups": 16, "num_channels": depth, "affine": affine_par}

        # 1x1 convolution
        self.conv2d_list.append(nn.Sequential(nn.Conv2d(in_f, depth, kernel_size=1, stride=1, bias=False),
                                NormModule(**kwargs),
                                nn.ReLU(inplace=True)))

        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Sequential(nn.Conv2d(in_f, depth, kernel_size=3, stride=1, padding=padding,
                                                            dilation=dilation, bias=False, groups=cardinality),
                                                  NormModule(**kwargs),
                                                  nn.ReLU(inplace=True)))

        # Global features
        self.conv2d_list.append(nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                                              nn.Conv2d(in_f, depth, kernel_size=1, stride=1, bias=False),
                                              NormModule(**kwargs),
                                              nn.ReLU(inplace=True)))

        if exist_decoder:
            self.conv2d_final = nn.Sequential(nn.Conv2d(depth * 5, depth, kernel_size=1, stride=1, bias=False),
                                              NormModule(**kwargs),
                                              nn.ReLU(inplace=True))
        else:
            self.conv2d_final = nn.Sequential(nn.Conv2d(depth * 5, depth, kernel_size=1, stride=1, bias=True))

        for m in self.conv2d_list:
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, self.bnorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        for m in self.conv2d_final:
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, self.bnorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        h, w = x.size(2), x.size(3)

        interm = []
        for i in range(len(self.conv2d_list)):
            interm.append(self.conv2d_list[i](x))

        # Upsample the global features
        interm[-1] = F.interpolate(input=interm[-1], size=(h, w), mode='bilinear', align_corners=False)

        # Concatenate the parallel streams
        out = torch.cat(interm, dim=1)

        # Final convolutional layer of the classifier
        out = self.conv2d_final(out)

        return out


class AtrousPyramidModule(nn.Module):
    """
    Atrous Pyramid Module (DeepLab-v2)
    """
    def __init__(self, dilation_series, padding_series, n_classes, in_f=2048):
        super(AtrousPyramidModule, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2d(in_f, n_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
        return out


class PSPModule(nn.Module):
    """
    Pyramid Scene Parsing Module (PSP Net)
    """
    def __init__(self, in_features=2048, out_features=512, sizes=(1, 2, 3, 6), n_classes=1,
                 static_graph=False, groupnorm=False, sync_bnorm=False):

        super(PSPModule, self).__init__()

        self.groupnorm = groupnorm
        self.stages = []
        self.static_graph = static_graph
        self.stages = nn.ModuleList([self._make_stage_1(in_features, size) for size in sizes])
        self.bottleneck = self._make_stage_2(in_features * (len(sizes)//4 + 1), out_features)
        self.n_classes = n_classes
        self.bnorm = nn.BatchNorm2d

        if self.n_classes > 0:
            self.final = nn.Conv2d(out_features, n_classes, kernel_size=1)

    def _make_stage_1(self, in_features, size):
        if self.static_graph:
            # For input_image = 256
            # kernel_size = {1: 32, 2: 16, 3: 10, 6: 5}
            # For input_image = 512 The stride for level 6 is not the same as in AdaptiveAvgPool2d
            kernel_stride_size = {1: [64, 64], 2: [32, 32], 3: [22, 21], 6: [11, 9]}
            prior = nn.AvgPool2d(kernel_size=kernel_stride_size[size][0], stride=kernel_stride_size[size][1])
        else:
            prior = nn.AdaptiveAvgPool2d(output_size=(size, size))

        conv = nn.Conv2d(in_features, in_features//4, kernel_size=1, bias=False)

        if not self.groupnorm:
            bn = self.bnorm(num_features=in_features//4, affine=affine_par)
        else:
            bn = nn.GroupNorm(num_groups=16, num_channels=in_features//4, affine=affine_par)

        relu = nn.ReLU(inplace=True)

        return nn.Sequential(prior, conv, bn, relu)

    def _make_stage_2(self, in_features, out_features):
        conv = nn.Conv2d(in_features, out_features, kernel_size=1, bias=False)

        if not self.groupnorm:
            bn = self.bnorm(num_features=out_features, affine=affine_par)
        else:
            bn = nn.GroupNorm(num_groups=32, num_channels=out_features, affine=affine_par
                              )
        relu = nn.ReLU(inplace=True)

        return nn.Sequential(conv, bn, relu)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=False)
                  for stage in self.stages]

        priors.append(feats)
        bottle = self.bottleneck(torch.cat(priors, 1))
        if self.n_classes > 0:
            out = self.final(bottle)
        else:
            out = bottle
        return out
