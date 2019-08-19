# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
from fblib.layers.squeeze import ConvCoupledSE

affine_par = True


class AtrousSpatialPyramidPoolingModule(nn.Module):
    """
    Atrous Spatial Pyramid Pooling Module (DeepLab-v3+)
    """

    def __init__(self, dilation_series=None, depth=256, in_f=2048, cardinality=1, exist_decoder=True,
                 tasks=None, squeeze=False, adapters=False, se_after_relu=True, norm_per_task=True):

        super(AtrousSpatialPyramidPoolingModule, self).__init__()

        if dilation_series is None:
            dilation_series = [6, 12, 18]
        padding_series = dilation_series

        self.bnorm = nn.BatchNorm2d
        self.squeeze = squeeze

        kwargs = {"num_features": depth, "affine": affine_par}

        self.conv2d_list = nn.ModuleList()

        # 1x1 convolution
        self.conv2d_list.append(
            ConvCoupledSE(tasks=tasks,
                          process_layers=nn.Conv2d(in_f, depth, kernel_size=1, stride=1, bias=False),
                          norm=self.bnorm,
                          norm_kwargs=kwargs,
                          norm_per_task=norm_per_task,
                          squeeze=self.squeeze,
                          adapters=adapters,
                          se_after_relu=se_after_relu))

        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                ConvCoupledSE(tasks=tasks,
                              process_layers=nn.Conv2d(in_f, depth, kernel_size=3, stride=1, padding=padding,
                                                       dilation=dilation, bias=False, groups=cardinality),
                              norm=self.bnorm,
                              norm_kwargs=kwargs,
                              norm_per_task=norm_per_task,
                              squeeze=self.squeeze,
                              adapters=adapters,
                              se_after_relu=se_after_relu))

        # Global features
        self.conv2d_list.append(
            ConvCoupledSE(tasks=tasks,
                          process_layers=[nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                                          nn.Conv2d(in_f, depth, kernel_size=1, stride=1, bias=False)],
                          norm=self.bnorm,
                          norm_kwargs=kwargs,
                          norm_per_task=norm_per_task,
                          squeeze=self.squeeze,
                          se_after_relu=se_after_relu))

        if exist_decoder:
            self.conv2d_final = ConvCoupledSE(tasks=tasks,
                                              process_layers=nn.Conv2d(depth * 5, depth, kernel_size=1,
                                                                       stride=1, bias=False),
                                              norm=self.bnorm,
                                              norm_kwargs=kwargs,
                                              norm_per_task=norm_per_task,
                                              squeeze=self.squeeze,
                                              adapters=adapters,
                                              se_after_relu=se_after_relu)
        else:
            self.conv2d_final = nn.Sequential(nn.Conv2d(depth * 5, depth, kernel_size=1, stride=1, bias=True))

    def forward(self, x, task=None):
        h, w = x.size(2), x.size(3)

        interm = []
        for i in range(len(self.conv2d_list)):
            interm.append(self.conv2d_list[i](x, task))

        # Upsample the global features
        interm[-1] = F.interpolate(input=interm[-1], size=(h, w), mode='bilinear', align_corners=False)

        # Concatenate the parallel streams
        out = torch.cat(interm, dim=1)

        # Final convolutional layer of the classifier
        out = self.conv2d_final(out, task)

        return out


class ConvClassifier(nn.Module):
    """
    A simple convolutional classifier
    """

    def __init__(self, depth=256, in_f=2048, cardinality=1, exist_decoder=True,
                 tasks=None, squeeze=False, se_after_relu=True, norm_per_task=False):

        super(ConvClassifier, self).__init__()

        self.bnorm = nn.BatchNorm2d
        self.squeeze = squeeze

        kwargs = {"num_features": depth, "affine": affine_par}

        self.conv2d = ConvCoupledSE(tasks=tasks,
                                    process_layers=nn.Conv2d(in_f, depth, kernel_size=3, stride=1,
                                                             padding=1, dilation=1,
                                                             bias=False, groups=cardinality),
                                    norm=self.bnom,
                                    norm_kwargs=kwargs,
                                    norm_per_task=norm_per_task,
                                    squeeze=self.squeeze,
                                    se_after_relu=se_after_relu)

        if exist_decoder:
            self.conv2d_final = ConvCoupledSE(tasks=tasks,
                                              process_layers=nn.Conv2d(depth, depth, kernel_size=3, stride=1,
                                                                       padding=1, dilation=1, bias=False),
                                              norm=self.bnorm,
                                              norm_kwargs=kwargs,
                                              norm_per_task=norm_per_task,
                                              squeeze=self.squeeze,
                                              se_after_relu=se_after_relu)
        else:
            self.conv2d_final = nn.Sequential(nn.Conv2d(depth * 5, depth, kernel_size=1, stride=1, bias=True))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, self.bnorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def forward(self, x, task=None):

        x = self.conv2d(x, task)

        # Final convolutional layer of the classifier
        x = self.conv2d_final(x, task)

        return x
