# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionModuleFree(nn.Module):
    """
    Attention Module
    """
    def __init__(self, input_size, offset=0.):
        super(AttentionModuleFree, self).__init__()

        # randomly initialize parameters
        self.weight = nn.Parameter(torch.rand(1, input_size, 1, 1) + offset)

    def forward(self, x):
        return torch.mul(self.weight, x)


class AttentionModule(AttentionModuleFree):
    """
    AttentionModuleFree with restricted real-valued parameters within range [0, 1]
    """
    def __init__(self, input_size):
        super(AttentionModule, self).__init__(input_size, offset=10)

        # randomly initialize the parameters
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        return torch.mul(self.sigm(self.weight), x)


class Conv2dAttentionAdapters(nn.Module):
    """
    2D convolution followed by optional per-task transformation. The transformation can include the following:
        - Residual adapters (in parallel)
        - Attention modules (per-task feature multiplications) with gating, which can be binary or real-valued
    During forward pass, except for the input tensor, the index of the task is required
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False,
                 n_tasks=1,
                 adapters=False,
                 attention=False,
                 bn_per_task=False,
                 binary_attention=False):

        super(Conv2dAttentionAdapters, self).__init__()

        self.adapters = adapters
        self.attention = attention
        self.bn_per_task = bn_per_task and (self.adapters or self.attention)

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation,
                              groups=groups, bias=bias)

        if self.attention:
            print('Constructing attention modules.')
            if binary_attention:
                print('Binary attention!')
                att_module = AttentionModuleFree
            else:
                att_module = AttentionModule

            self.attend = nn.ModuleList([att_module(out_channels) for i in range(n_tasks)])

        if self.adapters:
            print('Constructing parallel residual adapters.')
            self.adapt = nn.ModuleList([
                nn.Conv2d(in_channels, out_channels, stride=stride, kernel_size=1, bias=False)for i in range(n_tasks)])

        if self.bn_per_task:
            print('Constructing per task batchnorm layers')
            self.bn = nn.ModuleList([nn.BatchNorm2d(out_channels) for i in range(n_tasks)])
        else:
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x, task=None):
        if self.adapters:
            adapt = self.adapt[task](x)

        x = self.conv(x)

        if self.attention:
            # print('Attend, task {}'.format(task))
            x = self.attend[task](x)

        if self.adapters:
            # print('adapt, task {}'.format(task))
            x += adapt

        if self.bn_per_task:
            # print('Bnorm, task {}'.format(task))
            x = self.bn[task](x)
        else:
            x = self.bn(x)

        return x


class XPathLayer(nn.Module):
    """
    Create per task ResNeXt path
    """

    def __init__(self,
                 in_channels,
                 interm_channels,
                 out_channels,
                 stride,
                 n_tasks):

        super(XPathLayer, self).__init__()

        self.conv_reduce = nn.ModuleList([nn.Conv2d(in_channels=in_channels,
                                                    out_channels=interm_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0,
                                                    bias=False) for i in range(n_tasks)])
        self.bn_reduce = nn.ModuleList([nn.BatchNorm2d(interm_channels) for i in range(n_tasks)])

        self.conv = nn.ModuleList([nn.Conv2d(in_channels=interm_channels,
                                             out_channels=interm_channels,
                                             kernel_size=3,
                                             stride=stride,
                                             padding=1) for i in range(n_tasks)])
        self.bn = nn.ModuleList([nn.BatchNorm2d(interm_channels) for i in range(n_tasks)])

        self.conv_expand = nn.ModuleList([nn.Conv2d(in_channels=interm_channels,
                                                    out_channels=out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0,
                                                    bias=False) for i in range(n_tasks)])
        self.bn_expand = nn.ModuleList([nn.BatchNorm2d(out_channels) for i in range(n_tasks)])

    def forward(self, x, task=None):

        if task is None:
            raise NotImplementedError('XPathLayer: Task not given at forward pass')

        # Reduce
        x = self.conv_reduce[task](x)
        x = self.bn_reduce[task](x)
        x = F.relu(x, inplace=True)

        # Process
        x = self.conv[task](x)
        x = self.bn[task](x)
        x = F.relu(x, inplace=True)

        # Expand
        x = self.conv_expand[task](x)
        x = self.bn_expand[task](x)

        return x
