# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from torch import nn
from fblib.util.custom_container import SequentialMultiTask


class SELayer(nn.Module):
    """
    Squeeze and Excitation Layer
    """
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SELayerMultiTaskDict(nn.Module):
    """
    Squeeze and Excitation Layer for multiple tasks (dict)
    """
    def __init__(self, channel, reduction=16, tasks=None):
        super(SELayerMultiTaskDict, self).__init__()
        self.tasks = tasks

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        if self.tasks is None:
            self.fc = nn.Sequential(nn.Linear(channel, channel // reduction),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(channel // reduction, channel),
                                     nn.Sigmoid())
        else:
            print('Initializing squeeze and excitation modules:')
            self.fc = nn.ModuleDict()
            for task in self.tasks:
                print('SE for task: {}'.format(task))
                self.fc[task] = SequentialMultiTask(nn.Linear(channel, channel // reduction),
                                                    nn.ReLU(inplace=True),
                                                    nn.Linear(channel // reduction, channel),
                                                    nn.Sigmoid())

    def forward(self, x, task=None):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)

        if self.tasks:
            y = self.fc[task](y).view(b, c, 1, 1)
        else:
            y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ConvCoupledSE(nn.Module):
    """
    SE-layer per task, coupled with convolutions and batchnorm.
    Possibility to place convolutions before/after bn, deploy bn per task, and use/not use SE attention.
    """

    def __init__(self, tasks,
                 process_layers=None,
                 norm=None,
                 norm_kwargs=None,
                 norm_per_task=False,
                 squeeze=False,
                 adapters=False,
                 se_after_relu=True,
                 reduction=16):

        super(ConvCoupledSE, self).__init__()

        self.norm_per_task = norm_per_task
        self.squeeze = squeeze
        self.adapters = adapters
        self.se_after_relu = se_after_relu

        if not isinstance(process_layers, list):
            process_layers = [process_layers]

        self.process = nn.Sequential(*process_layers)

        se_module = SELayerMultiTaskDict

        if self.squeeze:
            self.se = se_module(process_layers[-1].out_channels, tasks=tasks, reduction=reduction)

        if self.adapters:
            print('Using parallel adapters')
            self.adapt = nn.ModuleDict({task: nn.Conv2d(process_layers[-1].in_channels, process_layers[-1].out_channels,
                                                        kernel_size=1, bias=False) for task in tasks})

        if self.norm_per_task:
            self.norm = nn.ModuleDict({task: norm(**norm_kwargs) for task in tasks})
        else:
            self.norm = norm(**norm_kwargs)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, task):
        if self.adapters:
            x = self.process(x) + self.adapt[task](x)
        else:
            x = self.process(x)

        if self.squeeze and not self.se_after_relu:
            x = self.se(x, task)

        if self.norm_per_task:
            x = self.norm[task](x)
        else:
            x = self.norm(x)

        x = self.relu(x)

        if self.squeeze and self.se_after_relu:
            x = self.se(x, task)

        return x
