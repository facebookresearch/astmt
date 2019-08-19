# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn
import torch.nn.functional as F

from fblib.networks.deeplab_multi_task.classifiers_multitask import AtrousSpatialPyramidPoolingModule
from fblib.layers.squeeze import ConvCoupledSE


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     dilation=dilation, padding=dilation, bias=False)


class UbernetDecoder(nn.Module):
    """
    Simple Shallow decoder (like Ubernet)
    """

    def __init__(self,
                 tasks,
                 in_channels_low,
                 in_channels_high,
                 n_classes,
                 norm=nn.BatchNorm2d,
                 ret_features=True):
        super(UbernetDecoder, self).__init__()

        self.tasks = tasks
        self.ret_features = ret_features

        self.high_level = nn.ModuleDict()
        self.low_level = nn.ModuleDict()
        self.predict = nn.ModuleDict()

        for task in tasks:
            self.high_level[task] = nn.Sequential(nn.Conv2d(in_channels=in_channels_high,
                                                            out_channels=n_classes[task],
                                                            kernel_size=1,
                                                            bias=False),
                                                  norm(n_classes[task]),
                                                  nn.ReLU(inplace=True))

            self.low_level[task] = nn.Sequential(nn.Conv2d(in_channels=in_channels_low,
                                                           out_channels=n_classes[task],
                                                           kernel_size=1,
                                                           bias=False),
                                                 norm(n_classes[task]),
                                                 nn.ReLU(inplace=True))

            self.predict[task] = nn.Conv2d(in_channels=2 * n_classes[task],
                                           out_channels=n_classes[task],
                                           kernel_size=1,
                                           bias=True)

    def forward(self, x_low, x_high, task=None):
        # Reduce dimensionality of low-level features
        x_low = self.low_level[task](x_low)

        # Reduce dimensionality of high-level features and upscale
        x_high = self.high_level[task](x_high)
        x_high = F.interpolate(x_high, size=(x_low.shape[2], x_low.shape[3]), mode='bilinear', align_corners=False)

        # Concatenate features
        x = torch.cat((x_low, x_high), dim=1)

        features = x

        # Make final prediction
        x = self.predict[task](x)

        if self.ret_features:
            return x, features
        else:
            return x


def test_ubernet():
    print('Testing UberNet-like decoder')
    tasks = ['edge', 'semseg', 'human_parts']
    out_channels = {'edge': 1, 'semseg': 21, 'human_parts': 7}

    in_channels_low = 256
    in_channels_high = 2048

    x_low = torch.rand(2, in_channels_low, 128, 128)
    x_high = torch.rand(2, in_channels_high, 64, 64)

    net = UbernetDecoder(tasks=tasks,
                         in_channels_low=in_channels_low,
                         in_channels_high=in_channels_high,
                         n_classes=out_channels)

    x_low, x_high, net = x_low.cuda(), x_high.cuda(), net.cuda()

    for task in tasks:
        out, _ = net(x_low, x_high, task=task)
        print('Task: {}, Output Shape: {}'.format(task, out.shape))


class ASPPv3Plus(nn.Module):
    """
    ASPP-v3 decoder
    """

    def __init__(self,
                 tasks,
                 n_classes,
                 classifier='atrous-v3',
                 in_channels_low=256,
                 in_channels_high=2048,
                 out_f_classifier=64,
                 atrous_rates=None,
                 norm=nn.BatchNorm2d,
                 norm_per_task=True,
                 squeeze=False,
                 adapters=False,
                 ):
        super(ASPPv3Plus, self).__init__()
        print('Initializing ASPP v3 Decoder for multiple tasks')

        if atrous_rates is None:
            atrous_rates = [6, 12, 18]

        out_f_low = 48 * out_f_classifier / 256  # Adapt in case of thinner classifiers
        assert (int(out_f_low) == out_f_low)
        out_f_low = int(out_f_low)

        kwargs_low = {"num_features": int(out_f_low), "affine": True}
        kwargs_out = {"num_features": out_f_classifier, "affine": True}

        self.tasks = tasks

        if classifier == 'atrous-v3':
            print('Initializing classifier: ASPP with global features (Deeplab-v3+)')
            self.layer5 = AtrousSpatialPyramidPoolingModule(in_f=in_channels_high,
                                                            depth=out_f_classifier,
                                                            dilation_series=atrous_rates,
                                                            tasks=self.tasks,
                                                            norm_per_task=norm_per_task,
                                                            squeeze=squeeze,
                                                            adapters=adapters)
        elif classifier == 'conv':
            self.layer5 = ConvCoupledSE(tasks=tasks,
                                        process_layers=nn.Conv2d(in_channels_high, out_f_classifier, kernel_size=1,
                                                                 bias=False),
                                        norm=norm,
                                        norm_kwargs=kwargs_low,
                                        norm_per_task=norm_per_task,
                                        squeeze=squeeze,
                                        adapters=adapters,
                                        reduction=4)
        else:
            raise NotImplementedError('Choose one of the available classifiers')

        self.low_level_reduce = ConvCoupledSE(tasks=tasks,
                                              process_layers=nn.Conv2d(in_channels_low, int(out_f_low), kernel_size=1,
                                                                       bias=False),
                                              norm=norm,
                                              norm_kwargs=kwargs_low,
                                              norm_per_task=norm_per_task,
                                              squeeze=squeeze,
                                              adapters=adapters,
                                              reduction=4)

        self.conv_concat = ConvCoupledSE(tasks=tasks,
                                         process_layers=conv3x3(out_f_classifier + int(out_f_low),
                                                                out_f_classifier),
                                         norm=norm,
                                         norm_kwargs=kwargs_out,
                                         norm_per_task=norm_per_task,
                                         squeeze=squeeze,
                                         adapters=adapters)

        self.conv_process = ConvCoupledSE(tasks=tasks,
                                          process_layers=conv3x3(out_f_classifier, out_f_classifier),
                                          norm=norm,
                                          norm_kwargs=kwargs_out,
                                          norm_per_task=norm_per_task,
                                          squeeze=squeeze,
                                          adapters=adapters)

        self.conv_predict = nn.ModuleDict(
            {task: nn.Conv2d(out_f_classifier, n_classes[task], kernel_size=1, bias=True) for task in tasks})

    def forward(self, x_low, x, task=None):
        x_low = self.low_level_reduce(x_low, task)

        x = self.layer5(x, task)

        x = F.interpolate(x, size=(x_low.shape[2], x_low.shape[3]), mode='bilinear', align_corners=False)

        x = torch.cat((x, x_low), dim=1)
        x = self.conv_concat(x, task)
        x = self.conv_process(x, task)

        features = x

        x = self.conv_predict[task](x)

        return x, features


def test_aspp():
    print('Testing ASPP-v3 decoder')
    import fblib.util.visualizepy as viz

    tasks = ['edge']
    n_classes = {'edge': 1}

    in_channels_low = 256
    in_channels_high = 2048
    out_f_classifier = 64

    x_low = torch.rand(2, in_channels_low, 128, 128).requires_grad_()
    x_high = torch.rand(2, in_channels_high, 64, 64).requires_grad_()

    net = ASPPv3Plus(tasks=tasks,
                     n_classes=n_classes,
                     classifier='atrous-v3',
                     in_channels_high=in_channels_high,
                     in_channels_low=in_channels_low,
                     out_f_classifier=out_f_classifier,
                     norm=nn.BatchNorm2d,
                     squeeze=True)

    x_low, x_high, net = x_low.cuda(), x_high.cuda(), net.cuda()

    out = {}
    for task in tasks:
        out[task], _ = net(x_low, x_high, task=task)
        print('Task: {}, Output Shape: {}'.format(task, out[task].shape))

    g = viz.make_dot(out, net.state_dict())
    g.view(directory='./')


def main():
    test_ubernet()
    test_aspp()


if __name__ == '__main__':
    main()
