import math
from copy import deepcopy
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from fblib.networks.classifiers import PSPModule, AtrousPyramidModule, AtrousSpatialPyramidPoolingModule
from fblib.layers.misc_layers import center_crop, interp_surgery
import fblib.networks.resnext_imagenet as resnext_imagenet
from encoding.nn import BatchNorm2d as SyncBatchNorm2d

try:
    from itertools import izip
except ImportError:  # python3.x
    izip = zip

affine_par = True  # Trainable Batchnorm for the classifier


def get_ngroups_gn(dim):
    """
    Get number of groups used by groupnorm, based on number of channels
    """
    n_lay_per_group_low = 16
    n_lay_per_group = 32
    if dim <= 256:
        assert (dim % n_lay_per_group_low == 0)
        return int(dim / n_lay_per_group_low)
    else:
        assert (dim % n_lay_per_group == 0)
        return int(dim / n_lay_per_group)


def outS(i):
    i = int(i)
    i = (i + 1) / 2
    i = int(np.ceil((i + 1) / 2.0))
    i = (i + 1) / 2
    return i


def conv3x3(in_planes, out_planes, stride=1, dilation=1, cardinality=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     dilation=dilation, padding=dilation, bias=False, groups=cardinality)


class ResNeXtBottleneck(nn.Module):
    expansion = 4
    """
  RexNeXt bottleneck type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
  """

    def __init__(self, inplanes, planes, base_width=4, stride=1, downsample=None,
                 dilation_=1, train_norm_layers=False, sync_bnorm=False, cardinality=32):
        super(ResNeXtBottleneck, self).__init__()

        D = int(math.floor(planes * (base_width / 64.0)))
        C = cardinality
        padding = dilation_

        self.bnorm = nn.BatchNorm2d if not sync_bnorm else SyncBatchNorm2d
        self.conv_reduce = nn.Conv2d(inplanes, D * C, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_reduce = self.bnorm(D * C, affine=affine_par)
        for i in self.bn_reduce.parameters():
            i.requires_grad = train_norm_layers

        self.conv_conv = nn.Conv2d(D * C, D * C, kernel_size=3, stride=stride, padding=padding, groups=cardinality,
                                   bias=False, dilation=dilation_)
        self.bn = self.bnorm(D * C, affine=affine_par)
        for i in self.bn.parameters():
            i.requires_grad = train_norm_layers

        self.conv_expand = nn.Conv2d(D * C, planes * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_expand = self.bnorm(planes * 4, affine=affine_par)
        for i in self.bn_expand.parameters():
            i.requires_grad = train_norm_layers

        self.downsample = downsample

    def forward(self, x):
        residual = x

        bottleneck = self.conv_reduce(x)
        bottleneck = F.relu(self.bn_reduce(bottleneck), inplace=True)

        bottleneck = self.conv_conv(bottleneck)
        bottleneck = F.relu(self.bn(bottleneck), inplace=True)

        bottleneck = self.conv_expand(bottleneck)
        bottleneck = self.bn_expand(bottleneck)

        if self.downsample is not None:
            residual = self.downsample(x)

        return F.relu(residual + bottleneck, inplace=True)


class ResNeXt(nn.Module):
    def __init__(self, block, layers, n_classes, nInputChannels=3, classifier="atrous",
                 output_stride=8, decoder=False,
                 static_graph=False, deconv_upsample=False, groupnorm=False, tasks=None, train_norm_layers=False,
                 sync_bnorm=False, cardinality=32):

        super(ResNeXt, self).__init__()

        print("Constructing ResNeXt model...")
        print("Output stride: {}".format(output_stride))
        print("Number of classes: {}".format(n_classes))
        print("Number of Input Channels: {}".format(nInputChannels))

        v3_atrous_rates = [6, 12, 18]

        if output_stride == 8:
            dilations = (2, 4)
            strides = (2, 2, 2, 1, 1)
            v3_atrous_rates = [x * 2 for x in v3_atrous_rates]
        elif output_stride == 16:
            dilations = (1, 2)
            strides = (2, 2, 2, 2, 1)
        else:
            raise ValueError('Choose between output_stride 8 and 16')

        self.cardinality = cardinality
        self.inplanes = 64
        self.classifier = classifier
        self.decoder = decoder
        self.deconv_upsample = deconv_upsample
        self.groupnorm = groupnorm
        self.train_norm_layers = train_norm_layers
        self.sync_bnorm = sync_bnorm
        self.bnorm = nn.BatchNorm2d if not self.sync_bnorm else SyncBatchNorm2d
        self.tasks = tasks
        self.task_dict = {x: i for i, x in enumerate(self.tasks)}

        # Network structure
        self.conv1 = nn.Conv2d(nInputChannels, 64, kernel_size=7, stride=strides[0], padding=3,
                               bias=False)
        self.bn1 = self.bnorm(64, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = self.train_norm_layers
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=strides[1], padding=1, ceil_mode=False)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[2])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[3], dilation__=dilations[0])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[4], dilation__=dilations[1])

        in_f, out_f = 2048, 512

        if not self.groupnorm:
            NormModule = self.bnorm
            kwargs_low = {"num_features": 48, "affine": affine_par}
            kwargs_out = {"num_features": 256, "affine": affine_par}
        else:
            NormModule = nn.GroupNorm
            kwargs_low = {"num_groups": get_ngroups_gn(48), "num_channels": 48, "affine": affine_par}
            kwargs_out = {"num_groups": get_ngroups_gn(256), "num_channels": 256, "affine": affine_par}

        self.classifiers = nn.ModuleDict()

        for task in self.tasks:
            self.classifiers[task] = nn.Module()
            print('\nCreating specialized layers for task: {}'.format(task))

            if decoder:
                print('Using decoder')
                if classifier == "atrous":
                    print('Initializing classifier: old atrous pyramid')
                    out_f_classifier = 256
                    self.classifiers[task].layer5 = AtrousPyramidModule(dilation_series=[6, 12, 18, 24],
                                                                        padding_series=[6, 12, 18, 24],
                                                                        n_classes=out_f_classifier,
                                                                        in_f=in_f)
                elif classifier == "psp":
                    print('Initializing classifier: PSP')
                    out_f_classifier = 256
                    self.classifiers[task].layer5 = PSPModule(in_features=in_f,
                                                              out_features=out_f_classifier,
                                                              sizes=(1, 2, 3, 6),
                                                              n_classes=0,
                                                              groupnorm=self.groupnorm,
                                                              sync_bnorm=self.sync_bnorm)
                elif classifier == 'atrous-v3':
                    print('Initializing classifier: A-trous with global features (Deeplab-v3+)')
                    out_f_classifier = 256
                    self.classifiers[task].layer5 = AtrousSpatialPyramidPoolingModule(depth=out_f_classifier,
                                                                                      groupnorm=self.groupnorm,
                                                                                      dilation_series=v3_atrous_rates,
                                                                                      sync_bnorm=self.sync_bnorm,
                                                                                      cardinality=self.cardinality)
                else:
                    raise NotImplementedError

                self.classifiers[task].low_level_reduce = nn.Sequential(
                    nn.Conv2d(256, 48, kernel_size=1, bias=False),
                    NormModule(**kwargs_low),
                    nn.ReLU(inplace=True)
                )
                self.classifiers[task].concat_and_predict = nn.Sequential(
                    conv3x3(out_f_classifier + 48, 256),
                    NormModule(**kwargs_out),
                    nn.ReLU(inplace=True),
                    conv3x3(256, 256, cardinality=self.cardinality),
                    NormModule(**kwargs_out),
                    nn.ReLU(inplace=True),
                    # final layer
                    nn.Conv2d(256, n_classes[task], kernel_size=1, bias=True)
                )

                if self.deconv_upsample:
                    print("Using upsampling with deconvolutions")
                    up_factor = 2
                    self.classifiers[task].upscale_1 = nn.ConvTranspose2d(out_f_classifier, out_f_classifier,
                                                                          kernel_size=up_factor * 2, stride=up_factor,
                                                                          bias=False)

            else:
                if classifier == "atrous":
                    print('Initializing classifier: A-trous pyramid')

                    self.classifiers[task].layer5 = AtrousPyramidModule(dilation_series=[6, 12, 18, 24],
                                                                        padding_series=[6, 12, 18, 24],
                                                                        n_classes=n_classes[task],
                                                                        in_f=in_f)
                elif classifier == "psp":
                    print('Initializing classifier: PSP')
                    self.classifiers[task].layer5 = PSPModule(in_features=in_f,
                                                              out_features=out_f,
                                                              sizes=(1, 2, 3, 6),
                                                              n_classes=n_classes[task],
                                                              static_graph=static_graph,
                                                              groupnorm=self.groupnorm,
                                                              sync_bnorm=self.sync_bnorm)
                elif classifier == "atrous-v3":
                    print('Initializing classifier: A-trous with global features (Deeplab-v3+)')
                    self.classifiers[task].layer5 = AtrousSpatialPyramidPoolingModule(depth=n_classes,
                                                                                      in_f=in_f,
                                                                                      groupnorm=self.groupnorm,
                                                                                      dilation_series=v3_atrous_rates,
                                                                                      sync_bnorm=self.sync_bnorm)
                else:
                    raise NotImplementedError

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, self.bnorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.zero_()
                m.weight.data = interp_surgery(m)

        # Check if batchnorm parameters are trainable
        self._verify_bnorm_params()

    def _make_layer(self, block, planes, blocks, stride=1, dilation__=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation__ == 2 or dilation__ == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                self.bnorm(planes * block.expansion, affine=affine_par),
            )

            # Train batchnorm?
            for i in downsample._modules['1'].parameters():
                i.requires_grad = self.train_norm_layers

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation_=dilation__, downsample=downsample,
                            train_norm_layers=self.train_norm_layers, sync_bnorm=self.sync_bnorm,
                            cardinality=self.cardinality))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation_=dilation__, train_norm_layers=self.train_norm_layers,
                                sync_bnorm=self.sync_bnorm, cardinality=self.cardinality))

        return nn.Sequential(*layers)

    def _verify_bnorm_params(self):
        verify_trainable = True
        a = s = 0
        for x in self.modules():
            if isinstance(x, nn.BatchNorm2d) or isinstance(x, SyncBatchNorm2d):
                for y in x.parameters():
                    verify_trainable = (verify_trainable and y.requires_grad)
                a += isinstance(x, nn.BatchNorm2d)
                s += isinstance(x, SyncBatchNorm2d)

        print("\nVerification: Trainable batchnorm parameters? Answer: {}\n".format(verify_trainable))
        print("Asynchronous bnorm layers: {}".format(a))
        print("Synchronous bnorm layers: {}".format(s))

    def forward(self, x, task_gts=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        if self.decoder:
            x_low = x
            low_h, low_w = int(x_low.size()[-2]), int(x_low.size()[-1])
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        out = {}
        for task in self.tasks:
            if self._chk_forward(task, task_gts):
                x_task = self.classifiers[task].layer5(x)
                if self.decoder:
                    if self.deconv_upsample:
                        x_task = center_crop(self.classifiers[task].upscale_1(x_task), low_h, low_w)
                    else:
                        x_task = F.interpolate(x_task,
                                               size=(x_low.shape[2], x_low.shape[3]),
                                               mode='bilinear',
                                               align_corners=False)
                    x_low_task = self.classifiers[task].low_level_reduce(x_low)
                    x_task = torch.cat([x_task, x_low_task], dim=1)
                    x_task = self.classifiers[task].concat_and_predict(x_task)
                    out[task] = x_task
        return out

    @staticmethod
    def _chk_forward(task, meta):
        if meta is None:
            return True
        else:
            return task in meta

    def load_pretrained(self, base_network, nInputChannels=3):
        flag = 0
        i = 0
        for module, module_ori in izip(self.modules(), base_network.modules()):
            if isinstance(module, nn.Conv2d) and isinstance(module_ori, nn.Conv2d):
                if not flag and nInputChannels != 3:
                    module.weight[:, :3, :, :].data = deepcopy(module_ori.weight.data)
                    module.bias = deepcopy(module_ori.bias)
                    for i in range(3, int(module.weight.data.shape[1])):
                        module.weight[:, i, :, :].data = deepcopy(
                            module_ori.weight[:, -1, :, :][:, np.newaxis, :, :].data)
                    flag = 1
                elif module.weight.data.shape == module_ori.weight.data.shape:
                    i += 1
                    module.weight.data = deepcopy(module_ori.weight.data)
                    module.bias = deepcopy(module_ori.bias)
                else:
                    print('Skipping Conv layer with size: {} and target size: {}'
                          .format(module.weight.data.shape, module_ori.weight.data.shape))
            elif isinstance(module, self.bnorm) and (isinstance(module_ori, nn.BatchNorm2d)
                                                     or isinstance(module_ori, self.bnorm)):
                if module.weight.data.shape == module_ori.weight.data.shape:
                    i += 1

                    # Copy running mean and variance of batchnorm layers!
                    module.running_mean.data = deepcopy(module_ori.running_mean.data)
                    module.running_var.data = deepcopy(module_ori.running_var.data)

                    module.weight.data = deepcopy(module_ori.weight.data)
                    module.bias.data = deepcopy(module_ori.bias.data)
                else:
                    print('Skipping Batchnorm layer with size: {} and target size: {}'
                          .format(module.weight.data.shape, module_ori.weight.data.shape))
        print("Content of {} layers successfully copied.".format(i))


def resnext50(n_classes, pretrained=False, nInputChannels=3, **kwargs):
    """Constructs a ResNeXt-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    print('Constructing ResNeXt50')

    model = ResNeXt(ResNeXtBottleneck, layers=[3, 4, 6, 3], n_classes=n_classes,
                    nInputChannels=nInputChannels, **kwargs)
    if pretrained:
        model_full = resnext_imagenet.resnext50_32x4d(pretrained=True)
        model.load_pretrained(model_full, nInputChannels=nInputChannels)
    else:
        print('Training from scratch')
    return model


def resnext101(n_classes, pretrained=False, nInputChannels=3, **kwargs):
    """Constructs a ResNeXt-101 model.
    Args:
        pretrained (bool): Select model trained on respective dataset.
    """

    print('Constructing ResNeXt101')

    model = ResNeXt(ResNeXtBottleneck, layers=[3, 4, 23, 3], n_classes=n_classes,
                    nInputChannels=nInputChannels, **kwargs)
    if pretrained:
        model_full = resnext_imagenet.resnext101_32x4d(pretrained=True)
        model.load_pretrained(model_full, nInputChannels=nInputChannels)
    else:
        print('Training from scratch')
    return model


def get_lr_params(model):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batchnorm layer,
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
    any batchnorm parameter
    """

    # for multi-GPU training
    if hasattr(model, 'module'):
        model = model.module

    b = [model.conv1, model.bn1, model.layer1, model.layer2, model.layer3, model.layer4, model.layer5]

    # If decoder exists
    if 'low_level_reduce' in model._modules.keys():
        b.extend([model.low_level_reduce, model.concat_and_predict])

    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k


def get_1x_lr_params(model):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batchnorm layer,
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
    any batchnorm parameter
    """

    # for multi-GPU training
    if hasattr(model, 'module'):
        model = model.module

    b = [model.conv1, model.bn1, model.layer1, model.layer2, model.layer3, model.layer4]

    # If decoder exists
    if 'low_level_reduce' in model._modules.keys():
        b.extend([model.low_level_reduce])

    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """

    # for multi-GPU training
    if hasattr(model, 'module'):
        model = model.module

    b = [model.layer5]

    # If decoder exists
    if 'low_level_reduce' in model._modules.keys():
        b.extend([model.concat_and_predict])

    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k


def lr_poly(base_lr, iter_, max_iter=100, power=0.9):
    return base_lr * ((1 - float(iter_) / max_iter) ** power)


if __name__ == '__main__':
    net = resnext50(n_classes=21, pretrained=True, nInputChannels=3, classifier="atrous-v3",
                    output_stride=8, decoder=True, static_graph=False, deconv_upsample=False, groupnorm=False,
                    train_norm_layers=False, sync_bnorm=False, cardinality=32)
