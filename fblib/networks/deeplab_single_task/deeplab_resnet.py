# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F

import torchvision.models.resnet as resnet
from fblib.networks.deeplab_single_task.classifiers import PSPModule, AtrousPyramidModule, AtrousSpatialPyramidPoolingModule
from fblib.layers.misc_layers import center_crop, interp_surgery
from fblib.networks.deeplab_single_task import resnet as custom_resnet
from fblib.util.mypath import Path


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
        assert(dim % n_lay_per_group_low == 0)
        return int(dim / n_lay_per_group_low)
    else:
        assert(dim % n_lay_per_group == 0)
        return int(dim / n_lay_per_group)


def outS(i):
    i = int(i)
    i = (i+1)/2
    i = int(np.ceil((i+1)/2.0))
    i = (i+1)/2
    return i


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     dilation=dilation, padding=dilation, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, train_norm_layers=False,
                 sync_bnorm=False):
        super(BasicBlock, self).__init__()
        self.bnorm = nn.BatchNorm2d

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = self.bnorm(planes)
        for i in self.bn1.parameters():
            i.requires_grad = train_norm_layers
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn2 = self.bnorm(planes)
        for i in self.bn2.parameters():
            i.requires_grad = train_norm_layers
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1,  dilation=1, downsample=None, train_norm_layers=False,
                 sync_bnorm=False):
        super(Bottleneck, self).__init__()

        padding = dilation

        self.bnorm = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)  # change
        self.bn1 = self.bnorm(planes, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = train_norm_layers

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,  # change
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = self.bnorm(planes, affine=affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = train_norm_layers

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = self.bnorm(planes * 4, affine=affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = train_norm_layers

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, n_classes, nInputChannels=3, classifier="atrous",
                 output_stride=16, decoder=True,
                 static_graph=False, deconv_upsample=False, groupnorm=False, train_norm_layers=False,
                 sync_bnorm=False):

        super(ResNet, self).__init__()

        print("Constructing ResNet model...")
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

        self.inplanes = 64
        self.classifier = classifier
        self.decoder = decoder
        self.deconv_upsample = deconv_upsample
        self.groupnorm = groupnorm
        self.train_norm_layers = train_norm_layers
        self.sync_bnorm = sync_bnorm
        self.bnorm = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(nInputChannels, 64, kernel_size=7, stride=strides[0], padding=3,
                               bias=False)
        self.bn1 = self.bnorm(64, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = self.train_norm_layers
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=strides[1], padding=1, ceil_mode=False)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[2])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[3], dilation=dilations[0])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[4], dilation=dilations[1])

        if block == BasicBlock:
            in_f, out_f = 512, 128
        else:
            in_f, out_f = 2048, 512
        if decoder:
            print('Using decoder')
            if classifier == "atrous":
                print('Initializing classifier: old atrous pyramid')
                out_f_classifier = 256
                self.layer5 = AtrousPyramidModule(dilation_series=[6, 12, 18, 24], padding_series=[6, 12, 18, 24],
                                                    n_classes=out_f_classifier, in_f=in_f)
            elif classifier == "psp":
                print('Initializing classifier: PSP')
                out_f_classifier = 256
                self.layer5 = PSPModule(in_features=in_f, out_features=out_f_classifier, sizes=(1, 2, 3, 6),
                                        n_classes=0, groupnorm=self.groupnorm, sync_bnorm=self.sync_bnorm)
            elif classifier == 'atrous-v3':
                print('Initializing classifier: A-trous with global features (Deeplab-v3+)')
                out_f_classifier = 256
                self.layer5 = AtrousSpatialPyramidPoolingModule(depth=out_f_classifier,
                                                                groupnorm=self.groupnorm,
                                                                dilation_series=v3_atrous_rates,
                                                                sync_bnorm=self.sync_bnorm)
            else:
                out_f_classifier = out_f
                self.layer5 = None

            if not self.groupnorm:
                NormModule = self.bnorm
                kwargs_low = {"num_features": 48, "affine": affine_par}
                kwargs_out = {"num_features": 256, "affine": affine_par}
            else:
                NormModule = nn.GroupNorm
                kwargs_low = {"num_groups": get_ngroups_gn(48), "num_channels": 48, "affine": affine_par}
                kwargs_out = {"num_groups": get_ngroups_gn(256), "num_channels": 256, "affine": affine_par}

            self.low_level_reduce = nn.Sequential(
                nn.Conv2d(256, 48, kernel_size=1, bias=False),
                NormModule(**kwargs_low),
                nn.ReLU(inplace=True)
            )
            self.concat_and_predict = nn.Sequential(
                conv3x3(out_f_classifier + 48, 256),
                NormModule(**kwargs_out),
                nn.ReLU(inplace=True),
                conv3x3(256, 256),
                NormModule(**kwargs_out),
                nn.ReLU(inplace=True),
                # final layer
                nn.Conv2d(256, n_classes, kernel_size=1, bias=True)
            )

            if self.deconv_upsample:
                print("Using upsampling with deconvolutions")
                up_factor = 2
                self.upscale_1 = nn.ConvTranspose2d(out_f_classifier, out_f_classifier,
                                                    kernel_size=up_factor * 2, stride=up_factor, bias=False)

        else:
            if classifier == "atrous":
                print('Initializing classifier: A-trous pyramid')

                self.layer5 = AtrousPyramidModule(dilation_series=[6, 12, 18, 24], padding_series=[6, 12, 18, 24],
                                                  n_classes=n_classes, in_f=in_f)
            elif classifier == "psp":
                print('Initializing classifier: PSP')
                self.layer5 = PSPModule(in_features=in_f, out_features=out_f, sizes=(1, 2, 3, 6), n_classes=n_classes,
                                        static_graph=static_graph, groupnorm=self.groupnorm, sync_bnorm=self.sync_bnorm)
            elif classifier == "atrous-v3":
                print('Initializing classifier: A-trous with global features (Deeplab-v3+)')
                self.layer5 = AtrousSpatialPyramidPoolingModule(depth=n_classes, in_f=in_f,
                                                                groupnorm=self.groupnorm,
                                                                dilation_series=v3_atrous_rates,
                                                                sync_bnorm=self.sync_bnorm)
            else:
                self.layer5 = None

        # Initialize weights
        self._initialize_weights()

        # Check if batchnorm parameters are trainable
        self._verify_bnorm_params()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                self.bnorm(planes * block.expansion, affine=affine_par),
            )

            # Train batchnorm?
            for i in downsample._modules['1'].parameters():
                i.requires_grad = self.train_norm_layers

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample,
                            train_norm_layers=self.train_norm_layers, sync_bnorm=self.sync_bnorm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation,
                                train_norm_layers=self.train_norm_layers, sync_bnorm=self.sync_bnorm))

        return nn.Sequential(*layers)

    def _verify_bnorm_params(self):
        verify_trainable = True
        a = 0
        for x in self.modules():
            if isinstance(x, nn.BatchNorm2d):
                for y in x.parameters():
                    verify_trainable = (verify_trainable and y.requires_grad)
                a += isinstance(x, nn.BatchNorm2d)

        print("\nVerification: Trainable batchnorm parameters? Answer: {}\n".format(verify_trainable))
        print("Asynchronous bnorm layers: {}".format(a))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, self.bnorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.zero_()
                m.weight.data = interp_surgery(m)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def forward(self, x, bbox=None):
        h, w = x.shape[2:]
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
        if self.layer5 is not None:
            x = self.layer5(x)
        if self.decoder:
            if self.deconv_upsample:
                x = center_crop(self.upscale_1(x), low_h, low_w)
            else:
                x = F.interpolate(x, size=(x_low.shape[2], x_low.shape[3]),
                                  mode='bilinear', align_corners=False)
            x_low = self.low_level_reduce(x_low)
            x = torch.cat([x, x_low], dim=1)
            x = self.concat_and_predict(x)

        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

        return x

    def load_pretrained(self, base_network, nInputChannels=3):
        flag = 0
        i = 0
        for module, module_ori in izip(self.modules(), base_network.modules()):
            if isinstance(module, nn.Conv2d) and isinstance(module_ori, nn.Conv2d):
                if not flag and nInputChannels != 3:
                    module.weight[:, :3, :, :].data = deepcopy(module_ori.weight.data)
                    module.bias = deepcopy(module_ori.bias)
                    for i in range(3, int(module.weight.data.shape[1])):
                        module.weight[:, i, :, :].data = deepcopy(module_ori.weight[:, -1, :, :][:, np.newaxis, :, :].data)
                    flag = 1
                    i += 1
                elif module.weight.data.shape == module_ori.weight.data.shape:
                    i += 1
                    module.weight.data = deepcopy(module_ori.weight.data)
                    module.bias = deepcopy(module_ori.bias)
                else:
                    print('Skipping Conv layer with size: {} and target size: {}'
                          .format(module.weight.data.shape, module_ori.weight.data.shape))
            elif isinstance(module, self.bnorm) and \
                    (isinstance(module_ori, nn.BatchNorm2d) or isinstance(module_ori, self.bnorm)):
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

    def load_pretrained_ms(self, base_network, nInputChannels=3):
        flag = 0
        for module, module_ori in izip(self.modules(), base_network.Scale.modules()):
            if isinstance(module, nn.Conv2d) and isinstance(module_ori, nn.Conv2d):
                if not flag and nInputChannels != 3:
                    module.weight[:, :3, :, :].data = deepcopy(module_ori.weight.data)
                    module.bias = deepcopy(module_ori.bias)
                    for i in range(3, int(module.weight.data.shape[1])):
                        module.weight[:, i, :, :].data = deepcopy(module_ori.weight[:, -1, :, :][:, np.newaxis, :, :].data)
                    flag = 1
                elif module.weight.data.shape == module_ori.weight.data.shape:
                    module.weight.data = deepcopy(module_ori.weight.data)
                    module.bias = deepcopy(module_ori.bias)
                else:
                    print('Skipping Conv layer with size: {} and target size: {}'
                          .format(module.weight.data.shape, module_ori.weight.data.shape))
            elif isinstance(module, self.bnorm) and (isinstance(module_ori, nn.BatchNorm2d) or isinstance(module_ori, self.bnorm)) \
                    and module.weight.data.shape == module_ori.weight.data.shape:

                # Copy running mean and variance of batchnorm layers!
                module.running_mean.data = deepcopy(module_ori.running_mean.data)
                module.running_var.data = deepcopy(module_ori.running_var.data)

                module.weight.data = deepcopy(module_ori.weight.data)
                module.bias.data = deepcopy(module_ori.bias.data)


class MS_Deeplab(nn.Module):
    def __init__(self, block, NoLabels, nInputChannels=3):
        super(MS_Deeplab, self).__init__()
        self.Scale = ResNet(block, [3, 4, 23, 3], NoLabels, nInputChannels=nInputChannels)

    def forward(self, x):
        input_size = x.size()[2]
        self.interp1 = nn.Upsample(size=(int(input_size*0.75)+1, int(input_size*0.75)+1), mode='bilinear', align_corners=False)
        self.interp2 = nn.Upsample(size=(int(input_size*0.5)+1, int(input_size*0.5)+1), mode='bilinear', align_corners=False)
        self.interp3 = nn.Upsample(size=(outS(input_size), outS(input_size)), mode='bilinear', align_corners=False)
        out = []
        x2 = self.interp1(x)
        x3 = self.interp2(x)
        out.append(self.Scale(x))	# for original scale
        out.append(self.interp3(self.Scale(x2)))	# for 0.75x scale
        out.append(self.Scale(x3))	# for 0.5x scale

        x2Out_interp = out[1]
        x3Out_interp = self.interp3(out[2])
        temp1 = torch.max(out[0], x2Out_interp)
        out.append(torch.max(temp1, x3Out_interp))
        return out[-1]


def Res_Deeplab(n_classes=21, pretrained=None):
    model = MS_Deeplab(Bottleneck, n_classes)
    if pretrained is not None:
        if pretrained == 'voc':
            pth_model = 'MS_DeepLab_resnet_trained_VOC.pth'
        elif pretrained == 'ms_coco':
            pth_model = 'MS_DeepLab_resnet_pretrained_COCO_init.pth'
        saved_state_dict = torch.load(os.path.join(Path.models_dir(), pth_model),
                                      map_location=lambda storage, loc: storage)
        if n_classes != 21:
            for i in saved_state_dict:
                i_parts = i.split('.')
                if i_parts[1] == 'layer5':
                    saved_state_dict[i] = model.state_dict()[i]
        model.load_state_dict(saved_state_dict)
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
    return base_lr*((1-float(iter_)/max_iter)**power)


def resnet18(n_classes, pretrained=None, nInputChannels=3, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], n_classes, nInputChannels=nInputChannels, **kwargs)
    model_full = resnet.resnet18(pretrained=True)
    if pretrained:
        model.load_pretrained(model_full,  nInputChannels=nInputChannels)
    return model


def resnet26(n_classes, pretrained='scratch', nInputChannels=3, **kwargs):
    """Constructs a ResNet-26 model.
    Args:
        pretrained ('imagenet', 'scratch): If 'imagenet', returns a model pre-trained on ImageNet
    """

    print('Constructing ResNet26')
    model = ResNet(Bottleneck, [2, 2, 2, 2], n_classes, nInputChannels=nInputChannels, **kwargs)

    if pretrained == 'imagenet':
        print('Loading resnet 26 weights')
        model_full = custom_resnet.resnet26(pretrained=True)
        model.load_pretrained(model_full, nInputChannels=nInputChannels)
    elif pretrained == 'scratch':
        print('Training from scratch')
    else:
        raise NotImplementedError('Please specify scratch or imagenet for pre-training')

    return model


def resnet50(n_classes, pretrained=None, nInputChannels=3, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], n_classes, nInputChannels=nInputChannels, **kwargs)

    if pretrained is not None:
        model_full = resnet.resnet50(pretrained=True)
        model.load_pretrained(model_full, nInputChannels=nInputChannels)
    return model


def resnet101(n_classes, pretrained='scratch', nInputChannels=3, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained ('imagenet', 'voc', 'ms_coco): Select model trained on respective dataset.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], n_classes, nInputChannels=nInputChannels, **kwargs)

    if pretrained == 'imagenet':
        print('Initializing from pre-trained ImageNet model..')
        model_full = resnet.resnet101(pretrained=True)
        model.load_pretrained(model_full, nInputChannels=nInputChannels)
    elif pretrained == 'ms_coco' or pretrained == 'voc':
        model_full = Res_Deeplab(n_classes, pretrained=pretrained)
        model.load_pretrained_ms(model_full, nInputChannels=nInputChannels)
    else:
        print('Training from scratch')
    return model


def test_flops():
    from fblib.util.model_resources.flops import compute_gflops
    net = resnet26(n_classes=21, pretrained='imagenet', classifier="atrous-v3",
                    output_stride=16, decoder=True)
    print('GFLOPS: {}'.format(compute_gflops(net, (2, 3, 256, 256))))


if __name__ == '__main__':
    test_flops()
