import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F

import fblib.networks.se_resnet_imagenet as se_resnet_imagenet
from fblib.networks.classifiers import AtrousSpatialPyramidPoolingModule
from fblib.layers.misc_layers import interp_surgery

from encoding.nn import BatchNorm2d as SyncBatchNorm2d

try:
    from itertools import izip
except ImportError:  # python3.x
    izip = zip

affine_par = True  # Trainable Batchnorm for the classifier


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


class SELayer(nn.Module):
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


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, train_norm_layers=False,
                 sync_bnorm=False, reduction=16):
        super(SEBottleneck, self).__init__()

        padding = dilation

        self.bnorm = nn.BatchNorm2d if not sync_bnorm else SyncBatchNorm2d

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = self.bnorm(planes)
        for i in self.bn1.parameters():
            i.requires_grad = train_norm_layers

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = self.bnorm(planes)
        for i in self.bn2.parameters():
            i.requires_grad = train_norm_layers

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = self.bnorm(planes * 4)
        for i in self.bn3.parameters():
            i.requires_grad = train_norm_layers

        self.relu = nn.ReLU(inplace=True)

        self.se = SELayer(planes * 4, reduction)
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
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEResNet(nn.Module):
    def __init__(self, block, layers, n_classes, nInputChannels=3, classifier="atrous",
                 output_stride=16, decoder=True, train_norm_layers=False, sync_bnorm=False):

        super(SEResNet, self).__init__()

        print("Constructing Squeeeze and Excitation ResNet model...")
        print("Output stride: {}".format(output_stride))
        print("Number of classes: {}".format(n_classes))

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
        self.train_norm_layers = train_norm_layers
        self.sync_bnorm = sync_bnorm
        self.bnorm = nn.BatchNorm2d if not self.sync_bnorm else SyncBatchNorm2d

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

        in_f, out_f = 2048, 512

        if decoder:
            print('Using decoder')
            if classifier == 'atrous-v3':
                print('Initializing classifier: A-trous with global features (Deeplab-v3+)')
                out_f_classifier = 256
                self.layer5 = AtrousSpatialPyramidPoolingModule(depth=out_f_classifier,
                                                                dilation_series=v3_atrous_rates,
                                                                sync_bnorm=self.sync_bnorm)
            else:
                raise NotImplementedError('Select one of the available decoders')

            NormModule = self.bnorm
            kwargs_low = {"num_features": 48, "affine": affine_par}
            kwargs_out = {"num_features": 256, "affine": affine_par}

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

        else:
            if classifier == "atrous-v3":
                print('Initializing classifier: A-trous with global features (Deeplab-v3+)')
                self.layer5 = AtrousSpatialPyramidPoolingModule(depth=n_classes, in_f=in_f,
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

        layers = [block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample,
                        train_norm_layers=self.train_norm_layers, sync_bnorm=self.sync_bnorm)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation,
                                train_norm_layers=self.train_norm_layers, sync_bnorm=self.sync_bnorm))

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
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        if self.decoder:
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
            elif isinstance(module, nn.Linear) and isinstance(module_ori, nn.Linear):
                module.weight.data = deepcopy(module_ori.weight.data)
                module.bias.data = deepcopy(module_ori.bias.data)
                i += 1
        print("Content of {} layers successfully copied.".format(i))


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


def se_resnet26(n_classes, pretrained=None, nInputChannels=3, **kwargs):
    """Constructs a ResNet-26 model.
    Args:
        pretrained ('imagenet', 'scratch'): If True, returns a model pre-trained on ImageNet
    """
    model = SEResNet(SEBottleneck, [2, 2, 2, 2], n_classes, nInputChannels=nInputChannels, **kwargs)
    if pretrained == 'imagenet':
        model_full = se_resnet_imagenet.se_resnet26(pretrained=True)
        model.load_pretrained(model_full,  nInputChannels=nInputChannels)
    elif pretrained == 'scratch':
        print('Training from scratch')
    else:
        raise NotImplementedError('Select imagenet or scratch for pre-training')

    return model


def se_resnet50(n_classes, pretrained=None, nInputChannels=3, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained ('imagenet', 'scratch'): If True, returns a model pre-trained on ImageNet
    """
    model = SEResNet(SEBottleneck, [3, 4, 6, 3], n_classes, nInputChannels=nInputChannels, **kwargs)

    if pretrained == 'imagenet':
        model_full = se_resnet_imagenet.se_resnet50(pretrained=True)
        model.load_pretrained(model_full, nInputChannels=nInputChannels)
    elif pretrained == 'scratch':
        print('Training from scratch')
    else:
        raise NotImplementedError('Select imagenet or scratch for pre-training')

    return model


def se_resnet101(n_classes, pretrained='scratch', nInputChannels=3, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained ('imagenet', 'scratch'): Select model trained on respective dataset.
    """
    model = SEResNet(SEBottleneck,  [3, 4, 23, 3], n_classes, nInputChannels=nInputChannels, **kwargs)

    if pretrained == 'imagenet':
        model_full = se_resnet_imagenet.se_resnet101(pretrained=True)
        model.load_pretrained(model_full, nInputChannels=nInputChannels)
    elif pretrained == 'scratch':
        print('Training from scratch')
    else:
        raise NotImplementedError('Select imagenet or scratch for pre-training')

    return model


def test_flops():
    from fblib.util.model_resources.flops import compute_gflops
    net = se_resnet50(n_classes=21, pretrained='imagenet', classifier="atrous-v3",
                      output_stride=16, decoder=True)
    print('GFLOPS: {}'.format(compute_gflops(net, (2, 3, 256, 256))))


if __name__ == '__main__':
    test_flops()
