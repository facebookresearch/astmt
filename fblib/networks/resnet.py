import os
import math
from copy import deepcopy
from collections import OrderedDict


import torch
import torch.nn as nn
from torch.nn.functional import upsample
import torchvision.models.resnet as resnet

from fblib.util.mypath import Path

try:
    from itertools import izip
except ImportError:  # python3.x
    izip = zip

__all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
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

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
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


class ResNet_C5(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet_C5, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.output = nn.Conv2d(2048, 1, kernel_size=1, stride=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        input_size = x.size()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.output(x)
        x = upsample(x, size=(input_size[2], input_size[3]), mode='bilinear')
        return x

    def load_pretrained(self, base_network):
        for module, module_ori in izip(self.modules(), base_network.modules()):
            if isinstance(module, nn.Conv2d) and isinstance(module_ori, nn.Conv2d):
                if module.weight.data.shape == module_ori.weight.data.shape:
                    module.weight.data = deepcopy(module_ori.weight.data)
                    module.bias = deepcopy(module_ori.bias)
                else:
                    print('Skipping Conv layer with size: {} and target size: {}'
                          .format(module.weight.data.shape, module_ori.weight.data.shape))
            elif isinstance(module, nn.BatchNorm2d) and isinstance(module_ori, nn.BatchNorm2d):
                if module.weight.data.shape == module_ori.weight.data.shape:
                    # Copy running mean and variance of batchnorm layers!
                    module.running_mean.data = deepcopy(module_ori.running_mean.data)
                    module.running_var.data = deepcopy(module_ori.running_var.data)

                    module.weight.data = deepcopy(module_ori.weight)
                    module.bias.data = deepcopy(module_ori.bias.data)
                else:
                    print('Skipping Batchnorm layer with size: {} and target size: {}'
                          .format(module.weight.data.shape, module_ori.weight.data.shape))


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_C5(BasicBlock, [2, 2, 2, 2], **kwargs)
    model_full = resnet.resnet18(pretrained=True)
    if pretrained:
        model.load_pretrained(model_full)
    return model


def resnet26(pretrained=False):
    """Constructs a ResNet-26 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_C5(Bottleneck, [2, 2, 2, 2])

    if pretrained:

        # Define ResNet26 ImageNet
        model_IN = resnet.ResNet(block=Bottleneck, layers=[2, 2, 2, 2], num_classes=1000)

        # Load checkpoint
        checkpoint = torch.load(
            os.path.join(Path.models_dir(), 'resnet26_epoch_100.pth'), map_location=lambda storage, loc: storage)
        checkpoint = checkpoint['model_state']

        # Handle DataParallel
        if 'module.' in list(checkpoint.keys())[0]:
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                name = k.replace('module.', '')  # remove `module.`
                new_state_dict[name] = v
        else:
            new_state_dict = checkpoint

        # Load pre-trained IN model
        model_IN.load_state_dict(new_state_dict)

        # Load weights to dense-labelling network
        model.load_pretrained(model_IN)

    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_C5(BasicBlock, [3, 4, 6, 3], **kwargs)
    model_full = resnet.resnet34(pretrained=True)
    if pretrained:
        model.load_pretrained(model_full)
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_C5(Bottleneck, [3, 4, 6, 3], **kwargs)
    model_full = resnet.resnet50(pretrained=True)
    if pretrained:
        model.load_pretrained(model_full)
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_C5(Bottleneck, [3, 4, 23, 3], **kwargs)
    model_full = resnet.resnet101(pretrained=True)
    if pretrained:
        model.load_pretrained(model_full)
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_C5(Bottleneck, [3, 8, 36, 3], **kwargs)
    model_full = resnet.resnet152(pretrained=True)
    if pretrained:
        model.load_pretrained(model_full)
    return model


if __name__ == '__main__':
    net = resnet26(pretrained=True)
