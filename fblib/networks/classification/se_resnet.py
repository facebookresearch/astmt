# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import math

import torch
import torch.nn as nn
from collections import OrderedDict

from fblib.util.mypath import Path

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

model_urls = {
    'se_resnet18': 'https://data.vision.ee.ethz.ch/kmaninis/share/MTL//models/se_resnet18-23d68cfd8.pth',
    'se_resnet26': 'https://data.vision.ee.ethz.ch/kmaninis/share/MTL//models/se_resnet26-5eb336d20.pth',
    'se_resnet50': 'https://data.vision.ee.ethz.ch/kmaninis/share/MTL//models/se_resnet50-ad8889f9f.pth',
    'se_resnet101': 'https://data.vision.ee.ethz.ch/kmaninis/share/MTL//models/se_resnet101-8dbb64f8e.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


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


class CBAMLayer(nn.Module):

    def __init__(self, channel, reduction=16):
        super(CBAMLayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
        )

        self.assemble = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)

    def forward(self, x):

        x = self._forward_se(x)
        x = self._forward_spatial(x)

        return x

    def _forward_se(self, x):

        # Channel attention module (SE with max-pool and average-pool)
        b, c, _, _ = x.size()
        x_avg = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        x_max = self.fc(self.max_pool(x).view(b, c)).view(b, c, 1, 1)

        y = torch.sigmoid(x_avg + x_max)

        return x * y

    def _forward_spatial(self, x):

        # Spatial attention module
        x_avg = torch.mean(x, 1, True)
        x_max, _ = torch.max(x, 1, True)
        y = torch.cat((x_avg, x_max), 1)
        y = torch.sigmoid(self.assemble(y))

        return x * y


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16, attention='se'):
        super(SEBasicBlock, self).__init__()

        if attention == 'se':
            attention_layer = SELayer
        elif attention == 'cbam':
            attention_layer = CBAMLayer
        else:
            raise NotImplementedError

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = attention_layer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16, attention='se'):
        super(SEBottleneck, self).__init__()

        if attention == 'se':
            attention_layer = SELayer
        elif attention == 'cbam':
            attention_layer = CBAMLayer
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = attention_layer(planes * 4, reduction)
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


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, attention='se'):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.attention = attention

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

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
        layers.append(block(self.inplanes, planes, stride, downsample, attention=self.attention))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, attention=self.attention))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResNetFeatures(ResNet):

    def __init__(self, block, layers, num_classes=1000, attention='se'):
        print('Initializing ResNet with Feature output')
        super(ResNetFeatures, self).__init__(block, layers, num_classes, attention)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        features = x
        x = self.fc(x)

        return x, features


def get_state_dict_se(model_name, remote=True):
    # Load checkpoint
    if remote:
        checkpoint = load_state_dict_from_url(model_urls[model_name], map_location='cpu', progress=True)
    else:
        checkpoint = torch.load(
            os.path.join(Path.models_dir(), model_name + '.pth'), map_location=lambda storage, loc: storage)
    checkpoint = checkpoint['model_state']

    # Handle DataParallel
    if 'module.' in list(checkpoint.keys())[0]:
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k.replace('module.', '')  # remove `module.`
            new_state_dict[name] = v
    else:
        new_state_dict = checkpoint

    return new_state_dict


def se_resnet18(num_classes=1000, pretrained=False, features=False, attention='se'):
    """Constructs a ResNet-18 model.
    Args:
        num_classes: number of output classes
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        attention: 'se' or 'cbam'
    """
    if not features:
        model = ResNet(SEBasicBlock, [2, 2, 2, 2], num_classes=num_classes, attention=attention)
    else:
        model = ResNetFeatures(SEBasicBlock, [2, 2, 2, 2], num_classes=num_classes, attention=attention)

    model.avgpool = nn.AdaptiveAvgPool2d(1)

    if pretrained:
        print('Loading se_resnet18 Imagenet')
        new_state_dict = get_state_dict_se(attention + '_resnet18')

        # Load pre-trained IN model
        model.load_state_dict(new_state_dict)

    return model


def se_resnet26(num_classes=1000, pretrained=False, features=False, attention='se'):
    """Constructs a ResNet-26 model.

    Args:
        num_classes: number of output classes
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        attention: 'se' or 'cbam'
    """

    if not features:
        model = ResNet(SEBottleneck, [2, 2, 2, 2], num_classes=num_classes, attention=attention)
    else:
        model = ResNetFeatures(SEBottleneck, [2, 2, 2, 2], num_classes=num_classes, attention=attention)

    model.avgpool = nn.AdaptiveAvgPool2d(1)

    if pretrained:
        print('Loading se_resnet26 Imagenet')
        new_state_dict = get_state_dict_se(attention + '_resnet26')

        # Load pre-trained IN model
        model.load_state_dict(new_state_dict)

    return model


def se_resnet50(num_classes=1000, pretrained=False, features=False, attention='se'):
    """Constructs a ResNet-50 model.

    Args:
        num_classes: number of output classes
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        attention: 'se' or 'cbam'
    """

    if not features:
        model = ResNet(SEBottleneck, [3, 4, 6, 3], num_classes=num_classes, attention=attention)
    else:
        model = ResNetFeatures(SEBottleneck, [3, 4, 6, 3], num_classes=num_classes, attention=attention)

    model.avgpool = nn.AdaptiveAvgPool2d(1)

    if pretrained:
        print('Loading se_resnet50 Imagenet')
        new_state_dict = get_state_dict_se(attention + '_resnet50')

        # Load pre-trained IN model
        model.load_state_dict(new_state_dict)

    return model


def se_resnet101(num_classes=1000, pretrained=False, features=False, attention='se'):
    """Constructs a ResNet-101 model.

    Args:
        num_classes: number of output classes
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        attention: 'se' or 'cbam'
    """
    if not features:
        model = ResNet(SEBottleneck, [3, 4, 23, 3], num_classes=num_classes, attention=attention)
    else:
        model = ResNetFeatures(SEBottleneck, [3, 4, 23, 3], num_classes=num_classes, attention=attention)

    model.avgpool = nn.AdaptiveAvgPool2d(1)

    if pretrained:
        print('Loading se_resnet101 Imagenet')
        new_state_dict = get_state_dict_se(attention + '_resnet101')

        # Load pre-trained IN model
        model.load_state_dict(new_state_dict)

    return model


def test_visualize_graph():
    import fblib.util.visualize as viz

    net = se_resnet26(pretrained=False, attention='se')
    net.eval()

    x = torch.randn(2, 3, 224, 224)
    x.requires_grad_()
    y = net(x)

    # pdf visualizer
    g = viz.make_dot(y, net.state_dict())
    g.view(directory='./')


def test_reproduce():
    import os
    import torch
    import pickle
    import cv2
    import numpy as np
    import urllib.request
    from fblib import PROJECT_ROOT_DIR
    classes = pickle.load(urllib.request.urlopen(
        'https://gist.githubusercontent.com/yrevar/6135f1bd8dcf2e0cc683/raw/d133d61a09d7e5a3b36b8c111a8dd5c4b5d560ee'
        '/imagenet1000_clsid_to_human.pkl'))

    model = se_resnet26(pretrained=True, attention='se')

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img = cv2.imread(os.path.join(PROJECT_ROOT_DIR, 'util/img/cat.jpg')).astype(np.float32) / 255.

    img = cv2.resize(img, dsize=(224, 224))
    img = (img - mean) / std

    img = img[:, :, :, np.newaxis]
    img = img.transpose((3, 2, 0, 1))
    img = torch.from_numpy(img.astype(np.float32))

    model = model.eval()
    with torch.no_grad():
        output = model(img)
        output = torch.nn.functional.softmax(output, dim=1)
        print('Class id: {}, class name: {}, probability: {:.2f}'''
              ''.format(output.argmax().item(), classes[output.argmax().item()], output.max().item()))


if __name__ == '__main__':
    test_visualize_graph()
