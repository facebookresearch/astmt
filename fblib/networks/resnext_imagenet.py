import torch.nn as nn
from copy import deepcopy
import math
import torch.nn.functional as F

try:
    from itertools import izip
except ImportError:  # python3.x
    izip = zip

__all__ = ['resnext50_32x4d', 'resnext101_32x4d']


class ResNeXtBottleneck(nn.Module):
    expansion = 4
    """
  RexNeXt bottleneck type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
  """

    def __init__(self, inplanes, planes, cardinality, base_width=4, stride=1, downsample=None):
        super(ResNeXtBottleneck, self).__init__()

        D = int(math.floor(planes * (base_width / 64.0)))
        C = cardinality

        self.conv_reduce = nn.Conv2d(inplanes, D * C, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_reduce = nn.BatchNorm2d(D * C)

        self.conv_conv = nn.Conv2d(D * C, D * C, kernel_size=3, stride=stride, padding=1, groups=cardinality,
                                   bias=False)
        self.bn = nn.BatchNorm2d(D * C)

        self.conv_expand = nn.Conv2d(D * C, planes * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_expand = nn.BatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)

        if downsample is not None:
            self.downsample = downsample

    def forward(self, x):
        residual = x

        bottleneck = self.conv_reduce(x)
        bottleneck = self.relu(self.bn_reduce(bottleneck))

        bottleneck = self.conv_conv(bottleneck)
        bottleneck = self.relu(self.bn(bottleneck))

        bottleneck = self.conv_expand(bottleneck)
        bottleneck = self.bn_expand(bottleneck)

        if self.downsample is not None:
            residual = self.downsample(x)

        return self.relu(residual + bottleneck)


class ResNeXt_C5(nn.Module):
    def __init__(self, block, layers, cardinality=32, base_width=4):
        self.inplanes = 64
        super(ResNeXt_C5, self).__init__()
        self.cardinality = cardinality
        self.base_width = base_width

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
        self.classifier = nn.Linear(2048, 10000, bias=True)
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
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample, cardinality=self.cardinality,
                            base_width=self.base_width))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cardinality=self.cardinality, base_width=self.base_width))

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
        x = self.classifier(x)
        return x

    def load_pretrained(self, base_network):
        modules_new = [x for x in self.modules() if (isinstance(x, nn.Conv2d) or isinstance(x, nn.BatchNorm2d)
                                                             or isinstance(x, nn.Linear))]
        modules_ori = [x for x in base_network.modules() if (isinstance(x, nn.Conv2d) or isinstance(x, nn.BatchNorm2d)
                                                             or isinstance(x, nn.Linear))]

        assert(len(modules_ori) == len(modules_new))
        for module, module_ori in izip(modules_new, modules_ori):
            if isinstance(module, nn.Conv2d) and isinstance(module_ori, nn.Conv2d):
                if module.weight.data.shape == module_ori.weight.data.shape:
                    module.weight.data = deepcopy(module_ori.weight.data)
                    if module_ori.bias is not None:
                        module.bias.data = deepcopy(module_ori.bias.data)
                else:
                    print('This should not happen. Skipping Conv layer with size: {} and target size: {}'
                          .format(module.weight.data.shape, module_ori.weight.data.shape))
            elif isinstance(module, nn.BatchNorm2d) and isinstance(module_ori, nn.BatchNorm2d):
                if module.weight.data.shape == module_ori.weight.data.shape:

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


def resnext50_32x4d(pretrained=False, debug=False):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNeXt_C5(ResNeXtBottleneck, [3, 4, 6, 3], cardinality=32, base_width=4)
    if pretrained:
        from fblib.networks.torch2pytorch import resnext_50_32x4d
        model_full = resnext_50_32x4d.resnext_50_32x4d()
        model.load_pretrained(model_full)

    if pretrained and debug:
        return model, model_full
    else:
        return model


def resnext101_32x4d(pretrained=False, debug=False):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNeXt_C5(ResNeXtBottleneck, [3, 4, 23, 3], cardinality=32, base_width=4)
    if pretrained:
        from fblib.networks.torch2pytorch import resnext_101_32x4d
        model_full = resnext_101_32x4d.resnext_101_32x4d()
        model.load_pretrained(model_full)

    if pretrained and debug:
        return model, model_full
    else:
        return model


def test_reproduce():
    import pickle
    classes = pickle.load(urllib.request.urlopen(
        'https://gist.githubusercontent.com/yrevar/6135f1bd8dcf2e0cc683/raw/d133d61a09d7e5a3b36b8c111a8dd5c4b5d560ee'
        '/imagenet1000_clsid_to_human.pkl'))

    model, model_full = resnext101_32x4d(pretrained=True, debug=True)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img = cv2.imread(
        os.path.join(PROJECT_ROOT_DIR, 'util/img/dog.jpg')) \
              .astype(np.float32) / 255.

    img = cv2.resize(img, dsize=(224, 224))
    img = (img - mean) / std

    img = img[:, :, :, np.newaxis]
    img = img.transpose((3, 2, 0, 1))
    img = torch.from_numpy(img.astype(np.float32))

    model_full.eval()
    with torch.no_grad():
        output_orig = F.softmax(model_full(img), dim=1)
        print(output_orig.max())
        print(output_orig.argmax())
        print(classes[np.asscalar(output_orig.argmax().numpy())])

    model = model.eval()
    with torch.no_grad():
        output = F.softmax(model(img), dim=1)
        print(output.max())
        print(output.argmax())
        print(classes[np.asscalar(output.argmax().numpy())])


def test_gflops():
    from fblib.util.model_resources.flops import compute_gflops
    from torchvision.models import resnet

    x50 = resnext101_32x4d(pretrained=False)

    print('GFLOPS for ResNeXt: {}'.format(compute_gflops(x50)))

    res50 = resnet.resnet101(pretrained=False)
    print('GFLOPS for ResNet: {}'.format(compute_gflops(res50)))


if __name__ == '__main__':
    import os
    import torch
    import cv2
    import numpy as np
    import urllib.request
    from fblib import PROJECT_ROOT_DIR

    test_gflops()
    test_reproduce()

