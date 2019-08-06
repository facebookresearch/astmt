import os
import math

import torch
import torch.nn as nn


from fblib.util.mypath import Path


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1., last_channel=1280):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class MobileNetV2Features(MobileNetV2):

    def __init__(self, n_class=1000, input_size=224, width_mult=1., last_channel=1280):
        super(MobileNetV2Features, self).__init__(n_class=n_class,
                                                  input_size=input_size,
                                                  width_mult=width_mult,
                                                  last_channel=last_channel)

    def forward(self, x):
        x = self.features(x)
        features = x.mean(3).mean(2)
        x = self.classifier(features)

        return x, features


def mobilenet_v2(pretrained=False, features=False, n_class=1000, last_channel=1280):

    if not features:
        model = MobileNetV2(n_class=n_class, last_channel=last_channel)
    else:
        model = MobileNetV2Features(n_class=n_class, last_channel=last_channel)

    if pretrained:
        state_dict = torch.load(os.path.join(Path.models_dir(), 'mobilenet_v2_' + str(last_channel) + '.pth.tar'),
                                map_location='cpu')

        model.load_state_dict(state_dict)
    return model


def test_visualize_network():
    import fblib.util.visualize as viz
    net = mobilenet_v2()
    net.eval()
    x = torch.randn(1, 3, 224, 224)
    x.requires_grad_()

    # pdf visualizer
    y = net.forward(x)
    g = viz.make_dot(y, net.state_dict())
    g.view(directory='.')

def test_reproduce():
    import os
    import cv2
    import numpy as np
    import pickle
    import urllib.request
    import torch.nn.functional as F
    from fblib import PROJECT_ROOT_DIR

    classes = pickle.load(urllib.request.urlopen(
        'https://gist.githubusercontent.com/yrevar/6135f1bd8dcf2e0cc683/raw/'
        'd133d61a09d7e5a3b36b8c111a8dd5c4b5d560ee/imagenet1000_clsid_to_human.pkl'))

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img = cv2.imread(
        os.path.join(PROJECT_ROOT_DIR, 'util/img/cat.jpg')) \
              .astype(np.float32) / 255.

    img = cv2.resize(img, dsize=(224, 224))
    img = (img - mean) / std

    img = img[:, :, :, np.newaxis]
    img = img.transpose((3, 2, 0, 1))
    img = torch.from_numpy(img.astype(np.float32))

    net = mobilenet_v2(pretrained=True, features=True)

    net.eval()
    with torch.no_grad():
        output, features = net(img)
        output = F.softmax(output, dim=1)
        print(output.max())
        print(output.argmax())
        print(classes[np.asscalar(output.argmax().numpy())])


if __name__ == '__main__':
    test_visualize_network()
