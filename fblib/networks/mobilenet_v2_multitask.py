import math

import torch.nn as nn

from fblib.layers.attention import Conv2dAttentionAdapters
from fblib.util.custom_container import SequentialMultiTask
from fblib.layers.squeeze import SELayerMultiTask


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
    def __init__(self, inp, oup, stride, expand_ratio,
                 n_tasks=1,
                 adapters=False,
                 attention=False,
                 squeeze=False,
                 bn_per_task=False,
                 binary_attention=False):

        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.squeeze = squeeze
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.pre = None

            # dw
            self.conv_conv = Conv2dAttentionAdapters(in_channels=hidden_dim,
                                                     out_channels=hidden_dim,
                                                     kernel_size=3,
                                                     stride=stride,
                                                     padding=1,
                                                     groups=hidden_dim,
                                                     bias=False,
                                                     n_tasks=n_tasks,
                                                     adapters=adapters,
                                                     attention=attention,
                                                     bn_per_task=bn_per_task,
                                                     binary_attention=binary_attention)

            self.meta = nn.Sequential(nn.BatchNorm2d(hidden_dim),
                                      nn.ReLU6(inplace=True),
                                      # pw-linear
                                      nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                                      nn.BatchNorm2d(oup)
                                      )
            if self.squeeze:
                self.se = SELayerMultiTask(oup, n_tasks=n_tasks)

        else:
            self.pre = nn.Sequential(  # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            )
            # dw
            self.conv_conv = Conv2dAttentionAdapters(in_channels=hidden_dim,
                                                     out_channels=hidden_dim,
                                                     kernel_size=3,
                                                     stride=stride,
                                                     padding=1,
                                                     groups=hidden_dim,
                                                     bias=False,
                                                     n_tasks=n_tasks,
                                                     adapters=adapters,
                                                     attention=attention,
                                                     bn_per_task=bn_per_task,
                                                     binary_attention=binary_attention)

            self.meta = nn.Sequential(nn.BatchNorm2d(hidden_dim),
                                      nn.ReLU6(inplace=True),
                                      # pw-linear
                                      nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                                      nn.BatchNorm2d(oup),
                                      )

            if self.squeeze:
                self.se = SELayerMultiTask(oup, n_tasks=n_tasks)

    def forward(self, x, task=None):
        y = x

        if self.pre is not None:
            y = self.pre(y)

        y = self.conv_conv(y, task)
        y = self.meta(y)

        if self.squeeze:
            y = self.se(y, task)

        if self.use_res_connect:
            return x + y
        else:
            return y


class MobileNetV2CIFARMulti(nn.Module):
    def __init__(self, num_classes=100, input_size=32, width_mult=1.,
                 n_tasks=1, use_orig=False,
                 bn_per_task=False, adapters=False, attention=False, squeeze=False, binary_attention=False):

        super(MobileNetV2CIFARMulti, self).__init__()

        self.num_classes = num_classes
        self.use_orig = use_orig
        self.n_tasks = n_tasks if not self.use_orig else n_tasks + 1
        self.adapters = adapters  # parallel adapters
        self.attention = attention  # Attention modules
        self.squeeze = squeeze  # SE modules
        self.bn_per_task = bn_per_task
        self.binary_attention = binary_attention

        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            # [6, 24, 2, 2], # For imagenet 224 x 224, uncomment and adapt avepool to 8 instead of 7
            [6, 32, 3, 2],
            # [6, 64, 4, 2], # For imagenet 224 x 224, uncomment and adapt avepool to 8 instead of 7
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = nn.ModuleList([conv_bn(3, input_channel, 1)])  # stride 2 for imagenet 224
        # building inverted residual blocks
        for i_stage, (t, c, n, s) in enumerate(interverted_residual_setting):

            output_channel = int(c * width_mult)

            if i_stage < 3:
                for i in range(n):
                    if i == 0:
                        self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                    else:
                        self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                    input_channel = output_channel
            else:
                for i in range(n):
                    if i == 0:
                        self.features.append(block(input_channel, output_channel, s, expand_ratio=t,
                                                   n_tasks=self.n_tasks,
                                                   adapters=self.adapters,
                                                   attention=self.attention,
                                                   squeeze=self.squeeze,
                                                   bn_per_task=self.bn_per_task,
                                                   binary_attention=self.binary_attention))
                    else:
                        self.features.append(block(input_channel, output_channel, 1, expand_ratio=t,
                                                   n_tasks=self.n_tasks,
                                                   adapters=self.adapters,
                                                   attention=self.attention,
                                                   squeeze=self.squeeze,
                                                   bn_per_task=self.bn_per_task,
                                                   binary_attention=self.binary_attention))
                    input_channel = output_channel

        # building last several layers
        self.features.append(nn.ModuleList([conv_1x1_bn(input_channel, self.last_channel)
                                            for i in range(self.n_tasks)]))

        self.avgpool = nn.AvgPool2d(8)

        # building classifiers
        self.classifier = nn.ModuleList()
        if self.use_orig:
            self.classifier.append(SequentialMultiTask(nn.Dropout(0.2),
                                                       nn.Linear(self.last_channel, 100)))
        for i in range(n_tasks):
            self.classifier.append(SequentialMultiTask(nn.Dropout(0.2),
                                                       nn.Linear(self.last_channel, self.num_classes)))

        self._initialize_weights()

    def forward(self, x, tasks=None):
        for i in range(8):
            x = self.features[i](x)

        outputs = []
        for task in range(self.n_tasks):
            tmp = x
            if task in tasks:
                for i in range(8, len(self.features) - 1):
                    tmp = self.features[i](tmp, task)
                tmp = self.features[-1][task](tmp)
                tmp = self.avgpool(tmp)
                outputs.append(self.classifier[task](tmp.view(tmp.size(0), -1)))
        return outputs

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


class MobileNetV2ImageNetMulti(nn.Module):
    def __init__(self, num_classes=1000, input_size=224, width_mult=1.,
                 n_tasks=1, use_orig=False,
                 bn_per_task=False, adapters=False, attention=False, squeeze=False, binary_attention=False):

        super(MobileNetV2ImageNetMulti, self).__init__()

        self.num_classes = num_classes
        self.use_orig = use_orig
        self.n_tasks = n_tasks if not self.use_orig else n_tasks + 1
        self.adapters = adapters  # parallel adapters
        self.attention = attention  # Attention modules
        self.squeeze = squeeze  # SE modules
        self.bn_per_task = bn_per_task
        self.binary_attention = binary_attention

        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
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
        self.features = nn.ModuleList([conv_bn(3, input_channel, 2)])
        # building inverted residual blocks
        for i_stage, (t, c, n, s) in enumerate(interverted_residual_setting):

            output_channel = int(c * width_mult)

            if i_stage < 3:
                for i in range(n):
                    if i == 0:
                        self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                    else:
                        self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                    input_channel = output_channel
            else:
                for i in range(n):
                    if i == 0:
                        self.features.append(block(input_channel, output_channel, s, expand_ratio=t,
                                                   n_tasks=self.n_tasks,
                                                   adapters=self.adapters,
                                                   attention=self.attention,
                                                   squeeze=self.squeeze,
                                                   bn_per_task=self.bn_per_task,
                                                   binary_attention=self.binary_attention))
                    else:
                        self.features.append(block(input_channel, output_channel, 1, expand_ratio=t,
                                                   n_tasks=self.n_tasks,
                                                   adapters=self.adapters,
                                                   attention=self.attention,
                                                   squeeze=self.squeeze,
                                                   bn_per_task=self.bn_per_task,
                                                   binary_attention=self.binary_attention))
                    input_channel = output_channel

        # building last several layers
        self.features.append(nn.ModuleList([conv_1x1_bn(input_channel, self.last_channel)
                                            for i in range(self.n_tasks)]))

        self.avgpool = nn.AvgPool2d(7)

        # building classifiers
        self.classifier = nn.ModuleList()
        if self.use_orig:
            self.classifier.append(SequentialMultiTask(nn.Dropout(0.2),
                                                       nn.Linear(self.last_channel, 100)))
        for i in range(n_tasks):
            self.classifier.append(SequentialMultiTask(nn.Dropout(0.2),
                                                       nn.Linear(self.last_channel, self.num_classes)))

        self._initialize_weights()

    def forward(self, x, tasks=None):
        for i in range(14):
            x = self.features[i](x)

        outputs = []
        for task in range(self.n_tasks):
            tmp = x
            if task in tasks:
                for i in range(14, len(self.features) - 1):
                    tmp = self.features[i](tmp, task)
                tmp = self.features[-1][task](tmp)
                tmp = self.avgpool(tmp)
                outputs.append(self.classifier[task](tmp.view(tmp.size(0), -1)))
        return outputs

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


def mobilenet_v2(num_classes=10, **kwargs):
    model = MobileNetV2CIFARMulti(num_classes=num_classes, **kwargs)

    return model


def visualize_network(net, n_tasks=1):
    import fblib.util.visualize as viz
    net.eval()
    x = torch.randn(1, 3, 32, 32)
    x.requires_grad_()

    # pdf visualizer
    y = net.forward(x, list(range(n_tasks+1)))
    g = viz.make_dot(y, net.state_dict())
    g.view(directory='/private/home/kmaninis')


if __name__ == '__main__':
    import torch

    n_tasks = 2

    net = mobilenet_v2(num_classes=10, n_tasks=n_tasks, use_orig=True, bn_per_task=False,
                       adapters=True, attention=True, squeeze=True, binary_attention=False)

    visualize_network(net, n_tasks=n_tasks)
