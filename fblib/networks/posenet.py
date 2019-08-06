import math

import torch
import torch.nn as nn
import torch.nn.functional as F

Pool = nn.MaxPool2d


def batchnorm(x):
    return nn.BatchNorm2d(x.size()[1])(x)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, (1. / n) ** 0.5)


class Full(nn.Module):
    def __init__(self, inp_dim, out_dim, bn=False, relu=False):
        super(Full, self).__init__()
        self.fc = nn.Linear(inp_dim, out_dim, bias=True)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        x = self.fc(x.view(x.size()[0], -1))
        if self.relu is not None:
            x = self.relu(x)
        if self.bn is not None:
            x = self.bn(x)
        return x


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=True)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.bn is not None:
            x = self.bn(x)
        return x


class Hourglass(nn.Module):
    def __init__(self, n, f, bn=None, increase=128):
        super(Hourglass, self).__init__()
        nf = f + increase
        self.up1 = Conv(f, f, 3, bn=bn)
        # Lower branch
        self.pool1 = Pool(2, 2)
        self.low1 = Conv(f, nf, 3, bn=bn)
        # Recursive hourglass
        if n > 1:
            self.low2 = Hourglass(n - 1, nf, bn=bn)
        else:
            self.low2 = Conv(nf, nf, 3, bn=bn)
        self.low3 = Conv(nf, f, 3)

    def forward(self, x):
        up1 = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = F.interpolate(low3, scale_factor=2, mode='nearest')
        return up1 + up2


class Merge(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=False)

    def forward(self, x):
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, n, f, bn=None, increase=128, n_classes=1000):
        super(Encoder, self).__init__()

        layers = []
        for i in range(n):
            nf = f + increase
            layers.append(Conv(f, f, 3, bn=bn))
            # Lower branch
            layers.append(Pool(2, 2))
            layers.append(Conv(f, nf, 3, bn=bn))
            f = nf

        self.encode = nn.Sequential(*layers)

        self.avgpool = nn.AvgPool2d(8)
        self.classifier = nn.Linear(nf, n_classes)

    def forward(self, x):
        x = self.encode(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return self.classifier(x)


class PoseNet(nn.Module):
    def __init__(self, nstack, inp_dim, oup_dim, bn=False, increase=128, train_imagenet=False):
        super(PoseNet, self).__init__()
        self.pre = nn.Sequential(
            Conv(3, 64, 7, 2, bn=bn),
            Conv(64, 128, bn=bn),
            Pool(2, 2),
            Conv(128, 128, bn=bn),
            Conv(128, inp_dim, bn=bn)
        )
        self.features = nn.ModuleList([
            nn.Sequential(
                Hourglass(4, inp_dim, bn, increase),
                Conv(inp_dim, inp_dim, 3, bn=bn),
                Conv(inp_dim, inp_dim, 3, bn=bn)
            ) for i in range(nstack)])

        if not train_imagenet:
            self.outs = nn.ModuleList([Conv(inp_dim, oup_dim, 1, relu=False, bn=False) for i in range(nstack)])
            self.merge_features = nn.ModuleList([Merge(inp_dim, inp_dim) for i in range(nstack - 1)])
            self.merge_preds = nn.ModuleList([Merge(oup_dim, inp_dim) for i in range(nstack - 1)])

        self.nstack = nstack

        # Needed for ImageNet
        if train_imagenet:
            self.encoder = nn.ModuleList([Encoder(3, inp_dim, bn=bn) for i in range(self.nstack)])

        self._initialize_weights()

    # Dirty temporary hack: replace to forward() when using dense prediction
    def forward_pose(self, x):
        x = self.pre(x)
        preds = []
        for i in range(self.nstack):
            feature = self.features[i](x)
            preds.append(self.outs[i](feature))
            if i != self.nstack - 1:
                x = x + self.merge_preds[i](preds[-1]) + self.merge_features[i](feature)
        return torch.stack(preds, 1)

    # Forward for ImageNet training
    def forward(self, x):
        x = self.pre(x)

        for i in range(self.nstack):
            feature = self.features[i](x)
            x = x + feature
            if i == 0:
                out = self.encoder[i](x)
            else:
                out += self.encoder[i](x)

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()


def posenet_imagenet():
    """
    Stacked Hourglasses model for ImageNet
    """
    return PoseNet(nstack=4,
                   inp_dim=256,
                   oup_dim=68,
                   bn=True,
                   train_imagenet=True)


def test_gflops():
    from fblib.util.model_resources.flops import compute_gflops

    net = posenet_imagenet()

    print('GFLOPS for PoseNet: {}'.format(compute_gflops(net, (1, 3, 256, 256))))


def test_visualize():
    net = posenet_imagenet()

    inputs = torch.rand(32, 3, 256, 256).requires_grad_().cuda()
    outputs = net(inputs)

    g = viz.make_dot(outputs, net.state_dict())
    g.view()


if __name__ == '__main__':
    import os
    from fblib.util import visualize as viz

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    test_gflops()