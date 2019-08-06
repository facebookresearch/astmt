import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import math

from fblib.layers.squeeze import SELayerMultiTask


class AttentionModule(nn.Module):
    def __init__(self, input_size):
        super(AttentionModule, self).__init__()

        # randomly initialize the parameters
        self.weight = nn.Parameter(torch.rand(1, input_size, 1, 1) - 0.5)
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        return torch.mul(self.sigm(self.weight), x)


class Conv2dAttentionAdapters(nn.Module):
    def __init__(self, in_size, out_size, n_tasks=2, adapters=False, attention=False):
        super(Conv2dAttentionAdapters, self).__init__()

        self.adapters = adapters
        self.attention = attention

        self.conv = nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=3, stride=1, padding=1)

        if self.attention:
            self.attend = nn.ModuleList([AttentionModule(out_size) for i in range(n_tasks)])

        if self.adapters:
            self.adapt = nn.ModuleList([
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=False) for i in range(n_tasks)])

    def forward(self, x, task=None):
        if self.adapters:
            adapt = self.adapt[task](x)

        x = self.conv(x)

        if self.attention:
            x = self.attend[task](x)

        if self.adapters:
            x += adapt

        return x


class MiniNet(nn.Module):
    def __init__(self, multitask=False, adapters=False, attention=False, out_c=2):
        super(MiniNet, self).__init__()
        self.multitask = multitask
        self.attention = attention
        self.adapters = adapters

        self.conv1 = Conv2dAttentionAdapters(in_size=1, out_size=out_c)
        self.conv2 = Conv2dAttentionAdapters(in_size=out_c, out_size=out_c,
                                             adapters=self.adapters, attention=self.attention)
        self.conv3 = Conv2dAttentionAdapters(in_size=out_c, out_size=out_c,
                                             adapters=self.adapters, attention=self.attention)

        self.avgpool = nn.AvgPool2d(7)

        if self.multitask:
            self.fc1 = nn.ModuleList([nn.Linear(out_c, 2) for i in range(2)])
        else:
            self.fc1 = nn.Linear(out_c, 2)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def forward(self, x, tasks):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        if self.multitask:
            outs = []
            for task in range(2):
                if task in tasks:
                    tmp = F.relu(self.conv2(x, task))
                    tmp = F.max_pool2d(tmp, kernel_size=2, stride=2)
                    tmp = F.relu(self.conv3(tmp, task))

                    tmp = self.avgpool(tmp)
                    tmp = tmp.view(-1, tmp.size()[1])
                    outs.append(self.fc1[task](tmp))
                else:
                    outs.append([])
        else:
            outs = []
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, kernel_size=2, stride=2)
            x = F.relu(self.conv3(x))
            x = self.avgpool(x)
            x = x.view(-1, x.size()[1])
            outs.append(self.fc1(x))
        return outs


class LeNetMT(nn.Module):
    def __init__(self, n_tasks=2, squeeze=False, use_bn=False):
        super(LeNetMT, self).__init__()
        self.squeeze = squeeze
        self.use_bn = use_bn

        # Network definition
        self.conv1 = nn.Conv2d(1, 10, 5)
        if self.squeeze:
            self.se1 = SELayerMultiTask(10, reduction=2, n_tasks=n_tasks)
        if self.use_bn:
            if self.squeeze:
                self.bn1 = nn.ModuleList([nn.BatchNorm2d(10) for i in range(n_tasks)])
            else:
                self.bn1 =nn.BatchNorm2d(10)

        self.conv2 = nn.Conv2d(10, 20, 5)
        if self.squeeze:
            self.se2 = SELayerMultiTask(20, reduction=4, n_tasks=n_tasks)
        if self.use_bn:
            if self.squeeze:
                self.bn2 = nn.ModuleList([nn.BatchNorm2d(20) for i in range(n_tasks)])
            else:
                self.bn2 = nn.BatchNorm2d(20)

        self.fc1 = nn.Linear(20 * 5 * 5, 50)
        self.fc2 = nn.ModuleList([nn.Linear(50, 10) for i in range(n_tasks)])

        self._create_gen_and_task_lst()

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def forward(self, x, task):
        # Conv1, bn, relu, squeeze
        x = self.conv1(x)
        if self.use_bn:
            if self.squeeze:
                x = self.bn1[task](x)
            else:
                x = self.bn1(x)
        x = F.relu(x)
        if self.squeeze:
            x = self.se1(x, task)
        x = F.max_pool2d(x, 2)

        # Conv2, bn, relu, squeeze
        x = self.conv2(x)
        if self.use_bn:
            if self.squeeze:
                x = self.bn2[task](x)
            else:
                x = self.bn2(x)
        if self.squeeze:
            x = self.se2(x, task)
        x = F.max_pool2d(x, 2)

        # FC layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2[task](x)

        return x

    def _create_gen_and_task_lst(self):
        """Separate generic and task-specific modules"""

        self.generic_modules = [self.conv1, self.conv2, self.fc1]
        self.task_modules = [self.fc2]

        if self.squeeze:
            self.task_modules.extend([self.se1, self.se2])
            if self.use_bn:
                self.task_modules.extend([self.bn1, self.bn2])
        else:
            if self.use_bn:
                self.generic_modules.extend([self.bn1, self.bn2])

    def get_lr_params(self, part=None):
        """
        Generator for generic and task specific layers
        """

        if part == 'generic':
            b = self.generic_modules
        elif part == 'task_specific':
            b = self.task_modules
        else:
            b = [self]

        for i in range(len(b)):
            for name, k in b[i].named_parameters():
                if k.requires_grad:
                    yield k

    def get_param_dict(self, part=None):
        """
        Used by Quadratic Optimization (NIPS 2018, Koltun)
        Gets a dictionary of generic and task specific parameters with their names
        """
        dic = {}
        for name, k in self.named_parameters():
            if k.requires_grad:
                condition = (name.find('.0.') >= 0 or name.find('.1.') >= 0)  # Change this hack
                if part == 'generic' and condition:
                    continue
                if part == 'task_specific' and not condition:
                    continue
                dic[name] = k
        return dic

    def __str__(self):
        return "LeNetMT"


if __name__ == '__main__':
    from fblib.dataloaders import mnist_multitask as mnist
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader

    multitask = True

    # Transformations
    train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

    # Load network
    net = MiniNet(multitask=True)
    db_train = mnist.MNIST(train=True, multitask=multitask, transform=train_transform)
    trainloader = DataLoader(db_train, batch_size=64, shuffle=True, num_workers=2)

    for ii, sample in enumerate(trainloader):
        input_var = sample[0].requires_grad_()
        outputs = net(input_var)
