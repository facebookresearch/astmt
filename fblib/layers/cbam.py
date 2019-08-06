import torch
from torch import nn


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


class CBAMLayerMultiTask(nn.Module):
    def __init__(self, channel, reduction=16, tasks=None):
        super(CBAMLayerMultiTask, self).__init__()

        if tasks is None:
            self.se = CBAMLayer(channel=channel, reduction=reduction)

        elif type(tasks) == list:
            print('Initializing Dictionary of {} Convolutional Block Attention modules'.format(tasks))
            self.se = nn.ModuleDict()
            for task in self.tasks:
                print('CBAM for task: {}'.format(task))
                self.se[task] = CBAMLayer(channel=channel, reduction=reduction)

        elif type(tasks) == int:
            print('Initializing List of {} Convolutional Block Attention modules'.format(tasks))
            self.se = nn.ModuleList()
            for task in range(self.tasks):
                print('CBAM for task: {}'.format(task))
                self.se[task] = CBAMLayer(channel=channel, reduction=reduction)

    def forward(self, x, task=None):
        if task is not None:
            x = self.se[task](x)
        else:
            x = self.se(x)
        return x


class ConvCoupledCBAM(nn.Module):
    """
    CBAM-layer per task, coupled with convolutions and batchnorm.
    """

    def __init__(self, tasks,
                 process_layers=None,
                 norm=None,
                 norm_kwargs=None,
                 squeeze=False,
                 reduction=16):

        super(ConvCoupledCBAM, self).__init__()

        self.squeeze = squeeze

        if not isinstance(process_layers, list):
            process_layers = [process_layers]

        self.process = nn.Sequential(*process_layers)

        se_module = CBAMLayerMultiTask

        if self.squeeze:
            self.se = se_module(process_layers[-1].out_channels, tasks=tasks, reduction=reduction)

        self.norm = nn.ModuleDict({task: norm(**norm_kwargs) for task in tasks})

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, task):
        x = self.process(x)
        x = self.norm[task](x)
        x = self.relu(x)
        x = self.se(x, task)

        return x
