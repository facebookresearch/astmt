import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.model(x)
        return x


class MiniDiscriminator(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(MiniDiscriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(in_channels, 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(16, n_classes),
        )

    def forward(self, x):
        x = self.model(x)
        return x


class FullyConvDiscriminator(nn.Module):
    def __init__(self, in_channels, n_classes, kernel_size=1, depth=1):
        super(FullyConvDiscriminator, self).__init__()

        padding = (kernel_size - 1) / 2
        assert(padding == int(padding))
        padding = int(padding)

        print('\nInitializing Fully Convolutional Discriminator with depth: {} and kernel size: {}'
              .format(depth, kernel_size))
        if depth == 1:
            self.model = nn.Sequential(
                nn.Conv2d(in_channels, n_classes, kernel_size=kernel_size, padding=padding, bias=True))
        elif depth == 2:
            self.model = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, bias=False),
                # nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, n_classes, kernel_size=kernel_size, padding=padding, bias=True))

    def forward(self, x):
        x = self.model(x)
        return x


class AvePoolDiscriminator(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(AvePoolDiscriminator, self).__init__()

        self.avepool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.model = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=in_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=in_channels, out_features=n_classes),
        )

    def forward(self, x):
        x = self.avepool(x)
        x = x.view(x.size(0), -1)
        x = self.model(x)
        return x


class ConvDiscriminator(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(ConvDiscriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False, stride=4),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False, stride=4),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False, stride=4),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )
        self.classifier = nn.Linear(in_features=in_channels, out_features=n_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
