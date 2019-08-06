import numpy as np
import torch
import torch.nn as nn


class Smoother(nn.Module):
    def __init__(self, kernel_sizes, sigmas, n_channels):
        super(Smoother, self).__init__()

        conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.ReflectionPad2d(int((kernel_size - 1) / 2.0)),
                nn.Conv2d(
                    in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
                    stride=1, padding=0, bias=False, groups=n_channels,
                )
            )
            for kernel_size in kernel_sizes])

        self.conv_layers = conv_layers
        self.sigmas = sigmas

        self.init_weights()

        for p in self.conv_layers.parameters():
            p.requires_grad = False

    def init_weights(self):
        """
        Initialize the weights of convolutional layers with Gaussian filters.
        """
        print('Initializing weights of Gaussian Smoother')
        for layer, sigma in zip(self.conv_layers, self.sigmas):
            n_out, n_in, h, w = layer[1].weight.data.size()
            weight = self._get_gaussian_filter(h, sigma)
            layer[1].weight.data.zero_()
            for i in range(0, n_out):
                layer[1].weight.data[i, 0, :, :] = weight

    @staticmethod
    def _get_gaussian_filter(size, sigma, normalize=True):
        """
        Make a 2D gaussian kernel.
        """
        assert size % 2 == 1
        factor = (size + 1) // 2
        center = factor - 1
        grid = np.linspace(-center, center, 2 * center + 1)
        grid_v, grid_h = np.meshgrid(grid, grid)

        x = np.exp(-(grid_v ** 2 + grid_h ** 2) / (2 * sigma * sigma))
        if normalize:
            x = x / x.sum()
        return torch.from_numpy(x)

    def forward(self, x, level=0):
        x = self.conv_layers[level](x)

        return x


def test_output():
    import os
    import cv2
    from matplotlib.pyplot import imshow, show
    from fblib import PROJECT_ROOT_DIR

    kernel_sizes = [6 + 1, 12 + 1, 18 + 1]
    sigmas = [k / 3 for k in kernel_sizes]

    net = Smoother(kernel_sizes=kernel_sizes, sigmas=sigmas, n_channels=3)

    img = cv2.imread(os.path.join(PROJECT_ROOT_DIR, 'util', 'img', 'dog.jpg')).astype(np.float32)
    img = np.transpose(img, [2, 0, 1])[np.newaxis, ...]
    img = torch.from_numpy(img)

    for i in range(3):
        pred = np.transpose(net(img, i).squeeze(0).numpy(), [1, 2, 0])
        imshow(pred/255)
        show()


if __name__ == '__main__':
    test_output()
