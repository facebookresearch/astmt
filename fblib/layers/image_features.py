# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch.nn as nn
import torch.nn.functional as F


class Normalize(object):
    """Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for i in range(len(self.mean)):
            tensor[:, i, :, :].sub_(self.mean[i]).div_(self.std[i])
        return tensor


class ImageFeatures(nn.Module):
    """
    Forward pass of an image on a pre-trained imagenet model.
    Resurns output and features of the forward pass.
    """
    def __init__(self, net, mean=None, std=None):
        super(ImageFeatures, self).__init__()

        if not mean:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

        self.normalize = Normalize(mean=mean, std=std)
        self.net = net

    def forward(self, x):
        x = (x - x.min()) / (x.max() - x.min())
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        x = self.normalize(x)
        out, features = self.net(x)

        return out, features


def main():
    import os
    import torch
    import pickle
    import cv2
    import numpy as np
    import urllib.request
    from fblib import PROJECT_ROOT_DIR
    from fblib.networks.classification.resnet import resnet101

    classes = pickle.load(urllib.request.urlopen(
        'https://gist.githubusercontent.com/yrevar/6135f1bd8dcf2e0cc683/raw/d133d61a09d7e5a3b36b8c111a8dd5c4b5d560ee'
        '/imagenet1000_clsid_to_human.pkl'))

    model = resnet101(pretrained=True, features=True)
    model = ImageFeatures(model)

    img = cv2.imread(os.path.join(PROJECT_ROOT_DIR, 'util/img/cat.jpg')).astype(np.float32)

    img = img[:, :, :, np.newaxis]
    img = img.transpose((3, 2, 0, 1))
    img = torch.from_numpy(img.astype(np.float32))

    model = model.eval()
    with torch.no_grad():
        output, features = model(img)
        output = torch.nn.functional.softmax(output, dim=1)
        print(output.max())
        print(output.argmax())
        print(classes[np.asscalar(output.argmax().numpy())])


if __name__ == '__main__':
    main()
