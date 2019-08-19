# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import sys
import tarfile
import cv2

import numpy as np
import torch.utils.data as data
from six.moves import urllib

from fblib.util.mypath import Path


class MSRA(data.Dataset):
    """
    MSRA10k dataset for Saliency Estimation
    """

    URL = 'https://data.vision.ee.ethz.ch/kmaninis/share/MTL/MSRA10K.tgz'
    FILE = 'MSRA10K.tgz'

    def __init__(self,
                 root=Path.db_root_dir('MSRA10K'),
                 download=True,
                 split='trainval',
                 transform=None,
                 retname=True,
                 overfit=False):

        if download:
            self._download()

        self.transform = transform

        self.retname = retname

        self.root = root
        self.gt_dir = os.path.join(self.root, 'gt')
        self.image_dir = os.path.join(self.root, 'Imgs')

        _splits_dir = os.path.join(self.root, 'gt_sets')

        self.split = split

        if isinstance(self.split, str):
            self.split = [self.split]

        self.images = []
        self.gts = []
        self.im_ids = []

        for sp in self.split:
            with open(os.path.join(os.path.join(_splits_dir, sp + '.txt')), "r") as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()

                _image = os.path.join(self.image_dir, line + ".jpg")
                _gt = os.path.join(self.gt_dir, line + ".png")

                assert os.path.isfile(_image)
                assert os.path.isfile(_gt)
                self.im_ids.append(line)
                self.images.append(_image)
                self.gts.append(_gt)

        assert (len(self.images) == len(self.gts) == len(self.im_ids))

        if overfit:
            n_of = 64
            self.images = self.images[:n_of]
            self.im_ids = self.im_ids[:n_of]

        # Display stats
        print('Number of images: {:d}'.format(len(self.im_ids)))

    def __getitem__(self, index):

        sample = {}

        _img = self._load_img(index)
        sample['image'] = _img

        _sal = self._load_sal(index)
        sample['sal'] = _sal

        if self.retname:
            sample['meta'] = {'image': str(self.im_ids[index]),
                              'im_size': (_img.shape[0], _img.shape[1])}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
            return len(self.im_ids)

    def _load_img(self, index):
        # Read Image
        _img = cv2.imread(self.images[index])[:, :, ::-1].astype(np.float32)

        return _img

    def _load_sal(self, index):
        # Read Target object
        _gt = cv2.imread(self.gts[index], flags=0).astype(np.float32) / 255.

        return _gt

    def _download(self):
        _fpath = os.path.join(Path.db_root_dir(), self.FILE)

        if os.path.isfile(_fpath):
            print('Files already downloaded')
            return
        else:
            print('Downloading ' + self.URL + ' to ' + _fpath)

            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> %s %.1f%%' %
                                 (_fpath, float(count * block_size) /
                                  float(total_size) * 100.0))
                sys.stdout.flush()

            urllib.request.urlretrieve(self.URL, _fpath, _progress)

        # extract file
        cwd = os.getcwd()
        print('\nExtracting tar file')
        tar = tarfile.open(_fpath)
        os.chdir(Path.db_root_dir())
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        print('Done!')

    def __str__(self):
        return 'MSRA(split=' + str(self.split) + ')'


if __name__ == '__main__':

    from matplotlib import pyplot as plt

    dataset = MSRA()

    for i, sample in enumerate(dataset):
        plt.imshow(sample['image']/255)
        plt.show()
        plt.imshow(sample['sal'])
        plt.show()



