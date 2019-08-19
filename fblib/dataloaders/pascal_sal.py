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

from PIL import Image
import numpy as np
import torch.utils.data as data
from six.moves import urllib

from fblib.util.mypath import Path


class PASCALS(data.Dataset):

    URL = 'https://data.vision.ee.ethz.ch/kmaninis/share/MTL/PASCAL-S.tgz'
    FILE = 'PASCAL-S.tgz'

    def __init__(self,
                 root=Path.db_root_dir('PASCAL-S'),
                 download=True,
                 transform=None,
                 retname=True,
                 overfit=False,
                 threshold=None,
                 ):

        self.root = root
        _image_dir = os.path.join(self.root, 'images')
        _sal_dir = os.path.join(self.root, 'masks')
        _split_dir = os.path.join(self.root, 'gt_sets')

        if download:
            self._download()

        self.transform = transform
        self.threshold = threshold
        self.retname = retname

        self.im_ids = []
        self.images = []
        self.sals = []

        print('Initializing dataloader for PASCAL Saliency')
        with open(os.path.join(os.path.join(_split_dir, 'all.txt')), 'r') as f:
                lines = f.read().splitlines()

        for ii, line in enumerate(lines):

            # Images
            _image = os.path.join(_image_dir, line + '.jpg')
            assert os.path.isfile(_image)
            self.images.append(_image)
            self.im_ids.append(line.rstrip('\n'))

            # Saliency
            _sal = os.path.join(_sal_dir, line + '.png')
            assert os.path.isfile(_sal)
            self.sals.append(_sal)

        assert (len(self.images) == len(self.sals))

        # Uncomment to overfit to one image
        if overfit:
            n_of = 64
            self.im_ids = self.im_ids[:n_of]
            self.images = self.images[:n_of]

        # Display stats
        print('Number of dataset images: {:d}'.format(len(self.images)))

    def __getitem__(self, index):

        sample = {}

        # Load Image
        _img = self._load_img(index)
        sample['image'] = _img

        # Load Saliency
        _sal = self._load_sal(index)
        if _sal.shape != _img.shape[:2]:
            _sal = cv2.resize(_sal, _img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
        sample['sal'] = _sal

        if self.retname:
            sample['meta'] = {'image': str(self.im_ids[index]),
                              'im_size': (_img.shape[0], _img.shape[1])}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.images)

    def _load_img(self, index):
        #
        _img = cv2.imread(self.images[index])[:, :, ::-1].astype(np.float32)

        return _img

    def _load_sal(self, index):
        tmp = np.array(Image.open(self.sals[index])) / 255.
        if self.threshold:
            _sal = (tmp > self.threshold).astype(np.float32)
        else:
            _sal = tmp.astype(np.float32)
        return _sal

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
        return 'PASCAL-S()'


if __name__ == '__main__':
    from matplotlib.pyplot import imshow, show

    dataset = PASCALS(threshold=.5)

    for i, sample in enumerate(dataset):
        imshow(sample['image'] / 255.)
        show()
        imshow(sample['sal'])
        show()
