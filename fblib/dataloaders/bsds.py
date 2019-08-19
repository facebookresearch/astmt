# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import sys
import tarfile

from PIL import Image
import numpy as np
from glob import glob
import scipy.io as sio
import torch.utils.data as data
from six.moves import urllib

from fblib.util.mypath import Path


class BSDS500(data.Dataset):
    """
    BSDS500 datasets for edge detection.
    """

    URL = 'https://data.vision.ee.ethz.ch/kmaninis/share/MTL/BSDS500.tgz'
    FILE = 'BSDS500.tgz'

    def __init__(self,
                 root=Path.db_root_dir('BSDS500'),
                 download=True,
                 split=['train', 'val'],
                 transform=None,
                 retname=True,
                 n_votes=1,
                 overfit=False):

        if download:
            self._download()

        self.transform = transform

        self.retname = retname
        self.n_votes = n_votes

        self.root = root
        self.gt_dir = os.path.join(self.root, 'data', 'groundTruth')
        self.image_dir = os.path.join(self.root, 'data', 'images')

        _splits_dir = os.path.join(self.root, 'lists')
        if not os.path.exists(os.path.join(_splits_dir)):
            os.mkdir(os.path.join(_splits_dir))

        self.split = split
        self._get_images_trainval()

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

                _image = os.path.join(self.image_dir, sp, line + ".jpg")
                _gt = os.path.join(self.gt_dir, sp, line + ".mat")

                assert os.path.isfile(_image)
                assert os.path.isfile(_gt)
                self.im_ids.append(line)
                self.images.append(_image)
                self.gts.append(_gt)

        assert (len(self.images) == len(self.gts) == len(self.im_ids))

        if overfit:
            n_of = 16
            self.images = self.images[:n_of]
            self.gts = self.gts[:n_of]
            self.im_ids = self.im_ids[:n_of]

        # Display stats
        print('Number of images: {:d}'.format(len(self.im_ids)))

    def __getitem__(self, index):

        sample = {}

        _img = self._load_img(index)
        sample['image'] = _img

        _edge = self._load_edge(index)
        sample['edge'] = _edge

        if self.retname:
            sample['meta'] = {'image': str(self.im_ids[index]),
                              'im_size': (_img.shape[0], _img.shape[1])}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
            return len(self.im_ids)

    def _get_images_trainval(self):
        for sp in self.split:
            if os.path.isfile(os.path.join(self.root, 'lists', sp + '.txt')):
                continue

            img_list = glob(os.path.join(self.gt_dir, sp, '*.mat'))
            img_list = sorted([x.split('/')[-1].split('.')[-2] for x in img_list])

            split_f = os.path.join(self.root, 'lists', sp + '.txt')
            with open(split_f, 'w') as f:
                for img in img_list:
                    assert os.path.isfile(os.path.join(self.image_dir, sp, img + '.jpg'))
                    f.write('{}\n'.format(img))

    def _load_img(self, index):
        # Read Image
        _img = np.array(Image.open(self.images[index]).convert('RGB')).astype(np.float32)

        return _img

    def _load_edge(self, index):

        # Read Target object
        _gt_mat = sio.loadmat(self.gts[index])

        _target = np.zeros(_gt_mat['groundTruth'][0][0]['Boundaries'][0][0].shape)
        for i in range(len(_gt_mat['groundTruth'][0])):
            _target += _gt_mat['groundTruth'][0][i]['Boundaries'][0][0]

        if self.n_votes and self.n_votes > 0:
            _target = (_target >= self.n_votes).astype(np.float32)
        else:
            _target = (_target / max(1e-8, _target.max())).astype(np.float32)

        return _target

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
        return 'BSDS500(split=' + str(self.split) + ', n_votes=' + str(self.n_votes) + ')'


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    dataset = BSDS500()

    for i, sample in enumerate(dataset):
        plt.imshow(sample['image'] / 255.)
        plt.show()
        plt.imshow(sample['edge'])
        plt.show()

