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
import scipy.io as sio
from six.moves import urllib

from fblib.util.mypath import Path


class NYUD_MT(data.Dataset):
    """
    NYUD dataset for multi-task learning.
    Includes edge detection, semantic segmentation, surface normals, and depth prediction
    """

    URL = 'https://data.vision.ee.ethz.ch/kmaninis/share/MTL/NYUD_MT.tgz'
    FILE = 'NYUD_MT.tgz'

    def __init__(self,
                 root=Path.db_root_dir('NYUD_MT'),
                 download=True,
                 split='val',
                 transform=None,
                 retname=True,
                 overfit=False,
                 do_edge=True,
                 do_semseg=False,
                 do_normals=False,
                 do_depth=False,
                 ):

        self.root = root

        if download:
            self._download()

        self.transform = transform

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.retname = retname

        # Original Images
        self.im_ids = []
        self.images = []
        _image_dir = os.path.join(root, 'images')

        # Edge Detection
        self.do_edge = do_edge
        self.edges = []
        _edge_gt_dir = os.path.join(root, 'edge')

        # Semantic segmentation
        self.do_semseg = do_semseg
        self.semsegs = []
        _semseg_gt_dir = os.path.join(root, 'segmentation')

        # Surface Normals
        self.do_normals = do_normals
        self.normals = []
        _normal_gt_dir = os.path.join(root, 'normals')

        # Depth
        self.do_depth = do_depth
        self.depths = []
        _depth_gt_dir = os.path.join(root, 'depth')

        # train/val/test splits are pre-cut
        _splits_dir = os.path.join(root, 'gt_sets')

        print('Initializing dataloader for NYUD {} set'.format(''.join(self.split)))
        for splt in self.split:
            with open(os.path.join(os.path.join(_splits_dir, splt + '.txt')), 'r') as f:
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):

                # Images
                _image = os.path.join(_image_dir, line + '.jpg')
                assert os.path.isfile(_image)
                self.images.append(_image)
                self.im_ids.append(line.rstrip('\n'))

                # Edges
                _edge = os.path.join(self.root, _edge_gt_dir, line + '.png')
                assert os.path.isfile(_edge)
                self.edges.append(_edge)

                # Semantic Segmentation
                _semseg = os.path.join(self.root, _semseg_gt_dir, line + '.mat')
                assert os.path.isfile(_semseg)
                self.semsegs.append(_semseg)

                _normal = os.path.join(self.root, _normal_gt_dir, line + '.jpg')
                assert os.path.isfile(_normal)
                self.normals.append(_normal)

                _depth = os.path.join(self.root, _depth_gt_dir, line + '.mat')
                assert os.path.isfile(_depth)
                self.depths.append(_depth)

        if self.do_edge:
            assert (len(self.images) == len(self.edges))
        if self.do_semseg:
            assert (len(self.images) == len(self.semsegs))
        if self.do_normals:
            assert (len(self.images) == len(self.normals))
        if self.do_depth:
            assert (len(self.images) == len(self.depths))

        # Uncomment to overfit to one image
        if overfit:
            n_of = 64
            self.images = self.images[:n_of]
            self.im_ids = self.im_ids[:n_of]

        # Display stats
        print('Number of dataset images: {:d}'.format(len(self.images)))

    def __getitem__(self, index):
        # if index == 1102:
        #     print('hi')
        sample = {}

        _img = self._load_img(index)
        sample['image'] = _img

        if self.do_edge:
            _edge = self._load_edge(index)
            if _edge.shape != _img.shape[:2]:
                _edge = cv2.resize(_edge, _img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            sample['edge'] = _edge

        if self.do_semseg:
            _semseg = self._load_semseg(index)
            if _semseg.shape != _img.shape[:2]:
                _semseg = cv2.resize(_semseg, _img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            sample['semseg'] = _semseg

        if self.do_normals:
            _normals = self._load_normals(index)
            if _normals.shape[:2] != _img.shape[:2]:
                _normals = cv2.resize(_normals, _img.shape[:2][::-1], interpolation=cv2.INTER_CUBIC)
            sample['normals'] = _normals

        if self.do_depth:
            _depth = self._load_depth(index)
            if _depth.shape[:2] != _img.shape[:2]:
                _depth = cv2.resize(_depth, _img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            sample['depth'] = _depth

        if self.retname:
            sample['meta'] = {'image': str(self.im_ids[index]),
                              'im_size': (_img.shape[0], _img.shape[1])}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.images)

    def _load_img(self, index):
        _img = np.array(Image.open(self.images[index]).convert('RGB')).astype(np.float32)
        return _img

    def _load_edge(self, index):
        _edge = np.array(Image.open(self.edges[index])).astype(np.float32) / 255.
        return _edge

    def _load_semseg(self, index):
        # Note: Related works are ignoring the background class (40-way classification), such as:
        # _semseg = np.array(sio.loadmat(self.semsegs[index])['segmentation']).astype(np.float32) - 1
        # _semseg[_semseg == -1] = 255
        # However, all experiments of ASTMT were conducted by using 41-way classification:
        _semseg = np.array(sio.loadmat(self.semsegs[index])['segmentation']).astype(np.float32)
        return _semseg

    def _load_normals(self, index):
        _tmp = np.array(Image.open(self.normals[index])).astype(np.float32)
        _normals = 2.0 * _tmp / 255.0 - 1.0
        return _normals

    def _load_depth(self, index):
        _depth = np.array(sio.loadmat(self.depths[index])['depth']).astype(np.float32)
        return _depth

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
        return 'NYUD Multitask (split=' + str(self.split) + ')'


class NYUDRaw(data.Dataset):
    """
    NYUD dataset for Surface Normal and Depth Estimation using NYUD raw data.
    """
    def __init__(self,
                 root=Path.db_root_dir('NYUD_raw'),
                 split='train',
                 transform=None,
                 do_normals=True,
                 do_depth=False,
                 retname=True,
                 overfit=False,
                 ):

        self.root = root
        self.transform = transform

        self.split = split

        self.retname = retname

        self.do_normals = do_normals
        self.do_depth = do_depth

        # Original Images
        self.im_ids = []
        self.images = []
        _image_dir = os.path.join(root, self.split, 'images')
        _mask_gt_dir = os.path.join(root, self.split, 'masks')

        # Surface Normals
        self.normals = []
        nrm_ext = '.png' if self.split == 'train' else '.jpg'
        self.masks = []
        _normal_gt_dir = os.path.join(root, self.split, 'normals')

        # Monocular depth
        self.depths = []
        _depth_gt_dir = os.path.join(root, self.split, 'depth')

        # train/val/test splits are pre-cut
        _splits_dir = os.path.join(root, 'gt_sets')

        print('Initializing dataloader for NYUD Raw,  {} set'.format(self.split))
        with open(os.path.join(os.path.join(_splits_dir, self.split + '.txt')), 'r') as f:
            lines = f.read().splitlines()

        for ii, line in enumerate(lines):

            # Images
            _image = os.path.join(_image_dir, line + '.jpg')
            assert os.path.isfile(_image)
            self.images.append(_image)
            self.im_ids.append(line.rstrip('\n'))

            if self.do_normals:
                # Normals
                _normal = os.path.join(self.root, _normal_gt_dir, line + nrm_ext)
                assert os.path.isfile(_normal)
                self.normals.append(_normal)

            if self.do_depth:
                # Depth
                _depth = os.path.join(self.root, _depth_gt_dir, line + '.mat')
                assert os.path.isfile(_depth)
                self.depths.append(_depth)

            if self.split == 'train':
                # Masks (only available for train data)
                _mask = os.path.join(self.root, _mask_gt_dir, line + '.png')
                assert os.path.isfile(_mask)
                self.masks.append(_mask)

        if self.do_normals:
            assert(len(self.images) == len(self.normals))
        if self.do_depth:
            assert(len(self.images) == len(self.depths))

        if self.split == 'train':
            assert(len(self.images) == len(self.masks))

        # uncomment to overfit to one image
        if overfit:
            n_of = 64
            self.images = self.images[:n_of]
            self.im_ids = self.im_ids[:n_of]

        # display stats
        print('number of dataset images: {:d}'.format(len(self.images)))

    def __getitem__(self, index):
        sample = {}

        _img = self._load_img(index)
        sample['image'] = _img

        if self.do_normals:
            _normals = self._load_normals(index)
            sample['normals'] = _normals

        if self.do_depth:
            _depth = self._load_depth(index)
            sample['depth'] = _depth

        if self.retname:
            sample['meta'] = {'image': str(self.im_ids[index]),
                              'im_size': (_img.shape[0], _img.shape[1])}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def _load_img(self, index):
        _img = cv2.imread(self.images[index])[:, :, ::-1].astype(np.float32)
        return _img

    def _load_normals(self, index):
        _tmp = cv2.imread(self.normals[index])[:, :, ::-1].astype(np.float32)
        _normals = 2.0 * _tmp / 255.0 - 1.0

        if self.split == 'train':
            _mask = cv2.imread(self.masks[index], 0)
            _normals[_mask == 0, :] = 0

        return _normals

    def _load_depth(self, index):
        _depth = np.array(sio.loadmat(self.depths[index])['depth']).astype(np.float32)

        if self.split == 'train':
            _mask = cv2.imread(self.masks[index], 0)
            _depth[_mask == 0] = 0

        return _depth

    def __len__(self):
        return len(self.images)

    def __str__(self):
        return 'NYUD-v2 Raw,split=' + str(self.split) + ')'


def test_mt():
    transform = transforms.Compose([tr.RandomHorizontalFlip(),
                                    tr.ScaleNRotate(rots=(-90, 90), scales=(1., 1.),
                                                    flagvals={'image': cv2.INTER_CUBIC,
                                                              'edge': cv2.INTER_NEAREST,
                                                              'semseg': cv2.INTER_NEAREST,
                                                              'normals': cv2.INTER_LINEAR,
                                                              'depth': cv2.INTER_LINEAR}),
                                    tr.FixedResize(resolutions={'image': (512, 512),
                                                                'edge': (512, 512),
                                                                'semseg': (512, 512),
                                                                'normals': (512, 512),
                                                                'depth': (512, 512)},
                                                   flagvals={'image': cv2.INTER_CUBIC,
                                                             'edge': cv2.INTER_NEAREST,
                                                             'semseg': cv2.INTER_NEAREST,
                                                             'normals': cv2.INTER_LINEAR,
                                                             'depth': cv2.INTER_LINEAR}),
                                    tr.AddIgnoreRegions(),
                                    tr.ToTensor()])
    dataset = NYUD_MT(split='train', transform=transform, retname=True,
                      do_edge=True,
                      do_semseg=True,
                      do_normals=True,
                      do_depth=True)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=False, num_workers=5)

    for i, sample in enumerate(dataloader):
        imshow(sample['image'][0, 0])
        show()
        imshow(sample['edge'][0, 0])
        show()
        imshow(sample['semseg'][0, 0])
        show()
        imshow(sample['normals'][0, 0])
        show()
        imshow(sample['depth'][0, 0])
        show()


if __name__ == '__main__':
    from matplotlib.pyplot import imshow, show
    import torch
    import fblib.dataloaders.custom_transforms as tr
    from torchvision import transforms

    test_mt()


