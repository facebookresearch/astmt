# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
from fblib.util.mypath import Path

import numpy as np
import torch.utils.data as data
import cv2


class FSVGTA(data.Dataset):

    def __init__(self,
                 root=Path.db_root_dir('FSV'),
                 split='test',
                 mini=True,
                 transform=None,
                 retname=True,
                 overfit=False,
                 do_semseg=False,
                 do_albedo=False,
                 do_depth=False,
                 prune_rare_classes=True,
                 ):

        self.root = root
        self.transform = transform
        self.prune = []
        if prune_rare_classes:
            self.prune = [1, 4, 5, 6, 7]

        self.split = split

        self.retname = retname

        # Original Images
        self.im_ids = []
        self.images = []
        _image_dir = os.path.join(root, 'gta_' + split)

        # Semantic segmentation
        self.do_semseg = do_semseg
        self.semsegs = []

        # Albedo
        self.do_albedo = do_albedo
        self.albedos = []

        # Depth
        self.do_depth = do_depth
        self.depths = []

        # train/val/test splits are pre-cut
        _splits_dir = os.path.join(root, 'gt_sets')

        print("Initializing dataloader for FSV GTA {} set".format(self.split))
        with open(os.path.join(os.path.join(_splits_dir, 'gta_' + self.split + '.txt')), "r") as f:
            lines = f.read().splitlines()

        if split == 'test' and mini:
            lines = lines[0:len(lines):int(len(lines)/5000)]

        for ii, line in enumerate(lines):

            # Images
            _image = os.path.join(_image_dir, line + "_final.webp")
            # assert os.path.isfile(_image)
            self.images.append(_image)
            self.im_ids.append(line.rstrip('\n'))

            # Semantic Segmentation
            _semseg = os.path.join(_image_dir, line + "_object_id.png")
            # assert os.path.isfile(_semseg)
            self.semsegs.append(_semseg)

            # Albedo
            _albedo = os.path.join(_image_dir, line + "_albedo.webp")
            # assert os.path.isfile(_albedo)
            self.albedos.append(_albedo)

            # Depth Estimation
            _depth = os.path.join(_image_dir, line + "_disparity.webp")
            # assert os.path.isfile(_depth)
            self.depths.append(_depth)

        if self.do_semseg:
            assert (len(self.images) == len(self.semsegs))
        if self.do_albedo:
            assert (len(self.images) == len(self.albedos))
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

        if self.do_semseg:
            _semseg = self._load_semseg(index)
            if _semseg.shape != _img.shape[:2]:
                _semseg = cv2.resize(_semseg, _img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            sample['semseg'] = _semseg

        if self.do_albedo:
            _albedo = self._load_albedo(index)
            if _albedo.shape[:2] != _img.shape[:2]:
                _depth = cv2.resize(_albedo, _img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            sample['albedo'] = _albedo

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
        _img = cv2.imread(self.images[index])[:, :, ::-1].astype(np.float32)
        return _img

    def _load_semseg(self, index):
        _semseg = cv2.imread(self.semsegs[index])[:, :, -1].astype(np.float32)

        # Prune rare classes
        if self.prune:
            uniq = np.unique(_semseg)
            for cls in self.prune:
                if cls in uniq:
                    _semseg[_semseg == cls] = 0
            _semseg = np.maximum(_semseg - 1, 0)
        return _semseg

    def _load_albedo(self, index):
        _albedo = cv2.imread(self.albedos[index])[:, :, ::-1].astype(np.float32) / 255.
        return _albedo

    def _load_depth(self, index):
        _depth = cv2.imread(self.depths[index])
        _depth = (_depth[:, :, 0] * 256 * 256 + _depth[:, :, 1] * 256 + _depth[:, :, 2]).astype(np.float32) / 8192
        return _depth

    def __str__(self):
        return 'FSV GTA Multitask (split=' + str(self.split) + ')'
