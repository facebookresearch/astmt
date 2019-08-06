import os
from fblib.util.mypath import Path

import numpy as np
import torch.utils.data as data
from PIL import Image
import cv2


class VOC12(data.Dataset):

    URL = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
    FILE = "VOCtrainval_11-May-2012.tar"
    MD5 = '6cd6e144f989b92b3379bac3b3de84fd'
    BASE_DIR = 'VOCdevkit/VOC2012'

    VOC_CATEGORY_NAMES = ['background',
                          'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                          'bus', 'car', 'cat', 'chair', 'cow',
                          'diningtable', 'dog', 'horse', 'motorbike', 'person',
                          'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    def __init__(self,
                 root=Path.db_root_dir('PASCAL'),
                 split='val',
                 transform=None,
                 area_thres=0,
                 retname=True,
                 suppress_void_pixels=False,
                 do_semseg=True,
                 overfit=False,
                 ):

        self.root = root
        _voc_root = os.path.join(self.root, self.BASE_DIR)
        _inst_dir = os.path.join(_voc_root, 'SegmentationObject')
        _cat_dir = os.path.join(_voc_root, 'SegmentationClass')
        _image_dir = os.path.join(_voc_root, 'JPEGImages')
        self.transform = transform

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.area_thres = area_thres
        self.retname = retname
        self.suppress_void_pixels = suppress_void_pixels

        self.do_semseg = do_semseg
        if self.do_semseg:
            self.semsegs = []

        # train/val/test splits are pre-cut
        _splits_dir = os.path.join(_voc_root, 'ImageSets', 'Segmentation')

        self.im_ids = []
        self.images = []

        print("Initializing dataloader for PASCAL VOC12 {} set".format(''.join(self.split)))
        for splt in self.split:
            with open(os.path.join(os.path.join(_splits_dir, splt + '.txt')), "r") as f:
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):

                # Images
                _image = os.path.join(_image_dir, line + ".jpg")
                assert os.path.isfile(_image)
                self.images.append(_image)
                self.im_ids.append(line.rstrip('\n'))

                # Semantic Segmentation
                if self.do_semseg:
                    _semseg = os.path.join(_cat_dir, line + '.png')
                    assert os.path.isfile(_semseg)
                    self.semsegs.append(_semseg)

        if self.do_semseg:
            assert (len(self.images) == len(self.semsegs))

        # Uncomment to overfit to one image
        if overfit:
            n_of = 32
            self.im_ids = self.im_ids[:n_of]
            self.images = self.images[:n_of]
            if self.do_semseg:
                self.semsegs = self.semsegs[:n_of]

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
            if _semseg is not None:
                if _semseg.shape != _img.shape[:2]:
                    _semseg = cv2.resize(_semseg, _img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
                sample['semseg'] = _semseg

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

    def _load_semseg(self, index):
        _semseg = np.array(Image.open(self.semsegs[index])).astype(np.float32)
        if self.suppress_void_pixels:
            _semseg[_semseg == 255] = 0

        return _semseg

    def get_img_size(self, idx=0):
        img = Image.open(os.path.join(self.root, 'JPEGImages', self.img_list[idx] + '.jpg'))
        return list(reversed(img.size))

    def __str__(self):
        return 'VOC12(split=' + str(self.split) + ',area_thres=' + str(self.area_thres) + ')'


if __name__ == '__main__':
    from matplotlib.pyplot import imshow, show
    import torch
    import fblib.dataloaders.custom_transforms as tr
    from torchvision import transforms

    transform = transforms.Compose(
        [tr.RandomHorizontalFlip(), tr.FixedResizeWithMIL(resolutions=[609, 512]), tr.ToTensor()])

    dataset = VOC12(split=['train'], transform=transform, retname=True, do_semseg=True, suppress_void_pixels=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    for i, sample in enumerate(dataloader):
        # debug_bag_points(sample, viz=False)
        if 1:
            imshow(sample['semseg'].numpy().transpose(2, 3, 1, 0)[:, :, 0, 0]); show()
            print(sample['semseg'].unique())
        print(i)

