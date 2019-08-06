import os

import cv2
import numpy as np
import torch.utils.data as data

from fblib.util.mypath import Path


class MSRA(data.Dataset):
    """
    MSRA10k dataset for Saliency Estimation
    """
    def __init__(self,
                 root=Path.db_root_dir('MSRA10K'),
                 split='trainval',
                 transform=None,
                 retname=True,
                 overfit=False):

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
        sample['sal'] =_sal

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

    def __str__(self):
        return 'MSRA(split=' + str(self.split) + ')'


if __name__ == '__main__':
    import torch
    from matplotlib import pyplot as plt
    import fblib.dataloaders.custom_transforms as tr
    from torchvision import transforms

    transform = transforms.Compose([tr.RandomHorizontalFlip(),
                                    tr.FixedResize(resolutions={'image': (512, 512), 'sal': (512, 512)},
                                                   flagvals={'image': cv2.INTER_CUBIC, 'sal': cv2.INTER_NEAREST}),
                                    tr.ToTensor()])

    dataset = MSRA(transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, num_workers=5)

    for i, sample in enumerate(dataloader):
        print(i)
        plt.imshow(sample['image'][0, 0, :, :]/255)
        plt.show()
        plt.imshow(sample['sal'][0, 0, :, :])
        plt.show()



