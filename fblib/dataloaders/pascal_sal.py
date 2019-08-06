import os
from fblib.util.mypath import Path

import numpy as np
import torch.utils.data as data
from PIL import Image
import cv2


class PASCALS(data.Dataset):

    def __init__(self,
                 root=Path.db_root_dir('PASCAL-S'),
                 transform=None,
                 retname=True,
                 overfit=False,
                 threshold=None,
                 ):

        self.root = root
        _image_dir = os.path.join(self.root, 'images')
        _sal_dir = os.path.join(self.root, 'masks')
        _split_dir = os.path.join(self.root, 'gt_sets')

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

    def __str__(self):
        return 'PASCAL-S()'


if __name__ == '__main__':
    from matplotlib.pyplot import imshow, show
    import torch
    import fblib.dataloaders.custom_transforms as tr
    from torchvision import transforms

    transform = transforms.Compose(
        [tr.RandomHorizontalFlip(), tr.ToTensor()])

    dataset = PASCALS(transform=transform, threshold=.5)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    for i, sample in enumerate(dataloader):
        imshow(sample['image'].numpy().transpose(2, 3, 1, 0)[:, :, :, 0] / 255.)
        show()
        imshow(sample['sal'].numpy().transpose(2, 3, 1, 0)[:, :, 0, 0])
        show()
