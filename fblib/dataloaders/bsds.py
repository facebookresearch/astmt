import os

import cv2
import numpy as np
from glob import glob
import scipy.io as sio
from PIL import Image
import torch.utils.data as data

from fblib.util.mypath import Path


class BSDS500(data.Dataset):
    """
    BSDS500 datasets for edge detection.
    """
    def __init__(self,
                 root=Path.db_root_dir('BSDS500'),
                 split=['train', 'val'],
                 custom_data=False,
                 use_mil=False,
                 transform=None,
                 retname=True,
                 n_votes=1,
                 overfit=False):

        self.transform = transform
        self.custom_data = custom_data
        self.use_mil = use_mil

        self.retname = retname
        self.n_votes = n_votes

        if self.custom_data:
            print('Initializing BSDS500 with custom data')
            self.root = Path.db_root_dir('BSDS500-mil')
            self.gt_dir = os.path.join(self.root, 'gt_sz_2.0_nmn_3_rsz_18_3S')
            self.image_dir = os.path.join(self.root, 'im_rsz_18_3S')
            self.split = "trainval_step_10_nsteps_8_rsz_18_3S"
            _splits_dir = os.path.join(self.root, 'ImageSets')
        else:
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

                if not self.custom_data:
                    _image = os.path.join(self.image_dir, sp, line + ".jpg")
                    _gt = os.path.join(self.gt_dir, sp, line + ".mat")
                else:
                    _image = os.path.join(self.image_dir, line + ".jpg")
                    _gt = os.path.join(self.gt_dir, line + ".mat")

                assert os.path.isfile(_image)
                assert os.path.isfile(_gt)
                self.im_ids.append(line)
                self.images.append(_image)
                self.gts.append(_gt)

        assert (len(self.images) == len(self.gts) == len(self.im_ids))

        if overfit:
            self.images = self.images[:14]
            self.gts = self.gts[:14]
            self.im_ids = self.im_ids[:14]

        # Display stats
        print('Number of images: {:d}'.format(len(self.im_ids)))

    def __getitem__(self, index):

        sample = {}

        _img = self._load_img(index)
        sample['image'] = _img

        if not self.custom_data:
            _edge = self._load_edge(index)
            sample['edge'] = _edge
        else:
            _edge, idxh, idxv, _ = self._load_edge_custom(index)
            sample['edge'] = _edge
            if self.use_mil:
                sample['idxh'] = idxh
                sample['idxv'] = idxv

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

    def _load_edge_custom(self, index):

        # Read Target ground truth (contains MIL)
        try:
            _gt_mat = sio.loadmat(self.gts[index])
        except ValueError('Something is wrong with .mat file: {}.'.format(self.gts[index])):
            return [None] * 4

        if self.use_mil:
            edge = _gt_mat['pos'].astype(np.float)
            idxh, idxv = _gt_mat['crsh'], _gt_mat['crsv']

            if idxh.shape[0] > 0:
                if 'clss' in _gt_mat:
                    clss = _gt_mat['clss']
                else:
                    clss = 0 * idxh
            else:
                clss = None
        else:
            edge = _gt_mat['edge'].astype(np.float32)
            idxh = idxv = clss = None

        return edge, idxh, idxv, clss

    def __str__(self):
        return 'BSDS500(split=' + str(self.split) + ', n_votes=' + str(self.n_votes) + ')'


def debug_bag_points(sample, viz):
    pos = sample['edge'].numpy()[0, 0, :]
    idxs = sample['edgeidx'][0][0,:].numpy()

    maxdist = 0

    for ii in range(0, idxs.shape[0]):
        if not idxs[ii, :].any():
            print("All zeros: {}".format(ii))
            continue

        rows, cols = ind2sub(pos.shape, idxs[ii, :])
        rows = np.array(rows)
        cols = np.array(cols)
        r_tmp, c_tmp = np.append(rows[1:], rows[0]), np.append(cols[1:], cols[0])
        dists = ((rows - r_tmp) ** 2 + (cols - c_tmp) ** 2) ** 0.5
        maxdist = max(dists.max(), maxdist)

        if dists.max() > 15:
            print('Gotcha! maxdist: {}'.format(dists.max()))
            print(idxs[ii, :])
            print(pos.ravel()[idxs[ii, :].astype(np.int)])
            if viz:
                plt.figure()
                plt.imshow(pos)
                plt.plot(rows,  cols, 'r+')
                plt.show()

        if (pos[cols, rows] != 1).all():
            print('pos is not 1 at index!')
            if viz:
                plt.figure()
                plt.imshow(pos)
                plt.plot(rows, cols, 'r+')
                plt.show()

    print("Maximum distance found: {} ".format(maxdist))


if __name__ == '__main__':
    import torch
    from matplotlib import pyplot as plt
    from fblib.util.helpers import ind2sub
    import fblib.dataloaders.custom_transforms as tr
    from torchvision import transforms
    from fblib.util.custom_collate import collate_mil

    transform = transforms.Compose([tr.RandomHorizontalFlip(),
                                    tr.FixedResize(resolutions={'image': (512, 512), 'edge': (512, 512)},
                                                   flagvals={'image': cv2.INTER_CUBIC, 'edge': cv2.INTER_NEAREST},
                                                   use_mil=True, mildil=False),
                                    tr.ToTensor()])

    dataset = BSDS500(custom_data=True, transform=transform, use_mil=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False,
                                             num_workers=5, collate_fn=collate_mil)

    for i, sample in enumerate(dataloader):
        assert(sample['edge'].size()[2:] == sample['image'].size()[2:])
        if 'edgeidx' in sample:
            debug_bag_points(sample, viz=False)


