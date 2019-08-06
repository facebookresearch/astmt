import errno
import json
import os
import sys
import tarfile
from fblib.util.mypath import Path

import numpy as np
import torch.utils.data as data
from PIL import Image
import cv2
from six.moves import urllib
import scipy.io as sio
from skimage.morphology import thin
from fblib.external.detail_api.PythonAPI.detail import Detail


class PASCALInDetail(data.Dataset):

    URL = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
    FILE = "VOCtrainval_11-May-2012.tar"
    MD5 = '6cd6e144f989b92b3379bac3b3de84fd'
    BASE_DIR = 'VOCdevkit/VOC2012'

    HUMAN_PART = {1: {'hair': 1, 'head': 1, 'lear': 1, 'lebrow': 1, 'leye': 1, 'lfoot': 1,
                        'lhand': 1, 'llarm': 1, 'llleg': 1, 'luarm': 1, 'luleg': 1, 'mouth': 1,
                        'neck': 1, 'nose': 1, 'rear': 1, 'rebrow': 1, 'reye': 1, 'rfoot': 1,
                        'rhand': 1, 'rlarm': 1, 'rlleg': 1, 'ruarm': 1, 'ruleg': 1, 'torso': 1},
                  4: {'hair': 1, 'head': 1, 'lear': 1, 'lebrow': 1, 'leye': 1, 'lfoot': 4,
                        'lhand': 3, 'llarm': 3, 'llleg': 4, 'luarm': 3, 'luleg': 4, 'mouth': 1,
                        'neck': 2, 'nose': 1, 'rear': 1, 'rebrow': 1, 'reye': 1, 'rfoot': 4,
                        'rhand': 3, 'rlarm': 3, 'rlleg': 4, 'ruarm': 3, 'ruleg': 4, 'torso': 2},
                  6: {'hair': 1, 'head': 1, 'lear': 1, 'lebrow': 1, 'leye': 1, 'lfoot': 6,
                        'lhand': 4, 'llarm': 4, 'llleg': 6, 'luarm': 3, 'luleg': 5, 'mouth': 1,
                        'neck': 2, 'nose': 1, 'rear': 1, 'rebrow': 1, 'reye': 1, 'rfoot': 6,
                        'rhand': 4, 'rlarm': 4, 'rlleg': 6, 'ruarm': 3, 'ruleg': 5, 'torso': 2},
                  14: {'hair': 1, 'head': 1, 'lear': 1, 'lebrow': 1, 'leye': 1, 'lfoot': 14,
                         'lhand': 8, 'llarm': 7, 'llleg': 13, 'luarm': 6, 'luleg': 12, 'mouth': 1,
                         'neck': 2, 'nose': 1, 'rear': 1, 'rebrow': 1, 'reye': 1, 'rfoot': 11,
                         'rhand': 5, 'rlarm': 4, 'rlleg': 10, 'ruarm': 3, 'ruleg': 9, 'torso': 2}
                  }

    VOC_CATEGORY_NAMES = ['background',
                          'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                          'bus', 'car', 'cat', 'chair', 'cow',
                          'diningtable', 'dog', 'horse', 'motorbike', 'person',
                          'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    CONTEXT_CATEGORY_LABELS = [0,
                               2, 23, 25, 31, 34,
                               45, 59, 65, 72, 98,
                               397, 113, 207, 258, 284,
                               308, 347, 368, 416, 427]

    def __init__(self,
                 root=Path.db_root_dir('PASCAL'),
                 split=['val'],
                 transform=None,
                 area_thres=0,
                 retname=True,
                 suppress_void_pixels=True,
                 overfit=False,
                 do_edge=True,
                 use_mil=False,
                 do_human_part=False,
                 do_semseg=False,
                 pascal_classes=True,
                 num_human_part=6
                 ):

        self.root = root
        _voc_root = os.path.join(self.root, self.BASE_DIR)
        _json_file = os.path.join(self.root, 'json', 'trainval_withkeypoints.json')
        _image_dir = os.path.join(_voc_root, 'JPEGImages')
        self.transform = transform
        self.pascal_classes = pascal_classes

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.db = Detail(annotation_file=_json_file, image_folder=_image_dir)
        data = self.db.getImgs(phase=''.join(self.split))
        self.im_ids = [x['file_name'].split('.')[0] for x in data]

        self.area_thres = area_thres
        self.retname = retname
        self.suppress_void_pixels = suppress_void_pixels

        # Different Tasks

        # Edge Detection
        self.do_edge = do_edge
        self.use_mil = use_mil
        if self.do_edge:
            if self.use_mil:
                edge_gt_dir, edge_suffix = Path.edge_params('PASCAL-mil')
                self.edges = []

        # Human Part Segmentation
        self.do_human_part = do_human_part
        if self.do_human_part:
            self.human_parts_category = 15
            self.parts = []
            _part_gt_dir = Path.parts_params('PASCAL')
            self.cat_part = json.load(open(os.path.join(os.path.dirname(__file__), '../util/pascal_part.json'), 'r'))
            self.cat_part["15"] = self.HUMAN_PART[num_human_part]
            self.parts_file = os.path.join(os.path.join(self.root, 'PascalParts'),
                                           ''.join(self.split) + '_part_instances.txt')

        self.do_semseg = do_semseg

        # train/val/test splits are pre-cut
        _splits_dir = os.path.join(_voc_root, 'ImageSets', 'Context')

        self.images = []

        print("Initializing dataloader for PASCAL in Detail {} set".format(''.join(self.split)))

        for ii, line in enumerate(self.im_ids):

            # Images
            _image = os.path.join(_image_dir, line + ".jpg")
            assert(os.path.isfile(_image))
            self.images.append(_image)

            # Edges
            if self.do_edge and self.use_mil:
                _edge = os.path.join(self.root, edge_gt_dir, line + edge_suffix + ".mat")
                assert os.path.isfile(_edge)
                self.edges.append(_edge)

            # Human Parts
            if self.do_human_part:
                _human_part = os.path.join(self.root, _part_gt_dir, line + ".mat")
                assert os.path.isfile(_human_part)
                self.parts.append(_human_part)

        if self.do_human_part:
            assert (len(self.images) == len(self.parts))

        if self.do_human_part:
            if not self._check_preprocess_parts():
                print('Pre-processing PASCAL dataset for human parts, this will take long, but will be done only once.')
                self._preprocess_parts()

            # Find images which have human parts
            self.has_human_parts = []
            for ii in range(len(self.im_ids)):
                if self.human_parts_category in self.part_obj_dict[self.im_ids[ii]]:
                    self.has_human_parts.append(1)
                else:
                    self.has_human_parts.append(0)

            print('Number of images with human parts: {:d}'.format(np.sum(self.has_human_parts)))

        # Uncomment to overfit to one image
        if overfit:
            n_of = 5
            self.images = self.images[:n_of]
            self.im_ids = self.im_ids[:n_of]
            if self.do_edge:
                self.edges = self.edges[:n_of]
            if self.do_human_part:
                self.parts = self.parts[:n_of]

        # Display stats
        print('Number of dataset images: {:d}'.format(len(self.images)))

    def __getitem__(self, index):
        # if index == 1102:
        #     print('hi')
        sample = {}

        _img = self._load_img(index)
        sample['image'] = _img

        if self.do_edge:
            if self.use_mil:
                _edge, idxh, idxv, _ = self._load_edge_mil(index)
                if _edge is not None:
                    sample['edge'] = _edge
                    sample['idxh'] = idxh
                    sample['idxv'] = idxv
                else:
                    print('No edges for idx: {}. Caught in dataloader'.format(index))

                if _edge.shape != _img.shape[:2]:
                    sample = self._process_edge_mil(sample)
            else:
                _edge = self.db.getBounds(img=self.im_ids[index])
                if _edge.shape != _img.shape[:2]:
                    _edge = cv2.resize(_edge, _img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
                sample['edge'] = _edge

        if self.do_human_part:
            _human_parts, _ = self._load_human_parts(index)
            if _human_parts is not None:
                if _human_parts.shape != _img.shape[:2]:
                    _human_parts = cv2.resize(_human_parts, _img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
                sample['human_parts'] = _human_parts

        if self.do_semseg:
            _semseg = self.db.getMask(img=self.im_ids[index])

            if self.pascal_classes:
                _semseg = self._remap_to_pascal(_semseg)

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

    def _load_edge_mil(self, index):

        try:
            _gt_mat = sio.loadmat(self.edges[index])
        except ValueError:
            print("Something is wrong with file: {}".format(self.edges[index]))
            return [], [], [], []

        pos = _gt_mat['pos'].astype(np.float)
        idxh, idxv = _gt_mat['crsh'], _gt_mat['crsv']

        if idxh.shape[0] > 0:
            if 'clss' in _gt_mat:
                clss = _gt_mat['clss']
            else:
                clss = 0 * idxh
        else:
            pos = None  # np.zeros(pos.shape[:2])
            clss = None  # np.zeros((1, 1))

        return pos, idxh, idxv, clss

    def _load_human_parts(self, index):
        if self.has_human_parts[index]:

            # Read Target object
            _part_mat = sio.loadmat(self.parts[index])['anno'][0][0][1][0]

            _inst_mask = _target = None

            for _obj_ii in range(len(_part_mat)):

                has_human = _part_mat[_obj_ii][1][0][0] == self.human_parts_category
                has_parts = len(_part_mat[_obj_ii][3]) != 0

                if has_human and has_parts:
                    if _inst_mask is None:
                        _inst_mask = _part_mat[_obj_ii][2].astype(np.float32)
                        _target = np.zeros(_inst_mask.shape)
                    else:
                        _inst_mask = np.maximum(_inst_mask, _part_mat[_obj_ii][2].astype(np.float32))

                    n_parts = len(_part_mat[_obj_ii][3][0])
                    for part_i in range(n_parts):
                        cat_part = str(_part_mat[_obj_ii][3][0][part_i][0][0])
                        mask_id = self.cat_part[str(self.human_parts_category)][cat_part]
                        mask = _part_mat[_obj_ii][3][0][part_i][1].astype(bool)
                        _target[mask] = mask_id

            if _target is not None:
                _target, _inst_mask = _target.astype(np.float32), _inst_mask.astype(np.float32)

            return _target, _inst_mask

        else:
            return None, None

    def _remap_to_pascal(self, semseg):
        result = np.zeros(semseg.shape, dtype=semseg.dtype)
        for i in range(len(self.CONTEXT_CATEGORY_LABELS)):
            if i == 0:
                continue
            tmp = (semseg == self.CONTEXT_CATEGORY_LABELS[i])
            result[tmp] = i

        return result

    def get_img_size(self, idx=0):
        img = Image.open(os.path.join(self.root, 'JPEGImages', self.img_list[idx] + '.jpg'))
        return list(reversed(img.size))

    def _check_preprocess_parts(self):
        if not os.path.isfile( self.parts_file):
            return False
        else:
            self.part_obj_dict = json.load(open( self.parts_file, 'r'))

            return list(np.sort([str(x) for x in self.part_obj_dict.keys()])) == list(np.sort(self.im_ids))

    def _process_edge_mil(self, sample):

        d_size = sample['image'].shape[:2]
        gt_shape = sample['edge'].shape
        idxh_ = sample['idxh'].astype(np.float) - 1
        idxv_ = sample['idxv'].astype(np.float) - 1

        sample['edge'] = cv2.resize(sample['edge'], dsize=d_size[::-1], interpolation=cv2.INTER_NEAREST)

        scaling = (float(d_size[0]) / float(gt_shape[0]), float(d_size[1]) / float(gt_shape[1]))
        idxh = np.clip(np.around(idxh_ * scaling[1]), 0, d_size[1] - 1).astype(np.float) + 1
        idxv = np.clip(np.around(idxv_ * scaling[0]), 0, d_size[0] - 1).astype(np.float) + 1

        abovz = (idxv_ >= 0) & (idxh_ >= 0) & (idxh > 0) & (idxv > 0)
        sample['idxh'] = idxh * abovz.astype(np.float)
        sample['idxv'] = idxv * abovz.astype(np.float)

        return sample

    # clss = np.zeros((1, 1))
    def _preprocess_parts(self):
        self.part_obj_dict = {}
        obj_counter = 0
        for ii in range(len(self.im_ids)):
            # Read object masks and get number of objects
            if ii % 100 == 0:
                print("Processing image: {}".format(ii))
            part_mat = sio.loadmat(os.path.join(self.root, 'PascalParts/Annotations_Part/{}.mat'.format(self.im_ids[ii])))
            n_obj = len(part_mat['anno'][0][0][1][0])

            # Get the categories from these objects
            _cat_ids = []
            for jj in range(n_obj):
                obj_area = np.sum(part_mat['anno'][0][0][1][0][jj][2])
                obj_cat = int(part_mat['anno'][0][0][1][0][jj][1])
                if obj_area > self.area_thres:
                    _cat_ids.append(int(part_mat['anno'][0][0][1][0][jj][1]))
                else:
                    _cat_ids.append(-1)
                obj_counter += 1

            self.part_obj_dict[self.im_ids[ii]] = _cat_ids

        with open(self.parts_file, 'w') as outfile:
            outfile.write('{{\n\t"{:s}": {:s}'.format(self.im_ids[0], json.dumps(self.part_obj_dict[self.im_ids[0]])))
            for ii in range(1, len(self.im_ids)):
                outfile.write(',\n\t"{:s}": {:s}'.format(self.im_ids[ii], json.dumps(self.part_obj_dict[self.im_ids[ii]])))
            outfile.write('\n}\n')

        print('Preprocessing for parts finished')

    def __str__(self):
        return 'PASCALInDetail(split=' + str(self.split) + ',area_thres=' + str(self.area_thres) + ')'


def debug_bag_points(sample, viz):
    pos = sample['edge'].numpy()[0, 0, :]
    idxs = sample['edgeidx'].numpy()[0, 0, :]

    maxdist = 0

    for ii in range(0, idxs.shape[0]):
        if not idxs[ii,:].any():
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
            print(idxs[ii,:])
            print(pos.ravel()[idxs[ii,:].astype(np.int)])
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
    import matplotlib.pyplot as plt
    import torch
    import fblib.dataloaders.custom_transforms as tr
    from torchvision import transforms
    from fblib.util.helpers import ind2sub

    transform = transforms.Compose(
        [tr.RandomHorizontalFlip(), tr.FixedResizeWithMIL(resolutions=[600, ]), tr.ToTensor()])

    dataset = PASCALInDetail(split=['train', 'val'], transform=transform, retname=True, use_mil=True,
                     do_edge=True, do_human_part=True, do_semseg=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)


    for i, sample in enumerate(dataloader):
        print(i)
        if i < 0:
            continue
        debug_bag_points(sample, viz=False)
        if 'human_parts' in sample and 'semseg' in sample:
            plt.imshow(sample['human_parts'].numpy()[0, 0, :]); plt.show()
            plt.imshow(sample['edge'].numpy()[0, 0, :]); plt.show()
            plt.imshow(sample['semseg'].numpy()[0, 0, :]); plt.show()
            print(sample['semseg'].unique())

