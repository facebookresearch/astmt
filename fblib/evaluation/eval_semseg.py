# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import warnings
import cv2
import os.path
import glob
import json
import numpy as np
from PIL import Image

VOC_CATEGORY_NAMES = ['background',
                      'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                      'bus', 'car', 'cat', 'chair', 'cow',
                      'diningtable', 'dog', 'horse', 'motorbike', 'person',
                      'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


NYU_CATEGORY_NAMES = ['background',
                      'wall', 'floor', 'cabinet', 'bed', 'chair',
                      'sofa', 'table', 'door', 'window', 'bookshelf',
                      'picture', 'counter', 'blinds', 'desk', 'shelves',
                      'curtain', 'dresser', 'pillow', 'mirror', 'floor mat',
                      'clothes', 'ceiling', 'books', 'refridgerator', 'television',
                      'paper', 'towel', 'shower curtain', 'box', 'whiteboard',
                      'person', 'night stand', 'toilet', 'sink', 'lamp',
                      'bathtub', 'bag', 'otherstructure', 'otherfurniture', 'otherprop']

FSV_CATEGORY_NAMES = ['background', 'vehicle', 'object']


def eval_semseg(loader, folder, n_classes=20, has_bg=True):

    n_classes = n_classes + int(has_bg)

    # Iterate
    tp = [0] * n_classes
    fp = [0] * n_classes
    fn = [0] * n_classes

    for i, sample in enumerate(loader):

        if i % 500 == 0:
            print('Evaluating: {} of {} objects'.format(i, len(loader)))

        # Load result
        filename = os.path.join(folder, sample['meta']['image'] + '.png')
        mask = np.array(Image.open(filename)).astype(np.float32)

        gt = sample['semseg']
        valid = (gt != 255)

        if mask.shape != gt.shape:
            warnings.warn('Prediction and ground truth have different size. Resizing Prediction..')
            mask = cv2.resize(mask, gt.shape[::-1], interpolation=cv2.INTER_NEAREST)

        # TP, FP, and FN evaluation
        for i_part in range(0, n_classes):
            tmp_gt = (gt == i_part)
            tmp_pred = (mask == i_part)
            tp[i_part] += np.sum(tmp_gt & tmp_pred & valid)
            fp[i_part] += np.sum(~tmp_gt & tmp_pred & valid)
            fn[i_part] += np.sum(tmp_gt & ~tmp_pred & valid)

    jac = [0] * n_classes
    for i_part in range(0, n_classes):
        jac[i_part] = float(tp[i_part]) / max(float(tp[i_part] + fp[i_part] + fn[i_part]), 1e-8)

    # Write results
    eval_result = dict()
    eval_result['jaccards_all_categs'] = jac
    eval_result['mIoU'] = np.mean(jac)

    return eval_result


def eval_and_store_semseg(database, save_dir, exp_name, overfit=False):

    # Dataloaders
    if database == 'VOC12':
        from fblib.dataloaders import pascal_voc as pascal
        n_classes = 20
        cat_names = VOC_CATEGORY_NAMES
        has_bg = True
        gt_set = 'val'
        db = pascal.VOC12(split=gt_set, do_semseg=True, overfit=overfit)
    elif database == 'PASCALContext':
        from fblib.dataloaders import pascal_context as pascal_context
        n_classes = 20
        cat_names = VOC_CATEGORY_NAMES
        has_bg = True
        gt_set = 'val'
        db = pascal_context.PASCALContext(split=gt_set, do_edge=False, do_human_parts=False, do_semseg=True,
                                          do_normals=False, overfit=overfit)
    elif database == 'NYUD':
        from fblib.dataloaders import nyud as nyud
        n_classes = 40
        cat_names = NYU_CATEGORY_NAMES
        has_bg = True
        gt_set = 'val'
        db = nyud.NYUD_MT(split=gt_set, do_semseg=True, overfit=overfit)
    elif database == 'FSV':
        from fblib.dataloaders import fsv as fsv
        n_classes = 2
        cat_names = FSV_CATEGORY_NAMES
        has_bg = True
        gt_set = 'test'
        db = fsv.FSVGTA(split=gt_set, mini=True, do_semseg=True, overfit=overfit)
    else:
        raise NotImplementedError

    res_dir = os.path.join(save_dir, exp_name, 'Results_' + database)
    base_name = database + '_' + gt_set + '_' + exp_name + '_semseg'
    fname = os.path.join(res_dir, base_name + '.json')

    # Check if already evaluated
    if os.path.isfile(fname):
        with open(fname, 'r') as f:
            eval_results = json.load(f)
    else:
        eval_results = eval_semseg(db, os.path.join(res_dir, 'semseg'), n_classes=n_classes, has_bg=has_bg)
        with open(fname, 'w') as f:
            json.dump(eval_results, f)

    # Print Results
    class_IoU = eval_results['jaccards_all_categs']
    mIoU = eval_results['mIoU']

    print('\nSemantic Segmentation mIoU: {0:.4f}\n'.format(100 * mIoU))
    for i in range(len(class_IoU)):
        spaces = ''
        for j in range(0, 15 - len(cat_names[i])):
            spaces += ' '
        print('{0:s}{1:s}{2:.4f}'.format(cat_names[i], spaces, 100 * class_IoU[i]))


def main():
    from fblib.util.mypath import Path
    database = 'PASCALContext'
    save_dir = os.path.join(Path.exp_dir(), 'pascal_se/edge_semseg_human_parts_normals_sal')

    # Evaluate all sub-folders
    exp_names = glob.glob(save_dir + '/*')
    exp_names = [x.split('/')[-1] for x in exp_names]
    for exp_name in exp_names:
        if os.path.isdir(os.path.join(save_dir, exp_name, 'Results_' + database, 'semseg')):
            print('Evaluating: {}'.format(exp_name))
            try:
                eval_and_store_semseg(database, save_dir, exp_name)
            except FileNotFoundError:
                print('Results of {} are not ready'.format(exp_name))


if __name__ == '__main__':
    main()
