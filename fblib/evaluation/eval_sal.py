# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import warnings
import cv2
import os.path
import numpy as np
import glob
import json
from PIL import Image

import fblib.evaluation.jaccard as evaluation


def eval_sal(loader, folder, mask_thres=None):
    if mask_thres is None:
        mask_thres = [0.5]

    eval_result = dict()
    eval_result['all_jaccards'] = np.zeros((len(loader), len(mask_thres)))
    eval_result['prec'] = np.zeros((len(loader), len(mask_thres)))
    eval_result['rec'] = np.zeros((len(loader), len(mask_thres)))

    for i, sample in enumerate(loader):

        if i % 500 == 0:
            print('Evaluating: {} of {} objects'.format(i, len(loader)))

        # Load result
        filename = os.path.join(folder, sample["meta"]["image"] + '.png')
        mask = np.array(Image.open(filename)).astype(np.float32) / 255.

        gt = sample["sal"]

        if mask.shape != gt.shape:
            warnings.warn('Prediction and ground truth have different size. Resizing Prediction..')
            mask = cv2.resize(mask, gt.shape[::-1], interpolation=cv2.INTER_NEAREST)

        for j, thres in enumerate(mask_thres):
            gt = (gt > thres).astype(np.float32)
            mask_eval = (mask > thres).astype(np.float32)
            eval_result['all_jaccards'][i, j] = evaluation.jaccard(gt, mask_eval)
            eval_result['prec'][i, j], eval_result['rec'][i, j] = evaluation.precision_recall(gt, mask_eval)

    # Average for each thresholds
    eval_result['mIoUs'] = np.mean(eval_result['all_jaccards'], 0)

    eval_result['mPrec'] = np.mean(eval_result['prec'], 0)
    eval_result['mRec'] = np.mean(eval_result['rec'], 0)
    eval_result['F'] = 2 * eval_result['mPrec'] * eval_result['mRec'] / \
                       (eval_result['mPrec'] + eval_result['mRec'] + 1e-12)

    # Maximum of averages (maxF, maxmIoU)
    eval_result['mIoU'] = np.max(eval_result['mIoUs'])
    eval_result['maxF'] = np.max(eval_result['F'])

    eval_result = {x: eval_result[x].tolist() for x in eval_result}

    return eval_result


def eval_and_store_sal(database, save_dir, exp_name, overfit=False):

    # Dataloaders
    if database == 'PASCALContext':
        from fblib.dataloaders import pascal_context as pascal_context
        split = 'val'
        db = pascal_context.PASCALContext(split=split, do_edge=False, do_human_parts=False, do_semseg=False,
                                          do_normals=False, do_sal=True, overfit=overfit)
    elif database == 'PASCAL-S':
        from fblib.dataloaders import pascal_sal as pascal_sal
        split = 'all'
        db = pascal_sal.PASCALS(overfit=overfit, threshold=None)
    else:
        raise NotImplementedError

    res_dir = os.path.join(save_dir, exp_name, 'Results_' + database)
    base_name = database + '_' + split + '_' + exp_name + '_sal'
    fname = os.path.join(res_dir, base_name + '.json')

    # Check if already evaluated
    if os.path.isfile(fname):
        with open(fname, 'r') as f:
            eval_results = json.load(f)
    else:
        eval_results = eval_sal(db, os.path.join(res_dir, 'sal'), mask_thres=np.linspace(0.2, 0.9, 15))
        with open(fname, 'w') as f:
            json.dump(eval_results, f)

    print('Results for Saliency Estimation')
    print('mIoU: {0:.3f}'.format(eval_results['mIoU']))
    print('maxF: {0:.3f}'.format(eval_results['maxF']))


def main():
    from fblib.util.mypath import Path
    database = 'PASCALContext'
    save_dir = os.path.join(Path.exp_dir(), 'pascal_se/edge_semseg_human_parts_normals_sal')

    # Evaluate all sub-folders
    exp_names = glob.glob(save_dir + '/*')
    exp_names = [x.split('/')[-1] for x in exp_names]
    for exp_name in exp_names:
        if os.path.isdir(os.path.join(save_dir, exp_name, 'Results_' + database, 'sal')):
            print('Evaluating: {}'.format(exp_name))
            try:
                eval_and_store_sal(database, save_dir, exp_name, overfit=False)
            except FileNotFoundError:
                print('Results of {} are not ready'.format(exp_name))


if __name__ == '__main__':
    main()
