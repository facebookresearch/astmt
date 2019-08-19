# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import warnings
import cv2
import glob
import json
import os.path
import numpy as np
from PIL import Image

PART_CATEGORY_NAMES = ['background',
                       'head', 'torso', 'uarm', 'larm', 'uleg', 'lleg']


def eval_human_parts(loader, folder, n_parts=6):

    tp = [0] * (n_parts + 1)
    fp = [0] * (n_parts + 1)
    fn = [0] * (n_parts + 1)

    counter = 0
    for i, sample in enumerate(loader):

        if i % 500 == 0:
            print('Evaluating: {} of {} objects'.format(i, len(loader)))

        if 'human_parts' not in sample:
            continue

        # Check for valid pixels
        gt = sample['human_parts']
        uniq = np.unique(gt)
        if len(uniq) == 1 and (uniq[0] == 255 or uniq[0] == 0):
            continue

        # Load result
        filename = os.path.join(folder, sample['meta']['image'] + '.png')
        mask = np.array(Image.open(filename)).astype(np.float32)

        # Case of a binary (probability) result
        if n_parts == 1:
            mask = (mask > 0.5 * 255).astype(np.float32)

        counter += 1
        valid = (gt != 255)

        if mask.shape != gt.shape:
            warnings.warn('Prediction and ground truth have different size. Resizing Prediction..')
            mask = cv2.resize(mask, gt.shape[::-1], interpolation=cv2.INTER_NEAREST)

        # TP, FP, and FN evaluation
        for i_part in range(0, n_parts + 1):
            tmp_gt = (gt == i_part)
            tmp_pred = (mask == i_part)
            tp[i_part] += np.sum(tmp_gt & tmp_pred & (valid))
            fp[i_part] += np.sum(~tmp_gt & tmp_pred & (valid))
            fn[i_part] += np.sum(tmp_gt & ~tmp_pred & (valid))

    print('Successful evaluation for {} images'.format(counter))
    jac = [0] * (n_parts + 1)
    for i_part in range(0, n_parts + 1):
        jac[i_part] = float(tp[i_part]) / max(float(tp[i_part] + fp[i_part] + fn[i_part]), 1e-8)

    # Write results
    eval_result = dict()
    eval_result['jaccards_all_categs'] = jac
    eval_result['mIoU'] = np.mean(jac)

    return eval_result


def eval_and_store_human_parts(database, save_dir, exp_name, overfit=False):

    # Dataloaders
    if database == 'PASCALContext':
        from fblib.dataloaders import pascal_context as pascal_context
        gt_set = 'val'
        db = pascal_context.PASCALContext(split=gt_set, do_edge=False, do_human_parts=True, do_semseg=False,
                                          do_normals=False, do_sal=False, overfit=overfit)
    else:
        raise NotImplementedError

    res_dir = os.path.join(save_dir, exp_name, 'Results_' + database)
    base_name = database + '_' + gt_set + '_' + exp_name + '_human_parts'
    fname = os.path.join(res_dir, base_name + '.json')

    # Check if already evaluated
    if os.path.isfile(fname):
        with open(fname, 'r') as f:
            eval_results = json.load(f)
    else:
        eval_results = eval_human_parts(db, os.path.join(res_dir, 'human_parts'))
        with open(fname, 'w') as f:
            json.dump(eval_results, f)

    # Print Results
    class_IoU = eval_results['jaccards_all_categs']
    mIoU = eval_results['mIoU']

    print('\nHuman Parts mIoU: {0:.4f}\n'.format(100 * mIoU))
    for i in range(len(class_IoU)):
        spaces = ''
        for j in range(0, 15 - len(PART_CATEGORY_NAMES[i])):
            spaces += ' '
        print('{0:s}{1:s}{2:.4f}'.format(PART_CATEGORY_NAMES[i], spaces, 100 * class_IoU[i]))


def main():
    from fblib.util.mypath import Path
    database = 'PASCALContext'
    save_dir = os.path.join(Path.exp_dir(), 'pascal_se/edge_semseg_human_parts_normals_sal')

    # Evaluate all sub-folders
    exp_names = glob.glob(save_dir + '/*')
    exp_names = [x.split('/')[-1] for x in exp_names]
    for exp_name in exp_names:
        if os.path.isdir(os.path.join(save_dir, exp_name, 'Results_' + database, 'human_parts')):
            print('Evaluating: {}'.format(exp_name))
            try:
                eval_and_store_human_parts(database, save_dir, exp_name)
            except FileNotFoundError:
                print('Results of {} are not ready'.format(exp_name))


if __name__ == '__main__':
    main()
