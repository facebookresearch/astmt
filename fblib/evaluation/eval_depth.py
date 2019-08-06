import warnings
import cv2
import os.path
import numpy as np
import glob
import json
import scipy.io as sio


def eval_depth(loader, folder):

    rmses = []
    log_rmses = []
    for i, sample in enumerate(loader):

        if i % 500 == 0:
            print('Evaluating depth: {} of {} objects'.format(i, len(loader)))

        # Load result
        filename = os.path.join(folder, sample['meta']['image'] + '.mat')
        pred = sio.loadmat(filename)['depth'].astype(np.float32)

        label = sample['depth']

        if pred.shape != label.shape:
            warnings.warn('Prediction and ground truth have different size. Resizing Prediction..')
            pred = cv2.resize(pred, label.shape[::-1], interpolation=cv2.INTER_LINEAR)

        label[label == 0] = 1e-9
        pred[pred <= 0] = 1e-9

        valid_mask = (label != 0)
        pred[np.invert(valid_mask)] = 0.
        label[np.invert(valid_mask)] = 0.
        n_valid = np.sum(valid_mask)

        log_rmse_tmp = (np.log(label) - np.log(pred)) ** 2
        log_rmse_tmp = np.sqrt(np.sum(log_rmse_tmp) / n_valid)
        log_rmses.extend([log_rmse_tmp])

        rmse_tmp = (label - pred) ** 2
        rmse_tmp = np.sqrt(np.sum(rmse_tmp) / n_valid)
        rmses.extend([rmse_tmp])

    rmses = np.array(rmses)
    log_rmses = np.array(log_rmses)

    eval_result = dict()
    eval_result['rmse'] = np.mean(rmses)
    eval_result['log_rmse'] = np.median(log_rmses)

    return eval_result


def eval_and_store_depth(database, save_dir, exp_name, overfit=False):

    # Dataloaders
    if database == 'NYUD':
        from fblib.dataloaders import nyud as nyud
        gt_set = 'val'
        db = nyud.NYUD_MT(split=gt_set, do_depth=True, overfit=overfit)
    elif database == 'FSV':
        from fblib.dataloaders import fsv as fsv
        gt_set = 'test'
        db = fsv.FSVGTA(split='test', do_depth=True, overfit=overfit)
    else:
        raise NotImplementedError

    res_dir = os.path.join(save_dir, exp_name, 'Results_' + database)
    base_name = database + '_' + gt_set + '_' + exp_name + '_depth'
    fname = os.path.join(res_dir, base_name + '.json')

    # Check if already evaluated
    if os.path.isfile(fname):
        with open(fname, 'r') as f:
            eval_results = json.load(f)
    else:
        eval_results = eval_depth(db, os.path.join(res_dir, 'depth'))
        with open(fname, 'w') as f:
            json.dump(eval_results, f)

    print('Results for Depth Estimation')
    for x in eval_results:
        spaces = ''
        for j in range(0, 15 - len(x)):
            spaces += ' '
        print('{0:s}{1:s}{2:.4f}'.format(x, spaces, eval_results[x]))


def main():
    from fblib.util.mypath import Path
    database = 'NYUD'
    save_dir = os.path.join(Path.exp_dir(), 'nyud_se/edge_semseg_normals_depth')

    # Evaluate all sub-folders
    exp_names = glob.glob(save_dir + '/*')
    exp_names = [x.split('/')[-1] for x in exp_names]
    for exp_name in exp_names:
        if os.path.isdir(os.path.join(save_dir, exp_name, 'Results_' + database)):
            print('Evaluating: {}'.format(exp_name))
            try:
                eval_and_store_depth(database, save_dir, exp_name)
            except FileNotFoundError:
                print('Results of {} are not ready'.format(exp_name))


if __name__ == '__main__':
    main()
