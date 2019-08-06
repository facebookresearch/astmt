import warnings
import cv2
import os.path
import numpy as np
import glob
import json


def normal_ize(arr):
    arr_norm = np.linalg.norm(arr, ord=2, axis=2)[..., np.newaxis] + 1e-12
    return arr / arr_norm


def eval_normals(loader, folder):

    deg_diff = []
    for i, sample in enumerate(loader):

        if i % 500 == 0:
            print('Evaluating Surface Normals: {} of {} objects'.format(i, len(loader)))

        # Check for valid labels
        label = sample['normals']
        uniq = np.unique(label)
        if len(uniq) == 1 and uniq[0] == 0:
            continue

        # Load result
        filename = os.path.join(folder, sample['meta']['image'] + '.png')
        pred = 2. * cv2.imread(filename).astype(np.float32)[..., ::-1] / 255. - 1
        pred = normal_ize(pred)

        if pred.shape != label.shape:
            warnings.warn('Prediction and ground truth have different size. Resizing Prediction..')
            pred = cv2.resize(pred, label.shape[::-1], interpolation=cv2.INTER_CUBIC)

        valid_mask = (np.linalg.norm(label, ord=2, axis=2) != 0)
        pred[np.invert(valid_mask), :] = 0.
        label[np.invert(valid_mask), :] = 0.
        label = normal_ize(label)

        deg_diff_tmp = np.rad2deg(np.arccos(np.clip(np.sum(pred * label, axis=2), a_min=-1, a_max=1)))
        deg_diff.extend(deg_diff_tmp[valid_mask])

    deg_diff = np.array(deg_diff)
    eval_result = dict()
    eval_result['mean'] = np.mean(deg_diff)
    eval_result['median'] = np.median(deg_diff)
    eval_result['rmse'] = np.mean(deg_diff ** 2) ** 0.5
    eval_result['11.25'] = np.mean(deg_diff < 11.25) * 100
    eval_result['22.5'] = np.mean(deg_diff < 22.5) * 100
    eval_result['30'] = np.mean(deg_diff < 30) * 100

    eval_result = {x: eval_result[x].tolist() for x in eval_result}

    return eval_result


def eval_and_store_normals(database, save_dir, exp_name, overfit=False):

    # Dataloaders
    if database == 'PASCALContext':
        from fblib.dataloaders import pascal_context as pascal_context
        gt_set = 'val'
        db = pascal_context.PASCALContext(split=gt_set, do_edge=False, do_human_parts=False, do_semseg=False,
                                          do_normals=True, overfit=overfit)
    elif database == 'NYUD':
        from fblib.dataloaders import nyud as nyud
        gt_set = 'val'
        db = nyud.NYUD_MT(split=gt_set, do_normals=True, overfit=overfit)
    else:
        raise NotImplementedError

    res_dir = os.path.join(save_dir, exp_name, 'Results_' + database)
    base_name = database + '_' + gt_set + '_' + exp_name + '_normals'
    fname = os.path.join(res_dir, base_name + '.json')

    # Check if already evaluated
    if os.path.isfile(fname):
        with open(fname, 'r') as f:
            eval_results = json.load(f)
    else:
        eval_results = eval_normals(db, os.path.join(res_dir, 'normals'))
        with open(fname, 'w') as f:
            json.dump(eval_results, f)

    print('Results for Surface Normal Estimation')
    for x in eval_results:
        spaces = ""
        for j in range(0, 15 - len(x)):
            spaces += ' '
        print('{0:s}{1:s}{2:.4f}'.format(x, spaces, eval_results[x]))


def main():
    from fblib.util.mypath import Path
    database = 'PASCALContext'
    save_dir = os.path.join(Path.exp_dir(), 'pascal_se/edge_semseg_human_parts_normals_sal')

    # Evaluate all sub-folders
    exp_names = glob.glob(save_dir + '/*')
    exp_names = [x.split('/')[-1] for x in exp_names]
    for exp_name in exp_names:
        if os.path.isdir(os.path.join(save_dir, exp_name, 'Results_' + database)):
            print('Evaluating: {}'.format(exp_name))
            try:
                eval_and_store_normals(database, save_dir, exp_name)
            except FileNotFoundError:
                print('Results of {} are not ready'.format(exp_name))


if __name__ == '__main__':
    main()
