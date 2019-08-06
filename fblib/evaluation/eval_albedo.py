import warnings
import cv2
import os.path
import numpy as np
import glob
import json


def eval_albedo(loader, folder):

    rmses = []
    for i, sample in enumerate(loader):

        if i % 500 == 0:
            print('Evaluating Albedo: {} of {} objects'.format(i, len(loader)))

        # Load result
        filename = os.path.join(folder, sample['meta']['image'] + '.png')
        pred = cv2.imread(filename).astype(np.float32)[..., ::-1] / 255

        label = sample['albedo']

        if pred.shape != label.shape:
            warnings.warn('Prediction and ground truth have different size. Resizing Prediction..')
            pred = cv2.resize(pred, label.shape[::-1], interpolation=cv2.INTER_LINEAR)

        rmse_tmp = (label - pred) ** 2
        rmse_tmp = np.sqrt(np.mean(rmse_tmp))
        rmses.extend([rmse_tmp])

    rmses = np.array(rmses)

    eval_result = dict()
    eval_result['rmse'] = np.mean(rmses)

    eval_result = {x: eval_result[x].tolist() for x in eval_result}

    return eval_result


def eval_and_store_albedo(database, save_dir, exp_name, overfit=False):

    # Dataloaders
    if database == 'FSV':
        from fblib.dataloaders import fsv as fsv
        gt_set = 'test'
        db = fsv.FSVGTA(split=gt_set, do_albedo=True, overfit=overfit)
    else:
        raise NotImplementedError

    res_dir = os.path.join(save_dir, exp_name, 'Results_' + database)
    base_name = database + '_' + gt_set + '_' + exp_name + '_albedo'
    fname = os.path.join(res_dir, base_name + '.json')

    # Check if already evaluated
    if os.path.isfile(fname):
        with open(fname, 'r') as f:
            eval_results = json.load(f)
    else:
        eval_results = eval_albedo(db, os.path.join(res_dir, 'albedo'))
        with open(fname, 'w') as f:
            json.dump(eval_results, f)

    print('Results for Albedo Estimation')
    for x in eval_results:
        spaces = ''
        for j in range(0, 15 - len(x)):
            spaces += ' '
        print('{0:s}{1:s}{2:.4f}'.format(x, spaces, eval_results[x]))


def main():
    from fblib.util.mypath import Path
    database = 'FSV'
    save_dir = os.path.join(Path.exp_dir(), 'fsv_se/albedo')

    # Evaluate all sub-folders
    exp_names = glob.glob(save_dir + '/*')
    exp_names = [x.split('/')[-1] for x in exp_names]
    for exp_name in exp_names:
        if os.path.isdir(os.path.join(save_dir, exp_name, 'Results_' + database)):
            print('Evaluating: {}'.format(exp_name))
            eval_and_store_albedo(database, save_dir, exp_name)


if __name__ == '__main__':
    main()
