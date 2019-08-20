# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import glob
import json

from fblib.util.mypath import Path


def exists_dir(dir_in):

    fold_lst = os.listdir(dir_in)
    for x in fold_lst:
        if os.path.isdir(os.path.join(dir_in, x)):
            return True
    return False


def parse_folder(exp_root=Path.exp_dir(),
                 exp_group='pascal_se',
                 tasks=None,
                 db_name='PASCALContext',
                 query='*',
                 dic={}):

    if tasks is None:
        tasks = ['edge', 'semseg', 'human_parts', 'normals', 'sal', 'depth']

    exp_group_dir = os.path.join(exp_root, exp_group)
    dirs = os.listdir(exp_group_dir)
    dirs.sort()

    best_perf = {task: 0 for task in tasks}
    for task in {'normals', 'depth', 'albedo'}:
        if task in tasks:
            best_perf[task] = 100

    # Examine all subdirectories
    for d in dirs:
        dir_in = os.path.join(exp_group_dir, d)

        # No dir or dir without subdirs
        if not os.path.isdir(dir_in) or not exists_dir(dir_in):
            continue

        # If results folder in dir, print results
        if ('Results_' + db_name) in os.listdir(dir_in):
            perf = {}
            task_counter = 0
            # Iterate through all tasks
            for i, task in enumerate(tasks):
                fnames = glob.glob(dir_in+'/Results_' + db_name + '/' + query + task + '.json')

                if not fnames:
                    perf[task] = -1
                    continue
                task_counter += 1

                with open(fnames[0], 'r') as f:
                    data = json.load(f)

                if task == 'edge':
                    perf[task] = 100 * data['ods_f']
                    if perf[task] > best_perf[task]:
                        best_perf[task] = perf[task]
                elif task == 'semseg':
                    perf[task] = 100 * data['mIoU']
                    if perf[task] > best_perf[task]:
                        best_perf[task] = perf[task]
                elif task == 'human_parts':
                    perf[task] = 100 * data['mIoU']
                    if perf[task] > best_perf[task]:
                        best_perf[task] = perf[task]
                elif task == 'normals':
                    perf[task] = data['mean']
                    if perf[task] < best_perf[task]:
                        best_perf[task] = perf[task]
                elif task == 'depth':
                    perf[task] = data['rmse']
                    if perf[task] < best_perf[task]:
                        best_perf[task] = perf[task]
                elif task == 'albedo':
                    perf[task] = data['rmse']
                    if perf[task] < best_perf[task]:
                        best_perf[task] = perf[task]
                elif task == 'sal':
                    perf[task] = 100 * data['mIoU']
                    if perf[task] > best_perf[task]:
                        best_perf[task] = perf[task]

            perf_str = [task + ' ' + '%06.3f' % perf[task] + '   ' for i, task in enumerate(tasks)]
            perf_str = "".join(perf_str)
            if task_counter > 0:
                print('{}: {}'.format(perf_str, d))
                dic[d] = perf

        elif 'models' in os.listdir(dir_in):
            # Results are not ready yet
            continue
        else:
            # Examine subdirectories recursively
            print('\n\n{}\n'.format(d))
            parse_folder(exp_group=os.path.join(exp_group, d), tasks=tasks, query=query, db_name=db_name, dic=dic)

    print(best_perf)


if __name__ == '__main__':

    print('\nResults:')
    dic = {}

    db = 'PASCALContext'

    if db == 'PASCALContext':
        parse_folder(exp_group='pascal_se', query='*res50*', db_name='PASCALContext', dic=dic,
                     tasks=['edge', 'semseg', 'human_parts', 'normals', 'sal'])
    elif db == 'NYUD':
        parse_folder(exp_group='nyud_se', query='*res101*', db_name='NYUD', dic=dic,
                     tasks=['edge', 'semseg', 'normals', 'depth'])
    elif db == 'FSV':
        parse_folder(exp_group='fsv_se', query='*res101*', db_name='FSV', dic=dic,
                     tasks=['semseg', 'albedo', 'depth'])
    elif db == 'VOC12':
        parse_folder(exp_group='single_task', query='*mnet*', db_name='VOC12', dic=dic,
                     tasks=['semseg'])
