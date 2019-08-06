import os
import glob

from fblib.util.mypath import Path


def parse_folder(exp_root=Path.exp_dir(),
                 exp_group='multitask-pascal',
                 tasks=None,
                 db_name='PASCALContext',
                 query='*',
                 dic={}):

    if tasks is None:
        tasks = ['edge', 'semseg', 'human_parts', 'normals', 'sal', 'depth']

    exp_group_dir = os.path.join(exp_root, exp_group)
    dirs = os.listdir(exp_group_dir)
    dirs.sort()

    # Examine all subdirectories
    best_perf = [0] * len(tasks)
    if 'normals' in tasks:
        best_perf[tasks.index('normals')] = 100
    if 'depth' in tasks:
        best_perf[tasks.index('depth')] = 100
    if 'albedo' in tasks:
        best_perf[tasks.index('albedo')] = 100

    for d in dirs:
        dir_in = os.path.join(exp_group_dir, d)

        # If results folder in dir, print results
        if ('Results_' + db_name) in os.listdir(dir_in):
            perf = [0] * len(tasks)
            task_counter = 0
            # Iterate through all tasks
            for i, task in enumerate(tasks):
                if task == 'normals' or task == 'depth' or task == 'albedo':
                    fnames = glob.glob(dir_in + '/Results_' + db_name + '/' + query + task + '_measures.txt')
                elif task == 'edge':
                    fnames = glob.glob(dir_in + '/Results_' + db_name + '/' + query + task + '_ods_f.txt')
                else:
                    fnames = glob.glob(dir_in+'/Results_' + db_name + '/' + query + task + '_mIoU.txt')

                if not fnames:
                    perf[i] = -1
                    continue
                task_counter += 1

                with open(fnames[0], 'r') as f:
                    if task == 'normals':
                        tmp = f.readline().replace("mean", "").strip()
                        perf[i] = float(tmp.strip())
                        # Check if min
                        if perf[i] < best_perf[i]:
                            best_perf[i] = perf[i]
                    elif task == 'depth' or task == 'albedo':
                        tmp = f.readline().replace("rmse", "").strip()
                        perf[i] = float(tmp.strip())
                        # Check if min
                        if perf[i] < best_perf[i]:
                            best_perf[i] = perf[i]
                    else:
                        tmp = f.read()
                        if task != 'edge' and task != 'sal':
                            perf[i] = float(tmp.strip())
                        else:
                            perf[i] = 100 * float(tmp.strip())

                        # Check if max
                        if perf[i] > best_perf[i]:
                            best_perf[i] = perf[i]

            perf_str = [task + ' ' + '%06.3f' % perf[i] + '   ' for i, task in enumerate(tasks)]
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


def find_best(dic, baseline, ignore=None):

    scores = {x: 0 for x in dic}
    max_score = -1000
    for x in dic:
        for j in range(len(dic[x])):
            if ignore and j in ignore:
                continue
            if dic[x][j] == -1 or baseline[j] == -1:
                continue
            tmp_score = (dic[x][j] - baseline[j]) / baseline[j] * 100
            if j != 3:
                scores[x] += tmp_score
            else:
                scores[x] -= tmp_score
        if scores[x] > max_score and scores[x] != 0:
            max_name = x
            max_score = scores[x]

    max_lst = dic[max_name]

    print('Max score with name {} : {}'.format(max_name, max_score))
    print('Result in lst: {}'.format(max_lst))
    return {'scores': scores, 'max_score': max_score, 'max_lst': max_lst, 'max_name': max_name}


if __name__ == '__main__':

    print('\nResults:')
    dic = {}

    db = 'PASCALContext'

    if db == 'PASCALContext':
        parse_folder(exp_group='pascal_se', query='*res101*trBatch-8*', db_name='PASCALContext', dic=dic,
                     tasks=['edge', 'semseg', 'human_parts', 'normals', 'sal'])
    elif db == 'NYUD':
        parse_folder(exp_group='nyud_se', query='*res101*', db_name='NYUD', dic=dic,
                     tasks=['edge', 'semseg', 'normals', 'depth'])
    elif db == 'FSV':
        parse_folder(exp_group='fsv_se', query='*res101*', db_name='FSV', dic=dic,
                     tasks=['semseg', 'albedo', 'depth'])

    baseline = [72, 67, 59, 15, 66]
    result = find_best(dic, baseline=baseline, ignore=None)


