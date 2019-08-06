import os
import glob
import json

from fblib.util.mypath import Path


def sync_and_evaluate_one_folder(database, save_dir, exp_name, prefix=None, all_tasks_present=False):
    # dataset specific parameters
    if database == 'BSDS500':
        num_req_files = 200
        gt_set = ''
    elif database == 'PASCALContext':
        if all_tasks_present:
            num_req_files = 1853
            gt_set = 'val_all_tasks_present'
        else:
            num_req_files = 5105
            gt_set = 'val'
    elif database == 'NYUD':
        num_req_files = 654
        gt_set = 'val'
    else:
        raise NotImplementedError

    if prefix is None:
        res_exp_name = exp_name
    else:
        res_exp_name = prefix + '_' + exp_name

    # Check whether results of experiments exist
    chk_dir = os.path.join(save_dir, exp_name, 'Results_' + database, 'edge')
    if not os.path.exists(chk_dir):
        print('Experiment {} is not yet ready. Omitting this directory'.format(exp_name))
        return

    # Check for filenames
    fnames = sorted(glob.glob(os.path.join(chk_dir, '*')))
    if len(fnames) < num_req_files:
        print('Something is wrong with this directory. Check required: {}'.format(exp_name))
        return
    elif len(fnames) > num_req_files:
        print('Already synced: {}'.format(exp_name))
    else:
        # Seism path
        seism_cluster_dir = Path.seism_root_dir()

        # rsync to seism
        rsync_str = 'rsync -aP {}/ '.format(chk_dir)
        rsync_str += 'kmaninis@reinhold.ee.ethz.ch:{}/datasets/{}/{} '.format(seism_cluster_dir, database, res_exp_name)
        rsync_str += '--exclude=models --exclude=*.txt'
        print(rsync_str)
        os.system(rsync_str)

        # Submit the job
        subm_job_str = 'ssh kmaninis@reinhold.ee.ethz.ch  "source /home/sgeadmin/BIWICELL/common/settings.sh;' \
                       'source /home/sgeadmin/BIWICELL/common/settings.sh;'
        subm_job_str += 'cp {}/parameters/HED.txt {}/parameters/{}.txt; ' \
                        ''.format(seism_cluster_dir, seism_cluster_dir, res_exp_name)
        subm_job_str += 'qsub -N evalFb -t 1-102 {}/eval_in_cluster.py {} read_one_cont_png fb 1 102 {} {}"' \
                        ''.format(seism_cluster_dir, res_exp_name, database, gt_set)
        print(subm_job_str)
        os.system(subm_job_str)

        # Leave the proof of submission
        os.system('touch {}/SYNCED_TO_REINHOLD'.format(chk_dir))


def sync_evaluated_results(database, save_dir, exp_name, prefix=None):
    if prefix is not None:
        res_exp_name = prefix + '_' + exp_name
    else:
        res_exp_name = exp_name

    split = 'val'

    # Check whether results of experiment exists
    chk_dir = os.path.join(save_dir, exp_name, 'Results_' + database)
    if not os.path.exists(chk_dir):
        print('Experiment {} is not yet ready. Omitting this directory'.format(exp_name))
        return

    chk_file = os.path.join(save_dir, exp_name, 'Results_' + database,
                            database + '_' + split + '_' + exp_name + '_edge.json')

    if os.path.isfile(chk_file):
        with open(chk_file, 'r') as f:
            eval_results = json.load(f)
    else:
        print('Creating json: {}'.format(res_exp_name))
        eval_results = {}
        for measure in {'ods_f', 'ois_f', 'ap'}:
            tmp_fname = os.path.join(Path.seism_root_dir(), 'results', 'pr_curves', database,
                                     database + '_' + split + '_fb_' + res_exp_name + '_' + measure + '.txt')
            if not os.path.isfile(tmp_fname):
                print('Result not available')
                continue

            with open(tmp_fname, 'r') as f:
                eval_results[measure] = float(f.read().strip())

        # Create edge json file
        if eval_results:
            print('Saving into .json: {}'.format(chk_file))
            with open(chk_file, 'w') as f:
                json.dump(eval_results, f)

    for measure in eval_results:
        print('{}: {}'.format(measure, eval_results[measure]))


def sync_and_evaluate_subfolders(p, database):
    print('Starting check in parent directory: {}'.format(p['save_dir_root']))
    dirs = os.listdir(p['save_dir_root'])
    for exp in dirs:
        sync_and_evaluate_one_folder(database=database,
                                     save_dir=p['save_dir_root'],
                                     exp_name=exp,
                                     prefix=p['save_dir_root'].split('/')[-1],
                                     all_tasks_present=(exp.find('mini') >= 0))


def gather_results(p, database):
    print('Gathering results: {}'.format(p['save_dir_root']))
    dirs = os.listdir(p['save_dir_root'])
    for exp in dirs:
        sync_evaluated_results(database=database,
                               save_dir=p['save_dir_root'],
                               exp_name=exp,
                               prefix=p['save_dir_root'].split('/')[-1])


def main():
    exp_root_dir = os.path.join(Path.exp_dir(), 'pascal_se')
    edge_dirs = glob.glob(os.path.join(exp_root_dir, 'edge*'))

    p = {}

    for edge_dir in edge_dirs:
        p['save_dir_root'] = os.path.join(exp_root_dir, edge_dir)
        # sync_and_evaluate_subfolders(p, 'NYUD')
        gather_results(p, 'PASCALContext')


if __name__ == '__main__':
    main()
