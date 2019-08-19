# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import sys
import cv2
import argparse
import torch
import tarfile
from six.moves import urllib
from easydict import EasyDict as edict

# Networks
import fblib.networks.deeplab_multi_task.deeplab_se_mobilenet_v2_multitask as se_mobilenet_v2

# Common configs
from experiments.dense_predict.common_configs import get_loss, get_train_loader, get_test_loader, get_transformations

from fblib.util.mypath import Path


def parse_args():

    def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1")

    parser = argparse.ArgumentParser(description='Multi-task Learning with PASCAL and MobileNet-v2')

    # Select tasks
    parser.add_argument('--active_tasks', type=int, nargs='+', default=[1, 1, 1, 1, 1],
                        help='Which tasks to train?')

    # General parameters
    parser.add_argument('--arch', type=str, default='mnetv2',
                        help='network: Mobilenet v2')
    parser.add_argument('--pretr', type=str, default='imagenet',
                        help='pre-trained model: "imagenet" or "scratch"')
    parser.add_argument('--trBatch', type=int, default=16,
                        help='training batch size')
    parser.add_argument('-lr', type=float, default=0.001,
                        help='initial learning rate. poly-learning rate is used.')
    parser.add_argument('--lr_dec', type=float, default=1,
                        help='decoder learning rate multiplier')
    parser.add_argument('-wd', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--epochs', type=int, default=130,
                        help='total number of epochs for training')
    parser.add_argument('--resume_epoch', type=int, default=0,
                        help='Resume Epoch #')
    parser.add_argument('--stride', type=int, default=16,
                        help='output stride of ResNet backbone. If set to 16 saves significant memory')
    parser.add_argument('--trNorm', type=str2bool, default=True,
                        help='train normalization layers of backbone?')
    parser.add_argument('--poly', type=str2bool, default=True,
                        help='Use poly learning rate')
    parser.add_argument('--overfit', type=str2bool, default=False,
                        help='overfit to small subset of data for debugging purposes')

    # Squeeze and Excitation Parameters
    parser.add_argument('--seenc', type=str2bool, default=True,
                        help='Squeeze and excitation per task on encoder?')
    parser.add_argument('--sedec', type=str2bool, default=True,
                        help='Squeeze and excitation per task on decoder?')
    parser.add_argument('--lr_tsk', type=float, default=-1,
                        help='Task Specific layer learning rate multiplier')

    # Discriminator parameters
    parser.add_argument('--dscr', type=str, default='None',
                        help='Use discriminator?')
    parser.add_argument('--lr_dscr', type=int, default=1,
                        help='learning rate multiplier of discriminator?')
    parser.add_argument('--dscr_w', type=float, default=0,
                        help='weight of discriminator in the range [0, 1]')
    parser.add_argument('--dscrd', type=int, default=2,
                        help='Depth of discriminator')
    parser.add_argument('--dscrk', type=int, default=1,
                        help='kernel size of discriminator')

    # Task-specific parameters
    parser.add_argument('--edge_w', type=float, default=0.95,
                        help='weighting the positive loss for boundary detection as w * L_pos + (1 - w) * L_neg')

    return parser.parse_args()


def create_config():

    cfg = edict()

    args = parse_args()

    # Parse tasks
    assert (len(args.active_tasks) == 5)
    args.do_edge = args.active_tasks[0]
    args.do_semseg = args.active_tasks[1]
    args.do_human_parts = args.active_tasks[2]
    args.do_normals = args.active_tasks[3]
    args.do_sal = args.active_tasks[4]

    print('\nThis script was run with the following parameters:')
    for x in vars(args):
        print('{}: {}'.format(x, str(getattr(args, x))))

    cfg.resume_epoch = args.resume_epoch

    cfg.DO_EDGE = args.do_edge
    cfg.DO_SEMSEG = args.do_semseg
    cfg.DO_HUMAN_PARTS = args.do_human_parts
    cfg.DO_NORMALS = args.do_normals
    cfg.DO_SAL = args.do_sal

    if not cfg.DO_EDGE and not cfg.DO_SEMSEG and not cfg.DO_HUMAN_PARTS and not cfg.DO_NORMALS and not cfg.DO_SAL:
        raise ValueError("Select at least one task")

    cfg['arch'] = args.arch
    cfg['pretr'] = args.pretr
    cfg['trBatch'] = args.trBatch
    cfg['lr'] = args.lr
    cfg['lr_dec'] = args.lr_dec
    cfg['wd'] = args.wd
    cfg['epochs'] = args.epochs
    cfg['stride'] = args.stride
    cfg['trNorm'] = args.trNorm
    cfg['poly'] = args.poly

    # Set squeeze and excitation parameters
    cfg['seenc'] = args.seenc
    cfg['sedec'] = args.sedec

    if args.dscr == 'None':
        args.dscr = None

    cfg['dscr_type'] = args.dscr
    cfg['lr_dscr'] = args.lr_dscr
    cfg['dscr_w'] = args.dscr_w
    cfg['dscrd'] = args.dscrd
    cfg['dscrk'] = args.dscrk

    task_args, name_args = get_exp_name(args)

    cfg['exp_folder_name'] = 'pascal_mnet'
    cfg['exp_name'] = "_".join(name_args)
    cfg['tasks_name'] = "_".join(task_args)
    cfg['save_dir_root'] = os.path.join(Path.exp_dir(), cfg['exp_folder_name'], cfg['tasks_name'])
    cfg['train_db_name'] = ['PASCALContext', ]
    cfg['test_db_name'] = 'PASCALContext'
    cfg['infer_db_names'] = ['PASCALContext', ]

    # Which tasks?
    cfg.TASKS = edict()
    cfg.TASKS.NAMES = []
    cfg.TASKS.NUM_OUTPUT = {}  # How many outputs per task?
    cfg.TASKS.TB_MIN = {}
    cfg.TASKS.TB_MAX = {}
    cfg.TASKS.LOSS_MULT = {}
    cfg.TASKS.FLAGVALS = {'image': cv2.INTER_CUBIC}
    cfg.TASKS.INFER_FLAGVALS = {}

    if cfg.DO_EDGE:
        # Edge Detection
        print('Adding task: Edge Detection')
        tmp = 'edge'
        cfg.TASKS.NAMES.append(tmp)
        cfg.TASKS.NUM_OUTPUT[tmp] = 1
        cfg.TASKS.TB_MIN[tmp] = 0
        cfg.TASKS.TB_MAX[tmp] = cfg.TASKS.NUM_OUTPUT[tmp]
        cfg.TASKS.LOSS_MULT[tmp] = 50
        cfg.TASKS.FLAGVALS[tmp] = cv2.INTER_NEAREST
        cfg.TASKS.INFER_FLAGVALS[tmp] = cv2.INTER_LINEAR

        # Add task-specific parameters from parser
        cfg['edge_w'] = args.edge_w
        cfg['eval_edge'] = False

    if cfg.DO_SEMSEG:
        # Semantic Segmentation
        print('Adding task: Semantic Segmentation')
        tmp = 'semseg'
        cfg.TASKS.NAMES.append(tmp)
        cfg.TASKS.NUM_OUTPUT[tmp] = 21
        cfg.TASKS.TB_MIN[tmp] = 0
        cfg.TASKS.TB_MAX[tmp] = cfg.TASKS.NUM_OUTPUT[tmp] - 1
        cfg.TASKS.LOSS_MULT[tmp] = 1
        cfg.TASKS.FLAGVALS[tmp] = cv2.INTER_NEAREST
        cfg.TASKS.INFER_FLAGVALS[tmp] = cv2.INTER_NEAREST

    if cfg.DO_HUMAN_PARTS:
        # Human Parts Segmentation
        print('Adding task: Human Part Segmentation')
        tmp = 'human_parts'
        cfg.TASKS.NAMES.append(tmp)
        cfg.TASKS.NUM_OUTPUT[tmp] = 7
        cfg.TASKS.TB_MIN[tmp] = 0
        cfg.TASKS.TB_MAX[tmp] = cfg.TASKS.NUM_OUTPUT[tmp] - 1
        cfg.TASKS.LOSS_MULT[tmp] = 2
        cfg.TASKS.FLAGVALS[tmp] = cv2.INTER_NEAREST
        cfg.TASKS.INFER_FLAGVALS[tmp] = cv2.INTER_NEAREST

    if cfg.DO_NORMALS:
        # Human Parts Segmentation
        print('Adding task: Normals')
        tmp = 'normals'
        cfg.TASKS.NAMES.append(tmp)
        cfg.TASKS.NUM_OUTPUT[tmp] = 3
        cfg.TASKS.TB_MIN[tmp] = -1
        cfg.TASKS.TB_MAX[tmp] = 1
        cfg.TASKS.LOSS_MULT[tmp] = 10
        cfg.TASKS.FLAGVALS[tmp] = cv2.INTER_CUBIC
        cfg.TASKS.INFER_FLAGVALS[tmp] = cv2.INTER_LINEAR

        cfg['normloss'] = 1  # Hard-coded L1 loss for normals

    if cfg.DO_SAL:
        # Saliency Estimation
        print('Adding task: Saliency')
        tmp = 'sal'
        cfg.TASKS.NAMES.append(tmp)
        cfg.TASKS.NUM_OUTPUT[tmp] = 1
        cfg.TASKS.TB_MIN[tmp] = 0
        cfg.TASKS.TB_MAX[tmp] = 1
        cfg.TASKS.LOSS_MULT[tmp] = 5
        cfg.TASKS.FLAGVALS[tmp] = cv2.INTER_NEAREST
        cfg.TASKS.INFER_FLAGVALS[tmp] = cv2.INTER_LINEAR

    cfg['lr_tsk'] = len(cfg.TASKS.NAMES) if args.lr_tsk < 0 else args.lr_tsk

    cfg.NETWORK = edict()

    # Visualize the network on Tensorboard / pdf?
    cfg.NETWORK.VIS_NET = False

    cfg.TRAIN = edict()
    cfg.TRAIN.SCALE = (512, 512)
    cfg.TRAIN.MOMENTUM = 0.9
    cfg.TRAIN.TENS_VIS = False
    cfg.TRAIN.TENS_VIS_INTER = 1000
    cfg.TRAIN.TEMP_LOSS_INTER = 1000

    cfg.TEST = edict()

    # See evolution of the test set when training?
    cfg.TEST.USE_TEST = True
    cfg.TEST.TEST_INTER = 10
    cfg.TEST.SCALE = (512, 512)

    cfg.SEED = 0
    cfg.EVALUATE = True
    cfg.DEBUG = False

    cfg['overfit'] = args.overfit
    if cfg['overfit']:
        cfg['save_dir_root'] = os.path.join(Path.exp_dir(), cfg['exp_folder_name'])
        cfg['exp_name'] = 'test'

    cfg['save_dir'] = os.path.join(cfg['save_dir_root'], cfg['exp_name'])
    return cfg


def check_downloaded(p):
    def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> %s %.1f%%' %
                         (_fpath, float(count * block_size) /
                          float(total_size) * 100.0))
        sys.stdout.flush()

    def _create_url(name):
        return 'https://data.vision.ee.ethz.ch/kmaninis/share/MTL//models/astmt/{}.tgz'.format(name)

    _model_urls = {
        'pascal_mnet_edge_semseg_human_parts_normals_sal_'
        'arch-mnetv2_pretr-imagenet_trBatch-16_lr-0.001_epochs-130_trNorm_poly_seenc_sedec_edge_w-0.95_130'
    }

    ans = False
    _check = p['exp_folder_name'] + '_' + p['tasks_name'] + '_' + p['exp_name'] + '_' + str(p['resume_epoch'])
    _fpath = os.path.join(Path.exp_dir(), _check + '.tgz')

    if _check in _model_urls:
        if not os.path.isfile(os.path.join(p['save_dir'], 'models',
                                           'model_epoch-' + str(p['resume_epoch'] - 1) + '.pth')):
            urllib.request.urlretrieve(_create_url(_check), _fpath, _progress)

            # extract file
            cwd = os.getcwd()
            print('\nExtracting tar file')
            tar = tarfile.open(_fpath)
            os.chdir(Path.exp_dir())
            tar.extractall()
            tar.close()
            os.chdir(cwd)
            print('Done!')
        ans = True
    return ans


def get_net_mnet(p):
    """
    Define the network (standard Deeplab ResNet101) and the trainable parameters
    """

    print('Creating DeepLab with Mobilenet-V2 model: {}'.format(p.NETWORK))
    network = se_mobilenet_v2.se_mobilenet_v2

    net = network(n_classes=p.TASKS.NUM_OUTPUT,
                  pretrained=p['pretr'],
                  tasks=p.TASKS.NAMES,
                  output_stride=p['stride'],
                  train_norm_layers=p['trNorm'],
                  mod_enc=p['seenc'],
                  mod_dec=p['sedec'],
                  use_dscr=(p['dscr_type'] == 'fconv'),
                  dscr_k=p['dscrk'],
                  dscr_d=p['dscrd'])

    if p['resume_epoch'] != 0:
        print("Initializing weights from: {}".format(
            os.path.join(p['save_dir'], 'models', 'model_epoch-' + str(p['resume_epoch'] - 1) + '.pth')))
        state_dict_checkpoint = torch.load(
            os.path.join(p['save_dir'], 'models', 'model_epoch-' + str(p['resume_epoch'] - 1) + '.pth')
            , map_location=lambda storage, loc: storage)

        net.load_state_dict(state_dict_checkpoint)

    return net


def get_train_params(net, p, lr):
    print('Adjusting learning rate')
    print('Base lr: {}'.format(lr))
    print('Decoder lr: {}'.format(lr * p['lr_dec']))
    print('Task-specific lr: {}'.format(lr * p['lr_tsk']))
    train_params = [{'params': se_mobilenet_v2.get_lr_params(net, part='backbone', tasks=p.TASKS.NAMES),
                     'lr': lr},
                    {'params': se_mobilenet_v2.get_lr_params(net, part='decoder',  tasks=p.TASKS.NAMES),
                     'lr': lr * p['lr_dec']},
                    {'params': se_mobilenet_v2.get_lr_params(net, part='task_specific', tasks=p.TASKS.NAMES),
                     'lr': lr * p['lr_tsk']}]
    if p['dscr_type'] is not None:
        print('Discriminator lr: {}'.format(lr * p['lr_dscr']))
        train_params.append(
            {'params': se_mobilenet_v2.get_lr_params(net, part='discriminator', tasks=p.TASKS.NAMES),
             'lr': lr * p['lr_dscr']})

    return train_params


def get_exp_name(args):
    """
    Creates the name experiment from the configuration file and the arguments
    """
    task_dict = {
        'do_edge': 0,
        'do_semseg': 0,
        'do_human_parts': 0,
        'do_normals': 0,
        'do_sal': 0
    }

    name_dict = {
        'arch': None,
        'pretr': None,
        'trBatch': None,
        'lr': None,
        'wd': 1e-04,
        'epochs': None,
        'stride': 16,
        'trNorm': False,
        'poly': False,
        'seenc': False,
        'sedec': False,
        'dscr': None,
        'lr_dscr': 1,
        'dscr_w': ('dscr', None),
        'dscrd': ('dscr', None),
        'dscrk': ('dscr', None),
        'edge_w': ('do_edge', None),
        'lr_dec': 1,
        'lr_tsk': -1,
    }

    # Experiment folder (task) string
    task_args = [x.replace('do_', '') for x in task_dict if getattr(args, x) != task_dict[x]]

    # Experiment name string
    name_args = []
    for x in name_dict:

        # Check dependencies in tuple
        if isinstance(name_dict[x], tuple):
            elem = name_dict if name_dict[x][0] in name_dict else task_dict
            if elem[name_dict[x][0]] == getattr(args, name_dict[x][0]):
                continue

        if getattr(args, x) != name_dict[x]:
            tmp = getattr(args, x)
            if isinstance(tmp, list):
                tmp = "_".join([str(x) for x in tmp])
            else:
                tmp = str(tmp)
            name_args.append(x + '-' + tmp)
    name_args = [x.replace('-True', '') for x in name_args]

    return task_args, name_args

