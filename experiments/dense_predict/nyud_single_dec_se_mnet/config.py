import os

import torch
from collections import OrderedDict

# Networks
import fblib.networks.deeplab_se_mobilenet_v2_multitask as se_mobilenet_v2

# Common configs
from experiments.dense_predict.common_configs import get_loss
from experiments.dense_predict.common_configs import get_train_loader, get_test_loader
from experiments.dense_predict.common_configs import get_transformations


def get_net_mnet(p, modelName='model'):
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
                  use_modulation=p['squeeze'])

    if 'finetune_model' not in p:
        if p['resume_epoch'] != 0:
            print("Initializing weights from: {}".format(
                os.path.join(p['save_dir'], 'models', modelName + '_epoch-' + str(p['resume_epoch'] - 1) + '.pth')))
            state_dict_checkpoint = torch.load(
                os.path.join(p['save_dir'], 'models', modelName + '_epoch-' + str(p['resume_epoch'] - 1) + '.pth')
                , map_location=lambda storage, loc: storage)

            if 'module.' in list(state_dict_checkpoint.keys())[0]:
                new_state_dict = OrderedDict()
                for k, v in state_dict_checkpoint.items():
                    name = k.replace('module.', '')  # remove `module.`
                    new_state_dict[name] = v
            else:
                new_state_dict = state_dict_checkpoint
            net.load_state_dict(new_state_dict)
    else:
        print("Finetuning, starting from {} weights".format(p['finetune_model']))
        print("Initializing weights from: {}".format(os.path.join(p['finetune_model'] + '.pth')))
        state_dict_checkpoint = torch.load(os.path.join(p['finetune_model'] + '.pth'),
                                           map_location=lambda storage, loc: storage)
        if 'module.' in list(state_dict_checkpoint.keys())[0]:
            new_state_dict = OrderedDict()
            for k, v in state_dict_checkpoint.items():
                name = k.replace('module.', '')  # remove `module.`
                new_state_dict[name] = v
        else:
            new_state_dict = state_dict_checkpoint
        net.load_state_dict(new_state_dict)

    return net


def get_train_params(net, p):

    train_params = [{'params': se_mobilenet_v2.get_lr_params(net, part='backbone', tasks=p.TASKS.NAMES),
                     'lr': p['lr']},
                    {'params': se_mobilenet_v2.get_lr_params(net, part='decoder',  tasks=p.TASKS.NAMES),
                     'lr': p['lr'] * p['lr_dec']},
                    {'params': se_mobilenet_v2.get_lr_params(net, part='task_specific', tasks=p.TASKS.NAMES),
                     'lr': p['lr'] * p['lr_tsk']}]
    if p['dscr_type'] is not None:
        train_params.append(
            {'params': se_mobilenet_v2.get_lr_params(net, part='discriminator', tasks=p.TASKS.NAMES),
             'lr': p['lr'] * p['lr_dscr']})

    return train_params


def get_exp_name(cfg, args):
    """
    Creates the name experiment from the configuration file and the arguments
    """
    # Delete convenient way to gridsearch
    del args.active_tasks

    # Resume epoch not needed for name
    del args.resume_epoch

    # Delete default parameters for edge detection
    if not cfg.DO_EDGE:
        del args.edge_w

    # Delete default weight decay from name
    if args.wd == 1e-04:
        del args.wd

    # Delete default learning rate of decoder
    if args.lr_dec == 1:
        del args.lr_dec

    # Delete discriminator params if not used
    if args.lr_dscr == 1 or not cfg['dscr_type']:
        del args.lr_dscr
    if not cfg['dscr_type']:
        del args.dscr_w
    if cfg['dscr_type'] != 'fconv':
        del args.dscrd
        del args.dscrk

    # Experiment name string
    name_args = []
    for x in vars(args):
        if x.find('do_') != 0 and x != 'overfit' and getattr(args, x):
            tmp = getattr(args, x)
            if isinstance(tmp, list):
                tmp = "_".join([str(x) for x in tmp])
            else:
                tmp = str(tmp)
            name_args.append(x + '-' + tmp)

    name_args = [x.replace('-True', '') for x in name_args]

    # Experiment folder (task) string
    task_args = [x.replace('do_', '') for x in vars(args) if x.find('do_') == 0 and getattr(args, x)]

    return task_args, name_args

