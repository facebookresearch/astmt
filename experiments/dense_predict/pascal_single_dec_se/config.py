import os

import torch
from collections import OrderedDict

# Networks
import fblib.networks.deeplab_se_resnet_multitask_single_dec as se_resnet_multitask

# Common configs
from experiments.dense_predict.common_configs import get_loss
from experiments.dense_predict.common_configs import get_train_loader, get_test_loader
from experiments.dense_predict.common_configs import get_transformations


def get_net_resnet(p, modelName='model'):
    """
    Define the network (standard Deeplab ResNet101) and the trainable parameters
    """
    classifier = p['classifier'] if 'classifier' in p else 'psp'
    groupnorm = p['gnorm'] if 'gnorm' in p else False

    if p['arch'] == 'se_res26':
        network = se_resnet_multitask.se_resnet26
    elif p['arch'] == 'se_res50':
        network = se_resnet_multitask.se_resnet50
    elif p['arch'] == 'se_res101':
        network = se_resnet_multitask.se_resnet101
    else:
        raise NotImplementedError('ResNet: Choose between among se_res26, se_res50, and se_res101')

    print('Creating ResNet model: {}'.format(p.NETWORK))

    net = network(n_classes=p.TASKS.NUM_OUTPUT, pretrained=p['pretr'], nInputChannels=p.NETWORK.N_INPUT_CHANNELS,
                  classifier=classifier, groupnorm=groupnorm, tasks=p.TASKS.NAMES, output_stride=p['stride'],
                  train_norm_layers=p['trNorm'], sync_bnorm=p['sync_bnorm'], width_decoder=p['dec_w'],
                  squeeze_mt=p['seenc'], squeeze_dec=p['sedec'], adapters=p['adapters'], smooth=p['smooth'],
                  norm_per_task=p['norm_per_task'], im_features=p['cond'], trans=p['trans'],
                  dscr_type=p['dscr_type'], dscr_d=p['dscrd'], dscr_k=p['dscrk'])

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

    train_params = [{'params': se_resnet_multitask.get_lr_params(net, part='backbone', tasks=p.TASKS.NAMES),
                     'lr': p['lr']},
                    {'params': se_resnet_multitask.get_lr_params(net, part='decoder',  tasks=p.TASKS.NAMES),
                     'lr': p['lr'] * p['lr_dec']},
                    {'params': se_resnet_multitask.get_lr_params(net, part='task_specific', tasks=p.TASKS.NAMES),
                     'lr': p['lr'] * p['lr_tsk']}]
    if p['dscr_type'] is not None:
        train_params.append(
            {'params': se_resnet_multitask.get_lr_params(net, part='discriminator', tasks=p.TASKS.NAMES),
             'lr': p['lr'] * p['lr_dscr']})

    return train_params

