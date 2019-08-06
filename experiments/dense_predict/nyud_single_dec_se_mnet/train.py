# Package Includes
from __future__ import division

import socket
import timeit
from datetime import datetime
import imageio
import argparse
import scipy.io as sio
from easydict import EasyDict as edict

# PyTorch includes
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import interpolate
import encoding

# Custom includes
from fblib.util.helpers import *
from fblib.util.dense_predict.utils import lr_poly
from fblib.layers.loss import normal_ize
from fblib.util.configs import generic_configs
from experiments.dense_predict import common_configs
from fblib.util.mypath import Path
from fblib.util.configs.multitask_visualizer import TBVisualizer
from fblib.util.model_resources.flops import compute_gflops
from fblib.util.model_resources.num_parameters import count_parameters
from fblib.util.dense_predict.utils import AverageMeter

# Custom optimizer
from fblib.util.optimizer_mtl.select_used_modules import make_closure

try:
    from . import config as config  # for PyCharm
except:
    import config as config

# Tensorboard include
from tensorboardX import SummaryWriter


def parse_args():

    def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1")

    parser = argparse.ArgumentParser(description='Multi-task Learning for NYUD')

    # Select tasks
    parser.add_argument('--active_tasks', type=int, nargs='+', default="-1",
                        help='Which tasks to train?')
    parser.add_argument('-w', type=float, default=-1,
                        help='Gridsearch parameter for 1 task')

    # General parameters
    parser.add_argument('--pretr', type=str, default=None,
                        help='pre-trained model: imagenet, voc, coco, or None')
    parser.add_argument('--trBatch', type=int, default=None,
                        help='training batch size')
    parser.add_argument('-lr', type=float, default=None,
                        help='initial learning rate. poly-learning rate is used.')
    parser.add_argument('--lr_dec', type=float, default=1,
                        help='decoder learning rate multiplier')
    parser.add_argument('-wd', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--epochs', type=int, default=None,
                        help='total number of epochs for training')
    parser.add_argument('--resume_epoch', type=int, default=0,
                        help='Resume Epoch #')
    parser.add_argument('--stride', type=int, default=8,
                        help='output stride of ResNet backbone. If set to 16 saves significant memory')
    parser.add_argument('--trNorm', type=str2bool, default=False,
                        help='train normalization layers of backbone?')
    parser.add_argument('--overfit', type=str2bool, default=False,
                        help='overfit to small subset of data for debugging purposes')
    parser.add_argument('--debug_alpha', type=str2bool, default=False,
                        help='Train classifier properly for debugging reasons')

    # Squeeze and Excitation Parameters
    parser.add_argument('--squeeze', type=str2bool, default=False,
                        help='Squeeze and excitation per task?')
    parser.add_argument('--lr_tsk', type=float, default=-1,
                        help='Task Specific layer learning rate multiplier')
    parser.add_argument('--newop', type=str2bool, default=False,
                        help='Use modified optimizer that updates only layers used in the graph?')

    # Discriminator parameters
    parser.add_argument('--dscr', type=str, default='conv',
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
    parser.add_argument('-edge_w', type=float, default=0.95,
                        help='weighting the positive loss for boundary detection as w * L_pos + (1 - w) * L_neg')

    return parser.parse_args()


def create_config():

    cfg = edict()

    args = parse_args()

    # Parse tasks
    assert (len(args.active_tasks) == 4)
    args.do_edge = args.active_tasks[0]
    args.do_semseg = args.active_tasks[1]
    args.do_normals = args.active_tasks[2]
    args.do_depth = args.active_tasks[3]

    print('\nThis script was run with the following parameters:')
    for x in vars(args):
        print('{}: {}'.format(x, str(getattr(args, x))))

    cfg.resume_epoch = args.resume_epoch

    cfg.DO_EDGE = args.do_edge
    cfg.DO_SEMSEG = args.do_semseg
    cfg.DO_NORMALS = args.do_normals
    cfg.DO_DEPTH = args.do_depth

    if not cfg.DO_EDGE and not cfg.DO_SEMSEG and not cfg.DO_NORMALS and not cfg.DO_DEPTH:
        raise ValueError("Select at least one task")

    cfg['pretr'] = args.pretr
    cfg['trBatch'] = args.trBatch
    cfg['lr'] = args.lr
    cfg['wd'] = args.wd
    cfg['lr_dec'] = args.lr_dec
    cfg['epochs'] = args.epochs
    cfg['stride'] = args.stride
    cfg['trNorm'] = args.trNorm
    cfg['newop'] = args.newop

    # Set squeeze and excitation parameters
    cfg['squeeze'] = args.squeeze

    args.dscr = None if args.dscr == 'None' else args.dscr
    cfg['dscr_type'] = args.dscr

    cfg['lr_dscr'] = args.lr_dscr
    cfg['dscr_w'] = args.dscr_w
    cfg['dscrd'] = args.dscrd
    cfg['dscrk'] = args.dscrk

    task_args, name_args = config.get_exp_name(cfg, args)

    exp_folder_name = 'nyud_se_mnet'
    cfg['exp_name'] = "_".join(name_args) + '_mnet'
    cfg['tasks_name'] = "_".join(task_args)
    cfg['save_dir_root'] = os.path.join(Path.exp_dir(), exp_folder_name, cfg['tasks_name'])
    cfg['train_db_name'] = ['NYUD', ]
    cfg['test_db_name'] = 'NYUD'
    cfg['infer_db_names'] = ['NYUD', ]

    # Which tasks?
    cfg.TASKS = edict()
    cfg.TASKS.NAMES = []
    cfg.TASKS.NUM_OUTPUT = {}  # How many outputs per task?
    cfg.TASKS.TB_MIN = {}
    cfg.TASKS.TB_MAX = {}
    cfg.TASKS.LOSS_MULT = {}
    cfg.TASKS.FLAGVALS = {'image': cv2.INTER_CUBIC}
    cfg.TASKS.INFER_FLAGVALS = {}
    cfg.CASC_TASKS = {}

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
        cfg['eval_edge'] = True

    if cfg.DO_SEMSEG:
        # Semantic Segmentation
        print('Adding task: Semantic Segmentation')
        tmp = 'semseg'
        cfg.TASKS.NAMES.append(tmp)
        cfg.TASKS.NUM_OUTPUT[tmp] = 40
        cfg.TASKS.TB_MIN[tmp] = 0
        cfg.TASKS.TB_MAX[tmp] = cfg.TASKS.NUM_OUTPUT[tmp] - 1
        cfg.TASKS.LOSS_MULT[tmp] = 1
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

    if cfg.DO_DEPTH:
        # Depth Estimation
        print('Adding task: Depth')
        tmp = 'depth'
        cfg.TASKS.NAMES.append(tmp)
        cfg.TASKS.NUM_OUTPUT[tmp] = 1
        cfg.TASKS.TB_MIN[tmp] = 1
        cfg.TASKS.TB_MAX[tmp] = 10
        cfg.TASKS.LOSS_MULT[tmp] = 1
        cfg.TASKS.FLAGVALS[tmp] = cv2.INTER_LINEAR
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

    cfg['debug_alpha'] = args.debug_alpha
    cfg['overfit'] = args.overfit
    if cfg['overfit']:
        cfg['save_dir_root'] = os.path.join(Path.exp_dir(), exp_folder_name)
        cfg['exp_name'] = 'test'

    cfg['save_dir'] = os.path.join(cfg['save_dir_root'], cfg['exp_name'])

    return cfg


def visualize_network(net, p):
    import fblib.util.visualize as viz
    net.eval()
    x = torch.randn(1, p.NETWORK.N_INPUT_CHANNELS, p.TRAIN.SCALE[0], p.TRAIN.SCALE[1])
    x.requires_grad_()

    # pdf visualizer
    for task in p.TASKS.NAMES:
        y, _ = net.forward(x, task)
    g = viz.make_dot(y, net.state_dict())
    g.view(directory='/private/home/kmaninis/')


def main():
    debug = False

    gpu_id = generic_configs.get_gpu_id()
    n_gpus = len(gpu_id) if type(gpu_id) == list else 1
    if n_gpus > 1:
        device = torch.device("cuda:" + str(gpu_id[0]))
    else:
        device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")
    p['device'] = device

    p.TEST.BATCH_SIZE = 3 * n_gpus

    # Setting parameters
    n_epochs = p['epochs']

    print("Total training epochs: {}".format(n_epochs))
    p['n_gpus'] = n_gpus
    print(p)
    print('Training on {}'.format(p['train_db_name']))

    snapshot = n_epochs  # Store a model every snapshot epochs
    test_interval = p.TEST.TEST_INTER  # Run on test set every test_interval epochs
    torch.manual_seed(p.SEED)

    exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]
    if not os.path.exists(os.path.join(p['save_dir'], 'models')):
        if p['resume_epoch'] == 0:
            os.makedirs(os.path.join(p['save_dir'], 'models'))
        else:
            print('Folder does not exist.No checkpoint to resume from. Exiting')
            exit(1)

    p['sync_bnorm'] = True if n_gpus > 1 and p['trNorm'] else False
    net = config.get_net_mnet(p)

    # Visualize the network
    if p.NETWORK.VIS_NET:
        visualize_network(net, p)

    if not isinstance(gpu_id, list):
        gflops = compute_gflops(net, in_shape=(p['trBatch'], 3, p.TRAIN.SCALE[0], p.TRAIN.SCALE[1]),
                                tasks=p.TASKS.NAMES[0])
        print('GFLOPS per task: {}'.format(gflops / p['trBatch']))

        print('\nNumber of parameters (in millions): {0:.3f}'.format(count_parameters(net) / 1e6))
        print('Number of parameters (in millions) for decoder: {0:.3f}\n'.format(count_parameters(net.decoder) / 1e6))

    if isinstance(gpu_id, list):
        print("Using multiple GPUs: {}".format(gpu_id))
        net = nn.DataParallel(net, device_ids=gpu_id).cuda()
        if p['trNorm']:
            encoding.parallel.patch_replication_callback(net)
    else:
        net.to(device)

    if p['resume_epoch'] != n_epochs:
        criteria_tr = {}
        criteria_ts = {}

        running_loss_tr = {task: 0. for task in p.TASKS.NAMES}
        running_loss_ts = {task: 0. for task in p.TASKS.NAMES}
        curr_loss_task = {task: 0. for task in p.TASKS.NAMES}
        counter_tr = {task: 0 for task in p.TASKS.NAMES}
        counter_ts = {task: 0 for task in p.TASKS.NAMES}

        # Discriminator loss variables for logging
        running_loss_tr_dscr = 0
        running_loss_ts_dscr = 0

        # Logging into Tensorboard
        log_dir = os.path.join(p['save_dir'], 'models',
                               datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
        writer = SummaryWriter(log_dir=log_dir)

        # Training parameters and their optimizer
        train_params = config.get_train_params(net, p)

        optimizer = optim.SGD(train_params, lr=p['lr'], momentum=p.TRAIN.MOMENTUM, weight_decay=p['wd'])
        p['optimizer'] = str(optimizer)

        for task in p.TASKS.NAMES:
            # Losses
            criteria_tr[task] = config.get_loss(p, task, phase='train')
            criteria_ts[task] = config.get_loss(p, task, phase='test')
            criteria_tr[task].cuda()
            criteria_ts[task].cuda()

        # Preparation of the data loaders
        transforms_tr, transforms_ts, _ = config.get_transformations(p)
        trainloader = config.get_train_loader(p, db_name=p['train_db_name'], transforms=transforms_tr)
        testloader = config.get_test_loader(p, db_name=p['test_db_name'], transforms=transforms_ts)

        # TensorBoard Image Visualizer
        tb_vizualizer = TBVisualizer(tasks=p.TASKS.NAMES, min_ranges=p.TASKS.TB_MIN, max_ranges=p.TASKS.TB_MAX,
                                     batch_size=p['trBatch'])

        p['transformations_train'] = [str(tran) for tran in transforms_tr.transforms]
        p['transformations_test'] = [str(tran) for tran in transforms_ts.transforms]
        generate_param_report(os.path.join(p['save_dir'], exp_name + '.txt'), p)
        # Train variables
        num_img_tr = len(trainloader)
        num_img_ts = len(testloader)

        print("Training Network")
        # Main Training and Testing Loop
        for epoch in range(p['resume_epoch'], n_epochs):
            top1_dscr = AverageMeter()
            start_time = timeit.default_timer()
            # One training epoch
            net.train()

            alpha = 2. / (1. + np.exp(-10 * ((epoch + 1) / n_epochs))) - 1  # Ganin et al for gradient reversal
            # alpha = min(2 * alpha, 1)
            if p.debug_alpha:
                alpha = -1
            if p['dscr_type']:
                print('Value of alpha: {}'.format(alpha))
            for ii, sample in enumerate(trainloader):
                curr_loss_dscr = 0

                # Grab the input
                inputs = sample['image']
                inputs.requires_grad_()
                inputs = inputs.cuda()

                task_gts = list(sample.keys())
                net_obj = net.module if isinstance(gpu_id, list) else net
                tasks = net_obj.tasks

                gt_elems = {x: sample[x].cuda(non_blocking=True) for x in tasks}
                uniq = {x: gt_elems[x].unique() for x in gt_elems}

                outputs = {}
                for task in tasks:
                    if task not in task_gts:
                        continue
                    if len(uniq[task]) == 1 and uniq[task][0] == 255:
                        continue

                    # Forward pass
                    output = {}
                    features = {}
                    output[task], features[task] = net.forward(inputs, task=task)
                    losses_tasks, losses_dscr, outputs_dscr, grads, task_labels \
                        = net_obj.compute_losses(output, features, criteria_tr, gt_elems, alpha, p)

                    loss_tasks = losses_tasks[task]
                    running_loss_tr[task] += losses_tasks[task].item()
                    curr_loss_task[task] = losses_tasks[task].item()

                    counter_tr[task] += 1

                    # Store output for logging
                    outputs[task] = output[task].detach()

                    if p['dscr_type']:
                        # measure loss, accuracy and record accuracy for discriminator
                        loss_dscr = losses_dscr[task]
                        running_loss_tr_dscr += losses_dscr[task].item()
                        curr_loss_dscr += loss_dscr.item()

                        prec1 = common_configs.accuracy(outputs_dscr[task].data, task_labels[task], topk=(1,))
                        if prec1 != -1:
                            top1_dscr.update(prec1[0].item(), task_labels[task].size(0))

                        loss = (1 - p['dscr_w']) * loss_tasks + p['dscr_w'] * loss_dscr
                    else:
                        loss = loss_tasks

                    if debug:
                        # Don't forget to turn off weight decay and momentum!!!
                        print(task)
                        numbers = [net.layer2[0].se.fc[task][0].weight[-1][-1].detach().cpu().numpy().item()
                                   for task in tasks]
                        print(numbers)

                    optimizer.zero_grad()
                    if p['newop']:
                        optimizer.step(closure=make_closure(loss=loss, net=net))
                    else:
                        loss.backward()
                        optimizer.step()

                    if debug:
                        numbers = [net.layer2[0].se.fc[task][0].weight[-1][-1].detach().cpu().numpy().item()
                                   for task in tasks]
                        print(numbers)

                # Print stuff and log epoch loss into Tensorboard
                if ii % num_img_tr == num_img_tr - 1:
                    print('[Epoch: %d, numImages: %5d]' % (epoch, ii + 1))
                    for task in p.TASKS.NAMES:
                        running_loss_tr[task] = running_loss_tr[task] / counter_tr[task]
                        writer.add_scalar('data/total_loss_epoch' + task,
                                          running_loss_tr[task] / p.TASKS.LOSS_MULT[task], epoch)
                        print('Loss %s: %f' % (task, running_loss_tr[task]))
                        running_loss_tr[task] = 0
                        counter_tr[task] = 0

                    if p['dscr_type']:
                        running_loss_tr_dscr = running_loss_tr_dscr / num_img_tr / len(p.TASKS.NAMES)
                        writer.add_scalar('data/total_loss_epoch_dscr', running_loss_tr_dscr, epoch)
                        print('Loss Discriminator: %f' % running_loss_tr_dscr)
                        print('Train Accuracy Discriminator: Prec@1 {top1.avg:.3f} Error@1 {error1:.3f}'.format(
                            top1=top1_dscr, error1=100 - top1_dscr.avg))
                        writer.add_scalar('data/train_accuracy_dscr', top1_dscr.avg, epoch)
                    stop_time = timeit.default_timer()
                    print("Execution time: " + str(stop_time - start_time) + "\n")

                # Log current train loss into Tensorboard
                for task in p.TASKS.NAMES:
                    writer.add_scalar('data/train_loss_iter_' + task, curr_loss_task[task], ii + num_img_tr * epoch)
                    curr_loss_task[task] = 0.
                    if p['dscr_type']:
                        writer.add_scalar('data/train_loss_dscr_iter', curr_loss_dscr, ii + num_img_tr * epoch)
                        curr_loss_dscr = 0.

                # Log train images to Tensorboard
                if p['overfit'] and p.TRAIN.TENS_VIS and ii % p.TRAIN.TENS_VIS_INTER == 0:
                    curr_iter = ii + num_img_tr * epoch
                    tb_vizualizer.visualize_images_tb(writer, sample, outputs,
                                                      global_step=curr_iter, tag=ii, phase='train')

                if ii % num_img_tr == num_img_tr - 1:
                    lr_ = lr_poly(p['lr'], iter_=epoch, max_iter=n_epochs)
                    print('(poly lr policy) learning rate: {0:.6f}'.format(lr_))
                    train_params = config.get_train_params(net, p)
                    optimizer = optim.SGD(train_params, lr=lr_, momentum=p.TRAIN.MOMENTUM, weight_decay=p['wd'])
                    optimizer.zero_grad()

            # Save the model
            if (epoch % snapshot) == snapshot - 1 and epoch != 0:
                torch.save(net.state_dict(), os.path.join(p['save_dir'], 'models', 'model_epoch-' + str(epoch) + '.pth'))

            # One testing epoch
            if p.TEST.USE_TEST and epoch % test_interval == (test_interval - 1):
                print('Testing Phase')
                top1_dscr = AverageMeter()
                net.eval()
                start_time = timeit.default_timer()
                for ii, sample in enumerate(testloader):

                    inputs = sample['image'].cuda()
                    task_gts = list(sample.keys())

                    net_obj = net.module if isinstance(gpu_id, list) else net
                    tasks = net_obj.tasks
                    gt_elems = {x: sample[x].cuda() for x in tasks}

                    uniq = {x: gt_elems[x].unique() for x in gt_elems}
                    
                    outputs = {}
                    for task in tasks:
                        if task not in task_gts:
                            continue
                        if len(uniq[task]) == 1 and uniq[task][0] == 255:
                            continue

                        # Forward pass of the mini-batch
                        output = {}
                        features = {}
                        output[task], features[task] = net.forward(inputs, task=task)
                        losses_tasks, losses_dscr, outputs_dscr, grads, task_labels \
                            = net_obj.compute_losses(output, features, criteria_tr, gt_elems, alpha, p)

                        running_loss_ts[task] += losses_tasks[task].item()
                        counter_ts[task] += 1

                        # For logging
                        outputs[task] = output[task].detach()

                        if p['dscr_type']:
                            running_loss_ts_dscr += losses_dscr[task].item()

                            # measure accuracy and record loss for discriminator
                            prec1 = common_configs.accuracy(outputs_dscr[task].data, task_labels[task], topk=(1,))
                            if prec1 != -1:
                                top1_dscr.update(prec1[0].item(), task_labels[task].size(0))

                    # Print stuff
                    if ii % num_img_ts == num_img_ts - 1:
                        print('[Epoch: %d, numTestImages: %5d]' % (epoch, ii + 1))
                        for task in p.TASKS.NAMES:
                            running_loss_ts[task] = running_loss_ts[task] / counter_ts[task]
                            writer.add_scalar('data/test_loss_' + task + '_epoch',
                                              running_loss_ts[task] / p.TASKS.LOSS_MULT[task], epoch)
                            print('Testing Loss %s: %f' % (task, running_loss_ts[task]))
                            running_loss_ts[task] = 0
                            counter_ts[task] = 0
                            # Free the graph
                            losses_tasks = {}

                        if p['dscr_type']:
                            running_loss_ts_dscr = running_loss_ts_dscr / num_img_ts / len(p.TASKS.NAMES)
                            writer.add_scalar('data/test_loss_dscr', running_loss_ts_dscr, epoch)
                            print('Loss Discriminator: %f' % running_loss_ts_dscr)
                            writer.add_scalar('data/test_accuracy_dscr', top1_dscr.avg, epoch)
                            print('Test Accuracy Discriminator: Prec@1 {top1.avg:.3f} Error@1 {error1:.3f}'.format(
                                top1=top1_dscr, error1=100 - top1_dscr.avg))
                            # Free the graph
                            losses_dscr = {}

                        stop_time = timeit.default_timer()
                        print("Execution time: " + str(stop_time - start_time) + "\n")

                    # Log test images to Tensorboard
                    if p.TRAIN.TENS_VIS and ii % p.TRAIN.TENS_VIS_INTER == 0:
                        curr_iter = ii + num_img_tr * epoch
                        tb_vizualizer.visualize_images_tb(writer, sample, outputs,
                                                          global_step=curr_iter, tag=ii, phase='test')

        writer.close()

    # Generate result of the validation images
    net.eval()
    _, _, transforms_infer = config.get_transformations(p)
    for db_name in p['infer_db_names']:

        testloader = config.get_test_loader(p, db_name=db_name, transforms=transforms_infer, infer=True)
        save_dir_res = os.path.join(p['save_dir'], 'Results_' + db_name)

        print('Testing Network')
        # Main Testing Loop
        with torch.no_grad():
            for ii, sample in enumerate(testloader):

                img, meta = sample['image'], sample['meta']

                # Forward of the mini-batch
                inputs = img.cuda()
                task_gts = list(sample.keys())
                net_obj = net.module if isinstance(gpu_id, list) else net
                tasks = net_obj.tasks

                for task in tasks:
                    if task not in task_gts:
                        continue
                    # Forward pass
                    output, _ = net.forward(inputs, task=task)

                    save_dir_task = os.path.join(save_dir_res, task)
                    if not os.path.exists(save_dir_task):
                        os.makedirs(save_dir_task)

                    output = interpolate(output, size=(inputs.size()[-2], inputs.size()[-1]),
                                         mode='bilinear', align_corners=False)

                    if task == 'normals':
                        output = (normal_ize(output) + 1.0) * 255 / 2.0

                    output = output.cpu().data.numpy()

                    for jj in range(int(inputs.size()[0])):

                        # Parameters
                        fname = meta['image'][jj]

                        pred = np.transpose(output[jj, :, :, :], (1, 2, 0))
                        if task == 'normals' or task == 'depth':
                            pass
                        elif p.TASKS.NUM_OUTPUT[task] > 1:
                            pred = np.argmax(pred, axis=2)
                        else:
                            pred = 255 / (1 + np.exp(-pred))

                        pred = np.squeeze(pred)
                        result = cv2.resize(pred, dsize=(meta['im_size'][1][jj], meta['im_size'][0][jj]),
                                            interpolation=p.TASKS.INFER_FLAGVALS[task])

                        if task != 'depth':
                            imageio.imwrite(os.path.join(save_dir_task, fname + '.png'), result.astype(np.uint8))
                        else:
                            sio.savemat(os.path.join(save_dir_task, fname + '.mat'), {'depth': result})


if __name__ == '__main__':
    p = create_config()
    main()

    if p.EVALUATE:
        common_configs.eval_all_results(p)