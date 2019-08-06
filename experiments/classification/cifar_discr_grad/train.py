import os
import sys
import shutil
import time
import random
import copy
import argparse
import glob
import numpy as np
from easydict import EasyDict as edict
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import torch
import torch.backends.cudnn as cudnn

import fblib.dataloaders.cifar_multitask as dset
import torchvision.transforms as transforms
from fblib.util.classification.utils import AverageMeter, RecorderMeterMultiTask, time_string, convert_secs2time
from fblib.networks import resnext_cifar_multitask_grad as resnext_multitask  # TODO: Change to resnext_multitask when merging done
from fblib.networks import mobilenet_v2_multitask
from fblib.layers import attention_binary
from fblib.layers import attention
from fblib.util.classification.utils import count_parameters
from fblib.util.mypath import Path
import fblib.util.tsne.tsne as tsne

USE_CUDA = torch.cuda.is_available()


def parse_args():
    def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1")

    parser = argparse.ArgumentParser(description='Trains ResNeXt on CIFAR or ImageNet',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar10', 'cifar100'],
                        help='Choose between Cifar10/100 and ImageNet.')
    parser.add_argument('--arch', metavar='ARCH', default='x20', choices=['x20', 'x29', "mnetv2"],
                        help='model architecture: (default: resnext20_8_64)')
    parser.add_argument('--n_tasks', type=int, default=None,
                        help='Number of tasks')
    parser.add_argument('--n_outputs', type=int, default=100,
                        help='How many classes per task?')
    parser.add_argument('--mix', type=str2bool, default=True,
                        help='Mix labels among tasks, so that they become conflicting')
    parser.add_argument('--use_orig', type=str2bool, default=True,
                        help='keep the original task (as the 0-th task)')
    parser.add_argument('--width', type=int, default=64,
                        help='base width of ResNeXt modules (default 64)')

    # Task-specific options
    parser.add_argument('--adapters', type=str2bool, default=False,
                        help='Use per-task residual adapters?')
    parser.add_argument('--attention', type=str2bool, default=False,
                        help='Use per-task multiplicative attention layers?')
    parser.add_argument('--squeeze', type=str2bool, default=False,
                        help='Use per-task squeeze and excitation layers?')
    parser.add_argument('--xpath', type=str2bool, default=False,
                        help='Task-specific ResNeXt path')
    parser.add_argument('--binarize', type=str2bool, default=False,
                        help='Use binary weights for attention module?')
    parser.add_argument('--freeze', type=str2bool, default=False,
                        help='freeze attention module weights after first lr jump?')
    parser.add_argument('--main_w', type=float, default=1,
                        help='Weight of the loss for the main task')
    parser.add_argument('--discr_w', type=float, default=1,
                        help='Weight of the discriminator loss')
    parser.add_argument('--reverse', type=str2bool, default=False,
                        help='Reverse gradient for adversarial training')
    parser.add_argument('--onlynorm', type=str2bool, default=False,
                        help='Put only the norm of the gradient into the discriminator')

    # Optimization options
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size.')
    parser.add_argument('-lr', type=float, default=0.1,
                        help='The Learning Rate.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum.')
    parser.add_argument('--decay', type=float, default=0.0005,
                        help='Weight decay (L2 penalty).')
    parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1],
                        help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
    parser.add_argument('--cosine', type=str2bool, default=False)

    # Checkpoints
    parser.add_argument('--print_freq', default=200, type=int, metavar='N',
                        help='print frequency (default: 200)')
    parser.add_argument('--save_path', type=str, default='./',
                        help='Folder to save checkpoints and log.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--tsne', default=False,
                        help='store tsne visualization results?')
    # Acceleration
    parser.add_argument('--n_gpu', type=int, default=1,
                        help='0 = CPU.')
    parser.add_argument('--workers', type=int, default=8,
                        help='number of data loading workers (default: 2)')
    # random seed
    parser.add_argument('--seed', type=int, default=0,
                        help='manual seed')

    # overfitting
    parser.add_argument('--overfit', type=str2bool, default=False,
                        help='True to overfit on small number of samples')

    return parser.parse_args()


def create_config():
    p = edict()

    args = parse_args()
    print('\nThis script was run with the following parameters:')
    for x in vars(args):
        print('{}: {}'.format(x, str(getattr(args, x))))

    if args.onlynorm:
        p.onlynorm = args.onlynorm

    if args.arch != 'x20':
        p.arch = args.arch

    p.n_tasks = args.n_tasks
    p.adapters = args.adapters
    p.attention = args.attention
    p.squeeze = args.squeeze

    if args.arch != 'mnetv2':
        if args.xpath:
            p.xpath = args.xpath
        if args.width != 64:
            p.width = args.width

    if args.epochs != 150:
        p.epochs = args.epochs

    if args.lr != 0.1:
        p.lr = args.lr

    if args.n_outputs != 10:
        p.n_outputs = args.n_outputs

    if args.batch_size != 64:
        p.batch_size = args.batch_size

    if args.overfit:
        p.overfit = args.overfit

    if args.attention:
        p.binarize = args.binarize
        p.freeze = args.freeze

    if args.decay != 0.0005:
        p.decay = args.decay

    if args.discr_w != 1:
        p.discr_w = args.discr_w

    if args.main_w != 1:
        p.main_w = args.main_w

    if args.reverse:
        p.reverse = args.reverse

    if args.cosine:
        p.cosine = args.cosine
    else:
        p.schedule = args.schedule

    name_args = []
    for x in vars(p):
        attr = getattr(p, x)
        if type(attr) == list:
            attr = [str(x) for x in attr]
            name_args.append(x + '-' + '-'.join(attr))
        else:
            name_args.append(x + '-' + str(attr))
    p.exp_names = '_'.join(name_args)
    save_path = os.path.join(Path.exp_dir(), 'classification_cifar_adversarial_cos', p.exp_names)

    runs = sorted(glob.glob(os.path.join(save_path, 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0
    args.save_path = os.path.join(save_path, 'run_' + str(run_id))

    if args.main_w == 1:
        args.main_w = [1.] * args.n_tasks
    else:
        tmp = [args.main_w] + [1.] * (args.n_tasks - 1)
        args.main_w = [x / (sum(tmp)/args.n_tasks) for x in tmp]

    if args.seed is None:
        args.seed = random.randint(1, 10000)

    return args


def main():
    args = create_config()

    n_aux_tasks = args.n_tasks if not args.use_orig else args.n_tasks - 1

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if USE_CUDA:
        torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True

    # Init logger

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    log = open(os.path.join(args.save_path, 'log_seed_{}.txt'.format(args.seed)), 'w')
    print_log('save path : {}'.format(args.save_path), log)
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)
    print_log("Random Seed: {}".format(args.seed), log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("torch  version : {}".format(torch.__version__), log)
    print_log("cudnn  version : {}".format(torch.backends.cudnn.version()), log)

    if args.dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif args.dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    else:
        assert False, "Unknow dataset : {}".format(args.dataset)

    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
         transforms.Normalize(mean, std)])
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)])

    if args.dataset == 'cifar100':
        train_data = dset.CIFAR100(train=True, transform=train_transform, download=True, use_orig=args.use_orig,
                                   n_tasks=n_aux_tasks, n_outputs=args.n_outputs, mix_labels=args.mix,
                                   use_task_labels=True, overfit=args.overfit)
        test_data = dset.CIFAR100(train=False, transform=test_transform, download=True, use_orig=args.use_orig,
                                  n_tasks=n_aux_tasks, n_outputs=args.n_outputs, mix_labels=args.mix,
                                  use_task_labels=True, overfit=args.overfit)
    else:
        assert False, 'Do not support dataset : {}'.format(args.dataset)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)

    print_log("=> creating model '{}'".format(args.arch), log)
    # Init model, criterion, and optimizer
    if args.arch == 'x20':
        net = resnext_multitask.resnext20(base_width=args.width, num_classes=args.n_outputs, n_tasks=n_aux_tasks,
                                          use_orig=args.use_orig, binary_attention=args.binarize,
                                          adapters=args.adapters, attention=args.attention, squeeze=args.squeeze,
                                          xpath=args.xpath, ret_features=True, n_gpu=args.n_gpu,
                                          reverse_grad=args.reverse, onlynorm=args.onlynorm)
    elif args.arch == 'x29':
        net = resnext_multitask.resnext29(base_width=args.width, num_classes=args.n_outputs, n_tasks=n_aux_tasks,
                                          use_orig=args.use_orig, binary_attention=args.binarize,
                                          adapters=args.adapters, attention=args.attention, squeeze=args.squeeze,
                                          xpath=args.xpath)
    elif args.arch == 'mnetv2':
        net = mobilenet_v2_multitask.mobilenet_v2(num_classes=args.n_outputs, n_tasks=n_aux_tasks,
                                                  use_orig=args.use_orig, binary_attention=args.binarize,
                                                  adapters=args.adapters, attention=args.attention,
                                                  squeeze=args.squeeze, ret_features=True)
    else:
        raise NotImplementedError

    print_log("=> network :\n {}".format(net), log)
    print_log("\nNumber of parameters (in millions): {0:.3f}\n".format(count_parameters(net) / 1e6), log)

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(net.parameters(), state['lr'], momentum=state['momentum'],
                                weight_decay=state['decay'], nesterov=True)

    if USE_CUDA:
        net.cuda()
        criterion.cuda()

    if args.binarize:
        bin_wrapper = attention_binary.BinaryAttention(model=net, ratio=args.n_tasks)
    else:
        bin_wrapper = []

    # Recorders for plotting and logging
    recorder = RecorderMeterMultiTask(args.epochs, n_tasks=args.n_tasks)
    recorder_discr = RecorderMeterMultiTask(args.epochs, n_tasks=args.n_tasks)
    recorder_gradnorm = RecorderMeterMultiTask(args.epochs, n_tasks=args.n_tasks)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print_log("=> loading checkpoint '{}'".format(args.resume), log)
            checkpoint = torch.load(args.resume)
            recorder = checkpoint['recorder']
            args.start_epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print_log("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']), log)
        else:
            raise ValueError("=> no checkpoint found at '{}'".format(args.resume))
    else:
        print_log("=> do not use any checkpoint for {} model".format(args.arch), log)

    if args.evaluate:
        validate(test_loader, net, criterion, log, bin_wrapper)
        return

    # Main loop
    start_time = time.time()
    epoch_time = AverageMeter()
    for epoch in range(args.start_epoch, args.epochs):
        current_learning_rate = adjust_learning_rate(optimizer, epoch, args.gammas, args.schedule, args)

        if args.binarize and args.freeze:
            optimizer = check_and_freeze_binary_attention_weights(epoch=epoch, schedule=args.schedule[0],
                                                                  bin_wrapper=bin_wrapper, model=net,
                                                                  optimizer=optimizer)
        elif args.freeze:
            optimizer = check_and_freeze_attention_weights(epoch=epoch, schedule=args.schedule[0],
                                                           model=net, optimizer=optimizer)

        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
        for task in range(args.n_tasks):
            print_log(
                '==>>{:s} [Epoch={:03d}/{:03d}] {:s} [lr={:6.4f}]'.format(
                    time_string(), epoch, args.epochs, need_time, current_learning_rate) \
                + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(
                    recorder.max_accuracy(False, task), 100 - recorder.max_accuracy(False, task)), log)
            print_log(
                '==>>{:s} [Epoch={:03d}/{:03d}] {:s} [lr={:6.4f}]'.format(
                    time_string(), epoch, args.epochs, need_time, current_learning_rate) \
                + ' [Discriminator Best : Accuracy={:.2f}, Error={:.2f}]'.format(
                    recorder_discr.max_accuracy(False, task), 100 - recorder_discr.max_accuracy(False, task)), log)

        # train for one epoch
        if args.reverse:
            alpha = 2. / (1. + np.exp(-10 * ((epoch + 1) / args.epochs))) - 1  # Ganin et al for gradient reversal
            alpha = min(1, 2 * alpha)
        else:
            alpha = 0.

        dic_train = train(train_loader, net, criterion, optimizer, epoch, log, args,
                                                       bin_wrapper=bin_wrapper, alpha=alpha)
        train_acc = dic_train['top1']
        train_los = dic_train['losses']
        train_acc_discr = dic_train['top1_discr']
        train_los_discr = dic_train['losses_discr_doc']
        train_grad_norms = dic_train['gradnorms']

        # evaluate on validation set
        # val_acc,   val_los   = extract_features(test_loader, net, criterion, log)
        dic_val = validate(test_loader, net, criterion, log, args, bin_wrapper)

        val_acc = dic_val['top1']
        val_los = dic_val['losses']
        val_acc_discr = dic_val['top1_discr']
        val_los_discr = dic_val['losses_discr_doc']
        val_grad_norms = dic_val['gradnorms']

        is_best = [0] * args.n_tasks
        for task in range(args.n_tasks):

            # Update Recorders to store figures
            is_best[task] = recorder.update(epoch, train_los[task], train_acc[task], val_los[task],
                                              val_acc[task], task)

            _ = recorder_discr.update(epoch, train_los_discr[task], train_acc_discr[task],
                                      val_los_discr[task], val_acc_discr[task], task)
            _ = recorder_gradnorm.update(epoch, 0, 2000 * train_grad_norms[task],
                                         0, 2000 * val_grad_norms[task],  task)

            # Store the best accuracy for the main task
            if task == 0:
                fname = os.path.join(args.save_path, 'best.txt')
                with open(fname, 'w') as f:
                    f.write('{}'.format(recorder.max_accuracy(False, task)))

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': net.state_dict(),
                'recorder': recorder,
                'optimizer': optimizer.state_dict(),
                'args': copy.deepcopy(args),
            }, is_best, args.save_path, 'checkpoint.pth.tar')

            # measure elapsed time
            epoch_time.update(time.time() - start_time)
            start_time = time.time()

        recorder.plot_curve(os.path.join(args.save_path, 'Accuracy.png'))
        recorder_discr.plot_curve(os.path.join(args.save_path, 'Discriminator.png'))
        recorder_gradnorm.plot_curve(os.path.join(args.save_path, 'Gradnorms.png'),
                                     labels=['null', 'train-grad-norm-x2000', 'null', 'val-grad-norm-x2000'])

    log.close()

    if args.tsne:
        store_tsne(args, net, test_loader, criterion, bin_wrapper)


# train function (forward, backward, update)
def train(train_loader, model, criterion, optimizer, epoch, log, args, bin_wrapper, alpha=0):
    print('Value of alpha: {}'.format(alpha))
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = [AverageMeter() for i in range(args.n_tasks)]
    top1 = [AverageMeter() for i in range(args.n_tasks)]
    top5 = [AverageMeter() for i in range(args.n_tasks)]

    losses_discr_doc = [AverageMeter() for i in range(args.n_tasks)]
    top1_discr = [AverageMeter() for i in range(args.n_tasks)]
    gradnorms = [AverageMeter() for i in range(args.n_tasks)]

    # switch to train mode
    model.train()

    end = time.time()
    for i, sample in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input_var = sample[0]
        target_var = sample[1]
        task_labels = sample[2]

        input_var = input_var.requires_grad_()

        if USE_CUDA:
            if isinstance(target_var, list):
                target_var = [x.cuda(non_blocking=True) for x in target_var]
                task_labels = [x.cuda(non_blocking=True) for x in task_labels]
            else:
                target_var = target_var.cuda(non_blocking=True)
                task_labels = task_labels.cuda(non_blocking=True)
            input_var = input_var.cuda()

        if args.binarize and not bin_wrapper.frozen:
            bin_wrapper.binarize()

        # compute output
        output, features = model(input_var, list(range(args.n_tasks)))
        losses_tasks, losses_discr, outputs_discr, grads\
            = model.compute_losses_discr(output, features, criterion, target_var, task_labels, alpha, args)

        loss_tasks = 0
        loss_discr = 0
        for task in range(args.n_tasks):
            loss_tasks += losses_tasks[task]
            loss_discr += losses_discr[task]

            # measure accuracy and record loss for classification
            prec1, prec5 = accuracy(output[task].data, target_var[task], topk=(1, 5))
            losses[task].update(losses_tasks[task].item() / args.main_w[task], input_var[task].size(0))
            top1[task].update(prec1.item(), input_var[task].size(0))
            top5[task].update(prec5.item(), input_var[task].size(0))

            # Update grad norm logger
            mean_grad_norm = (grads[task].detach().norm(p=2, dim=1) + 1e-10).mean()
            gradnorms[task].update(mean_grad_norm.item(), grads[task].size(0))

            # measure accuracy and record loss for discriminator
            prec1 = accuracy(outputs_discr[task].data, task_labels[task], topk=(1,))
            losses_discr_doc[task].update(losses_discr[task].item(), task_labels[task].size(0))
            top1_discr[task].update(prec1[0].item(), task_labels[task].size(0))

        optimizer.zero_grad()
        loss = (1 - args.discr_w) * loss_tasks + args.discr_w * loss_discr
        loss.backward()  # backward for both discriminator and classification

        if args.binarize and not bin_wrapper.frozen:
            bin_wrapper.restore()

        optimizer.step()

        if args.binarize and not bin_wrapper.frozen:
            bin_wrapper.clip()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            for task in range(args.n_tasks):
                print_log('  Task: {} Epoch: [{:03d}][{:03d}/{:03d}]   '
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
                          'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
                    task, epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses[task], top1=top1[task], top5=top5[task])
                          + time_string(), log)

    for task in range(args.n_tasks):
        print_log(
            '  **Train** Task: {} Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(
                task, top1=top1[task], top5=top5[task], error1=100 - top1[task].avg), log)
        print_log(
            '  **Train** Task: {} Discriminator: Prec@1 {top1.avg:.3f} Error@1 {error1:.3f}'.format(
                task, top1=top1_discr[task], error1=100 - top1_discr[task].avg), log)

    dic_train = {'top1': [x.avg for x in top1],
                 'losses': [x.avg for x in losses],
                 'top1_discr': [x.avg for x in top1_discr],
                 'losses_discr_doc': [x.avg for x in losses_discr_doc],
                 'gradnorms': [x.avg for x in gradnorms]}

    return dic_train


def validate(val_loader, model, criterion, log, args, bin_wrapper=[], alpha=0):
    losses = [AverageMeter() for i in range(args.n_tasks)]
    top1 = [AverageMeter() for i in range(args.n_tasks)]
    top5 = [AverageMeter() for i in range(args.n_tasks)]

    losses_discr_doc = [AverageMeter() for i in range(args.n_tasks)]
    top1_discr = [AverageMeter() for i in range(args.n_tasks)]
    gradnorms = [AverageMeter() for i in range(args.n_tasks)]

    # switch to evaluate mode
    model.eval()

    if args.binarize and not bin_wrapper.frozen:
        bin_wrapper.binarize(deterministic=True)

    for i, sample in enumerate(val_loader):

        input_var = sample[0]
        target_var = sample[1]
        task_labels = sample[2]

        if USE_CUDA:
            if isinstance(target_var, list):
                target_var = [x.cuda(non_blocking=True) for x in target_var]
                task_labels = [x.cuda(non_blocking=True) for x in task_labels]
            else:
                target_var = target_var.cuda(non_blocking=True)
                task_labels = task_labels.cuda(non_blocking=True)
            input_var = input_var.cuda()

        # compute output
        output, features = model(input_var, list(range(args.n_tasks)), train=False)
        losses_tasks, losses_discr, outputs_discr, grads \
            = model.compute_losses_discr(output, features, criterion, target_var, task_labels, alpha, args)

        for task in range(args.n_tasks):

            # measure accuracy and record loss for classification
            prec1, prec5 = accuracy(output[task].data, target_var[task], topk=(1, 5))
            losses[task].update(losses_tasks[task].item() / args.main_w[task], input_var[task].size(0))
            top1[task].update(prec1.item(), input_var[task].size(0))
            top5[task].update(prec5.item(), input_var[task].size(0))

            # Update grad norm logger
            mean_grad_norm = (grads[task].detach().norm(p=2, dim=1) + 1e-10).mean()
            gradnorms[task].update(mean_grad_norm.item(), grads[task].size(0))

            # measure accuracy and record loss for discriminator
            prec1 = accuracy(outputs_discr[task].data, task_labels[task], topk=(1,))
            losses_discr_doc[task].update(losses_discr[task].item(), task_labels[task].size(0))
            top1_discr[task].update(prec1[0].item(), task_labels[task].size(0))

    if args.binarize and not bin_wrapper.frozen:
        bin_wrapper.restore()

    for task in range(args.n_tasks):
        print_log('  **Test** Task: {} Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(
            task, top1=top1[task], top5=top5[task], error1=100 - top1[task].avg), log)
        print_log('  **Test** Task: {} Discriminator: Prec@1 {top1.avg:.3f} Error@1 {error1:.3f}'.format(
            task, top1=top1_discr[task], error1=100 - top1_discr[task].avg), log)

    dic_val = {'top1': [x.avg for x in top1],
               'losses': [x.avg for x in losses],
               'top1_discr': [x.avg for x in top1_discr],
               'losses_discr_doc': [x.avg for x in losses_discr_doc],
               'gradnorms': [x.avg for x in gradnorms]}

    return dic_val


def store_tsne(args, model, val_loader, criterion, bin_wrapper):
    print('Starting Processing for tsne visualization')
    # switch to evaluate mode
    model.eval()

    if args.binarize and not bin_wrapper.frozen:
        bin_wrapper.binarize(deterministic=True)

    feats = []
    labels = []
    for i, sample in enumerate(val_loader):

        input_var = sample[0]
        target_var = sample[1]
        task_labels = sample[2]

        if USE_CUDA:
            if isinstance(target_var, list):
                target_var = [x.cuda(non_blocking=True) for x in target_var]
                task_labels = [x.cuda(non_blocking=True) for x in task_labels]
            else:
                target_var = target_var.cuda(non_blocking=True)
                task_labels = task_labels.cuda(non_blocking=True)
            input_var = input_var.cuda()

        # compute output
        output, features = model(input_var, list(range(args.n_tasks)), train=False)
        _, _, _, grads = model.compute_losses_discr(output, features, criterion, target_var, task_labels, 0, args)

        grads = [x.detach().cpu().numpy() for x in grads]
        task_labels = [x.detach().cpu().numpy() for x in task_labels]
        for task in range(args.n_tasks):
            feats.extend(grads[task])
            labels.extend(task_labels[task])

        if len(feats) > 10000:
            break

    x = np.array(feats)
    y = tsne.tsne(x,  no_dims=2, initial_dims=50, perplexity=30.0, max_iter=500)
    fig = plt.figure()
    plt.scatter(y[:, 0], y[:, 1], 20, labels)

    fig.savefig(os.path.join(args.save_path, 'Tsne.png'),  dpi=100, bbox_inches='tight')
    print('---- save figure into {}'.format(os.path.join(args.save_path, 'Tsne.png')))


def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()


def save_checkpoint(state, is_best, save_path, filename):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:
        bestname = os.path.join(save_path, 'model_best.pth.tar')
        shutil.copyfile(filename, bestname)


def adjust_learning_rate(optimizer, epoch, gammas, schedule, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs if args.cosine==False"""
    if not args.cosine:
        lr = args.lr
        assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
        for (gamma, step) in zip(gammas, schedule):
            if epoch >= step:
                lr = lr * gamma
            else:
                break
    else:
        lr = args.lr * np.cos(np.pi / 2 * epoch / args.epochs)
        print('Adjusted Cosine Learning rate: {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def check_and_freeze_binary_attention_weights(epoch, schedule, bin_wrapper, model, optimizer):
    if epoch == schedule:
        print("Freezing learned attention weights!")
        bin_wrapper.binarize(deterministic=True, freeze=True, nonzero=True)

        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=optimizer.defaults['lr'],
                                    momentum=optimizer.defaults['momentum'],
                                    weight_decay=optimizer.defaults['weight_decay'],
                                    nesterov=optimizer.defaults['nesterov'])
    return optimizer


def check_and_freeze_attention_weights(epoch, schedule, model, optimizer):
    if epoch == schedule:
        for m in model.modules():
            if isinstance(m, attention.AttentionModule):
                m.weight.requires_grad = False
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=optimizer.defaults['lr'],
                                    momentum=optimizer.defaults['momentum'],
                                    weight_decay=optimizer.defaults['weight_decay'],
                                    nesterov=optimizer.defaults['nesterov'])

    return optimizer


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
