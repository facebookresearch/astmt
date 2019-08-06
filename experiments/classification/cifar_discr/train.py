import os
import sys
import shutil
import time
import random
import copy
import argparse
from easydict import EasyDict as edict

import torch
import torch.backends.cudnn as cudnn

import fblib.dataloaders.cifar_multitask as dset
import torchvision.transforms as transforms
from fblib.util.classification.utils import AverageMeter, RecorderMeterMultiTask, time_string, convert_secs2time
from fblib.networks import resnext_cifar_multitask as resnext_multitask
from fblib.networks import mobilenet_v2_multitask
from fblib.layers import attention_binary
from fblib.layers import attention
from fblib.util.classification.utils import count_parameters
from fblib.util.mypath import Path

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
    parser.add_argument('--active_tasks', type=int, nargs='+', default="-1",
                        help='Which tasks to train (eg. for training for the second task only in a multi-task setup')
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
    parser.add_argument('--discr_w', type=float, default=1,
                        help='Weight of the discriminator loss')

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

    p.arch = args.arch
    p.n_tasks = args.n_tasks
    p.active_tasks = args.active_tasks
    p.n_outputs = args.n_outputs
    p.epochs = args.epochs
    p.schedule = args.schedule
    p.batch_size = args.batch_size
    p.lr = args.lr
    p.adapters = args.adapters
    p.attention = args.attention
    p.squeeze = args.squeeze

    if args.arch != 'mnetv2':
        p.xpath = args.xpath
        p.width = args.width

    if args.overfit:
        p.overfit = args.overfit

    if args.attention:
        p.binarize = args.binarize
        p.freeze = args.freeze

    if args.decay != 0.0005:
        p.decay = args.decay

    if args.discr_w != 1:
        p.discr_w = args.discr_w

    if not isinstance(args.active_tasks, list):
        args.active_tasks = [args.active_tasks]

    name_args = []
    for x in vars(p):
        attr = getattr(p, x)
        if type(attr) == list:
            attr = [str(x) for x in attr]
            name_args.append(x + '-' + '-'.join(attr))
        else:
            name_args.append(x + '-' + str(attr))
    p.exp_names = '_'.join(name_args)
    args.save_path = os.path.join(Path.exp_dir(), 'classification_cifar_discr', p.exp_names)

    return args


def main():
    args = create_config()
    if args.active_tasks == [-1]:
        args.active_tasks = list(range(args.n_tasks))

    print('Active tasks: {}'.format(args.active_tasks))

    n_aux_tasks = args.n_tasks if not args.use_orig else args.n_tasks - 1

    if args.seed is None:
        args.seed = random.randint(1, 10000)
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
                                          xpath=args.xpath, use_discriminator=True)
    elif args.arch == 'x29':
        net = resnext_multitask.resnext29(base_width=args.width, num_classes=args.n_outputs, n_tasks=n_aux_tasks,
                                          use_orig=args.use_orig, binary_attention=args.binarize,
                                          adapters=args.adapters, attention=args.attention, squeeze=args.squeeze,
                                          xpath=args.xpath)
    elif args.arch == 'mnetv2':
        net = mobilenet_v2_multitask.mobilenet_v2(num_classes=args.n_outputs, n_tasks=n_aux_tasks,
                                                  use_orig=args.use_orig, binary_attention=args.binarize,
                                                  adapters=args.adapters, attention=args.attention,
                                                  squeeze=args.squeeze, use_discriminator=True)
    else:
        raise NotImplementedError

    print_log("=> network :\n {}".format(net), log)
    print_log("\nNumber of parameters (in millions): {0:.3f}\n".format(count_parameters(net) / 1e6), log)

    net = torch.nn.DataParallel(net, device_ids=list(range(args.n_gpu)))

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(net.parameters(), state['lr'], momentum=state['momentum'],
                                weight_decay=state['decay'], nesterov=True)

    if USE_CUDA:
        net.cuda()
        criterion.cuda()

    if args.binarize:
        bin_wrapper = attention_binary.BinaryAttention(model=net, ratio=len(args.active_tasks))
    else:
        bin_wrapper = []

    recorder = RecorderMeterMultiTask(args.epochs, n_tasks=args.n_tasks)
    recorder_discr = RecorderMeterMultiTask(args.epochs, n_tasks=1)

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
        for i_task in range(args.n_tasks):
            if i_task in args.active_tasks:
                print_log(
                    '==>>{:s} [Epoch={:03d}/{:03d}] {:s} [lr={:6.4f}]'.format(
                        time_string(), epoch, args.epochs, need_time, current_learning_rate) \
                    + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(
                        recorder.max_accuracy(False, i_task), 100 - recorder.max_accuracy(False, i_task)), log)
        print_log(
            '==>>{:s} [Epoch={:03d}/{:03d}] {:s} [lr={:6.4f}]'.format(
                time_string(), epoch, args.epochs, need_time, current_learning_rate) \
            + ' [Discriminator Best : Accuracy={:.2f}, Error={:.2f}]'.format(
                recorder_discr.max_accuracy(False, 0), 100 - recorder_discr.max_accuracy(False, 0)), log)

        # train for one epoch
        train_acc, train_los = train(train_loader, net, criterion, optimizer, epoch, log, args, bin_wrapper=bin_wrapper)

        # evaluate on validation set
        # val_acc,   val_los   = extract_features(test_loader, net, criterion, log)
        val_acc, val_los = validate(test_loader, net, criterion, log, args, bin_wrapper)

        is_best = [0] * args.n_tasks
        for i_task in range(args.n_tasks):
            if i_task in args.active_tasks:
                is_best[i_task] = recorder.update(epoch, train_los[i_task], train_acc[i_task], val_los[i_task],
                                                  val_acc[i_task], i_task)

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

        recorder.plot_curve(os.path.join(args.save_path, 'curve.png'))
        _ = recorder_discr.update(epoch, train_los[-1], train_acc[-1], val_los[-1], val_acc[-1], 0)
        recorder_discr.plot_curve(os.path.join(args.save_path, 'curve_discr.png'))

    log.close()


# train function (forward, backward, update)
def train(train_loader, model, criterion, optimizer, epoch, log, args, bin_wrapper):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = [AverageMeter() for i in range(args.n_tasks + 1)]
    top1 = [AverageMeter() for i in range(args.n_tasks + 1)]
    top5 = [AverageMeter() for i in range(args.n_tasks)]

    loss_task = [0] * args.n_tasks
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
        output, discr_output = model(input_var, args.active_tasks)

        loss = 0
        for i_task in range(args.n_tasks):
            if i_task in args.active_tasks:
                # classification loss
                curr_loss = criterion(output[i_task], target_var[i_task])
                loss += curr_loss
                loss_task[i_task] = curr_loss.item()

                # measure accuracy and record loss for classification
                prec1, prec5 = accuracy(output[i_task].data, target_var[i_task], topk=(1, 5))
                losses[i_task].update(loss_task[i_task], input_var[i_task].size(0))
                top1[i_task].update(prec1.item(), input_var[i_task].size(0))
                top5[i_task].update(prec5.item(), input_var[i_task].size(0))

        # discriminator loss
        discr_output = torch.cat(discr_output)
        task_labels = torch.cat(task_labels)
        loss_discr = criterion(discr_output, task_labels)
        loss += args.discr_w * loss_discr

        # measure accuracy and record loss for discriminator
        prec1 = accuracy(discr_output.data, task_labels, topk=(1,))
        losses[-1].update(loss_discr.item(), task_labels.size(0))
        top1[-1].update(prec1[0].item(), task_labels.size(0))

        optimizer.zero_grad()
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
            for i_task in range(args.n_tasks):
                if i_task in args.active_tasks:
                    print_log('  Task: {} Epoch: [{:03d}][{:03d}/{:03d}]   '
                              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                              'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
                              'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                              'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
                        i_task, epoch, i, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses[i_task], top1=top1[i_task], top5=top5[i_task])
                              + time_string(), log)

    for i_task in range(args.n_tasks):
        if i_task in args.active_tasks:
            print_log(
                '  **Train** Task: {} Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(
                    i_task, top1=top1[i_task], top5=top5[i_task], error1=100 - top1[i_task].avg), log)
    print_log(
        '  **Train** Discriminator: Prec@1 {top1.avg:.3f} Error@1 {error1:.3f}'.format(
            top1=top1[-1], error1=100 - top1[-1].avg), log)

    return [x.avg for x in top1], [x.avg for x in losses]


def validate(val_loader, model, criterion, log, args, bin_wrapper=[]):
    losses = [AverageMeter() for i in range(args.n_tasks + 1)]
    top1 = [AverageMeter() for i in range(args.n_tasks + 1)]
    top5 = [AverageMeter() for i in range(args.n_tasks)]

    loss_task = [0] * args.n_tasks
    # switch to evaluate mode
    model.eval()

    if args.binarize and not bin_wrapper.frozen:
        bin_wrapper.binarize(deterministic=True)

    with torch.no_grad():
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
            output, discr_output = model(input_var, args.active_tasks)
            for i_task in range(args.n_tasks):
                if i_task in args.active_tasks:
                    curr_loss = criterion(output[i_task], target_var[i_task])
                    loss_task[i_task] = curr_loss.item()

                    # measure accuracy and record loss
                    prec1, prec5 = accuracy(output[i_task].data, target_var[i_task], topk=(1, 5))
                    losses[i_task].update(loss_task[i_task], input_var[i_task].size(0))
                    top1[i_task].update(prec1.item(), input_var[i_task].size(0))
                    top5[i_task].update(prec5.item(), input_var[i_task].size(0))

            # discriminator loss
            discr_output = torch.cat(discr_output)
            task_labels = torch.cat(task_labels)
            loss_discr = criterion(discr_output, task_labels)

            # measure accuracy and record loss for discriminator
            prec1 = accuracy(discr_output.data, task_labels, topk=(1,))
            losses[-1].update(loss_discr.item(), task_labels.size(0))
            top1[-1].update(prec1[0].item(), task_labels.size(0))

    if args.binarize and not bin_wrapper.frozen:
        bin_wrapper.restore()

    for i_task in range(args.n_tasks):
        if i_task in args.active_tasks:
            print_log('  **Test** Task: {} Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(
                i_task, top1=top1[i_task], top5=top5[i_task], error1=100 - top1[i_task].avg), log)
    print_log('  **Test** Discriminator: Prec@1 {top1.avg:.3f} Error@1 {error1:.3f}'.format(
        top1=top1[-1], error1=100 - top1[-1].avg), log)

    return [x.avg for x in top1], [x.avg for x in losses]


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
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr
    assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
    for (gamma, step) in zip(gammas, schedule):
        if epoch >= step:
            lr = lr * gamma
        else:
            break
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
