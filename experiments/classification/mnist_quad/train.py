from __future__ import division

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

import fblib.dataloaders.mnist_multitask as dset
import torchvision.transforms as transforms
from fblib.util.classification.utils import AverageMeter, RecorderMeterMultiTask, time_string, convert_secs2time
from fblib.networks import tiny_nets_multiclass
from fblib.util.mypath import Path
from fblib.util.optimizer_mtl.qp import return_optimal_gradients
from fblib.util.optimizer_mtl.select_used_modules import make_closure
from fblib.util.model_resources.num_parameters import count_parameters


USE_CUDA = torch.cuda.is_available()


def parse_args():
    def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1")

    parser = argparse.ArgumentParser(description='Trains LeNet on MultiMNIST',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist'],
                        help='Choose dataset.')
    parser.add_argument('--n_tasks', type=int, default=2)
    parser.add_argument('--active_tasks', type=int, nargs='+', default=[0, 1],
                        help='Which tasks to train (eg. for training for the second task only in a multi-task setup')
    parser.add_argument('--quad', type=str2bool, default=False,
                        help='use quadratic optimization?')
    parser.add_argument('--newop', type=str2bool, default=True,
                        help='Modified optimization process due to weight decay and momentum bug')
    parser.add_argument('--lr_gen', type=float, default=-1,
                        help='Learning rate multiplier for generic layers (apples to apples is -1)')
    parser.add_argument('--squeeze', type=str2bool, default=False,
                        help='Squeeze and Excitation module')
    parser.add_argument('--use_bn', type=str2bool, default=False,
                        help='Use batchnorm layers?')
    parser.add_argument('--alphas', type=str2bool, default=False)

    # Optimization options
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size.')
    parser.add_argument('-lr', type=float, default=0.02,
                        help='The Learning Rate.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum.')
    parser.add_argument('--decay', type=float, default=0.00005,
                        help='Weight decay (L2 penalty).')
    parser.add_argument('--schedule', type=int, nargs='+', default=[15, 23],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1],
                        help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
    parser.add_argument('--save_checkpoint', type=str2bool, default=False)

    # Checkpoints
    parser.add_argument('--print_freq', default=200, type=int, metavar='N',
                        help='print frequency (default: 200)')
    parser.add_argument('--save_path', type=str, default='./',
                        help='Folder to save checkpoints and log.')

    # Acceleration
    parser.add_argument('--n_gpu', type=int, default=1,
                        help='0 = CPU.')
    parser.add_argument('--workers', type=int, default=4,
                        help='number of data loading workers (default: 2)')
    # random seed
    parser.add_argument('--seed', type=int, default=0,
                        help='manual seed')

    return parser.parse_args()


def create_config():

    p = edict()

    args = parse_args()
    print('\nThis script was run with the following parameters:')
    for x in vars(args):
        print('{}: {}'.format(x, str(getattr(args, x))))

    if args.lr_gen < 0:
        args.lr_gen = (1 / len(args.active_tasks)) if not args.quad else 1
    else:
        p.lr_gen = args.lr_gen

    if args.squeeze:
        p.squeeze = args.squeeze
    if args.use_bn:
        p.use_bn = args.use_bn

    p.active_tasks = args.active_tasks
    p.epochs = args.epochs
    p.schedule = args.schedule
    p.batch_size = args.batch_size
    p.lr = args.lr
    p.quad = args.quad
    p.newop = args.newop
    p.alphas = args.alphas

    name_args = []
    for x in vars(p):
        attr = getattr(p, x)
        if type(attr) == list:
            attr = [str(x) for x in attr]
            name_args.append(x + '-' + '-'.join(attr))
        else:
            name_args.append(x + '-' + str(attr))
    p.exp_names = '_'.join(name_args)
    args.save_path = os.path.join(Path.exp_dir(), 'mnist_quad', p.exp_names)

    return args


def main():

    args = create_config()

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

    train_transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(),
                                          transforms.Normalize((0.5,), (1.0,))])
    test_transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(),
                                         transforms.Normalize((0.5,), (1.0,))])

    if args.dataset == 'mnist':
        train_data = dset.MultiMNIST(train=True, transform=train_transform, download=True,)
        test_data = dset.MultiMNIST(train=False, transform=test_transform, download=True)
    else:
        assert False, 'Do not support dataset : {}'.format(args.dataset)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)

    print_log("=> creating model '{}'".format('LeNet'), log)

    # Init model, criterion, and optimizer
    net = tiny_nets_multiclass.LeNetMT(n_tasks=2, squeeze=args.squeeze, use_bn=args.use_bn)

    print_log("=> network :\n {}".format(net), log)
    print('\nNumber of parameters (in thousands): {0:.3f}'.format(count_parameters(net) / 1e3))

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss()

    optimizer_gen = torch.optim.SGD(net.get_lr_params(part='generic'), state['lr'] * args.lr_gen,
                                    momentum=state['momentum'], weight_decay=state['decay'], nesterov=True)
    optimizer_task = torch.optim.SGD(net.get_lr_params(part='task_specific'), state['lr'],
                                     momentum=state['momentum'], weight_decay=state['decay'], nesterov=True)

    if USE_CUDA:
        net.cuda()
        criterion.cuda()

    recorder = RecorderMeterMultiTask(args.epochs, n_tasks=args.n_tasks)

    # Main loop
    start_time = time.time()
    epoch_time = AverageMeter()
    for epoch in range(0, args.epochs):
        _ = adjust_learning_rate(optimizer_gen, epoch, args.gammas, args.schedule, args, True)
        current_learning_rate = adjust_learning_rate(optimizer_task, epoch, args.gammas, args.schedule, args, False)

        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
        for i_task in range(args.n_tasks):
            if i_task in args.active_tasks:
                print_log(
                    '==>>{:s} [Epoch={:03d}/{:03d}] {:s} [lr={:6.4f}]'.format(
                        time_string(), epoch, args.epochs, need_time, current_learning_rate) \
                    + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(
                        recorder.max_accuracy(False, i_task), 100 - recorder.max_accuracy(False, i_task)), log)

        # train for one epoch
        train_acc, train_los = train(train_loader, net, criterion, optimizer_gen, optimizer_task, epoch, log, args)

        # evaluate on validation set
        # val_acc,   val_los   = extract_features(test_loader, net, criterion, log)
        val_acc, val_los = validate(test_loader, net, criterion, log, args)

        is_best = [0] * args.n_tasks
        for i_task in range(args.n_tasks):
            if i_task in args.active_tasks:
                is_best[i_task] = recorder.update(epoch, train_los[i_task], train_acc[i_task],
                                                  val_los[i_task], val_acc[i_task], i_task)

            # measure elapsed time
            epoch_time.update(time.time() - start_time)
            start_time = time.time()

        if args.save_checkpoint:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'recorder': recorder,
                'optimizer_gen': optimizer_gen.state_dict(),
                'optimizer_task': optimizer_task.state_dict(),
                'args': copy.deepcopy(args),
            }, is_best, args.save_path, 'checkpoint.pth.tar')

        recorder.plot_curve(os.path.join(args.save_path, 'curve.png'))

    log.close()


# train function (forward, backward, update)
def train(train_loader, model, criterion, optimizer_gen, optimizer_task, epoch, log, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = [AverageMeter() for i in range(args.n_tasks)]
    top1 = [AverageMeter() for i in range(args.n_tasks)]

    # switch to train mode
    model.train()

    end = time.time()
    for i, sample in enumerate(train_loader):
        generic_param_grads = {}
        # measure data loading time
        data_time.update(time.time() - end)
        input_var = sample[0].requires_grad_()
        target_vars = sample[1:]
        if USE_CUDA:
            target_vars = [x.cuda(non_blocking=True) for x in target_vars]
            input_var = input_var.cuda()

        # compute output
        for i_task in range(args.n_tasks):
            if i_task in args.active_tasks:
                # print(i_task)
                output = model(input_var, task=i_task)
                curr_loss = criterion(output, target_vars[i_task])
                loss = curr_loss
                loss_task = curr_loss.item()

                # measure accuracy and record loss
                prec1 = accuracy(output.data, target_vars[i_task], topk=(1,))
                losses[i_task].update(loss_task, input_var.size(0))
                top1[i_task].update(prec1.item(), input_var.size(0))

                optimizer_gen.zero_grad()
                optimizer_task.zero_grad()
                loss.backward()

                if not args.quad:
                    optimizer_gen.step()
                    optimizer_task.step()
                else:
                    generic_params = model.get_param_dict(part='generic')
                    tmp = []
                    for name in generic_params:
                        tmp.append(generic_params[name].grad.detach().clone())
                    generic_param_grads[i_task] = tmp

        if args.quad:
            if not args.alphas:
                optimal_grads = return_optimal_gradients(generic_param_grads)
            else:
                optimal_grads = return_optimal_gradients(generic_param_grads, [0.5, 0.5])

            # compute output
            for i_task in range(args.n_tasks):
                if i_task in args.active_tasks:
                    # print(i_task)
                    output = model(input_var, task=i_task)
                    curr_loss = criterion(output, target_vars[i_task])
                    loss = curr_loss
                    loss_task = curr_loss.item()

                    # measure accuracy and record loss
                    prec1 = accuracy(output.data, target_vars[i_task], topk=(1,))
                    losses[i_task].update(loss_task, input_var.size(0))
                    top1[i_task].update(prec1.item(), input_var.size(0))

                    optimizer_task.zero_grad()
                    if not args.newop:
                        loss.backward()
                        optimizer_task.step()
                    else:
                        optimizer_task.step(closure=make_closure(loss=loss, net=model))

            optimizer_gen.zero_grad()
            for jj, name in enumerate(generic_params):
                assert (generic_params[name].grad.data.shape ==
                        optimal_grads[jj].data.shape)
                generic_params[name].grad.data = optimal_grads[jj].data
            optimizer_gen.step()

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
                              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '.format(
                               i_task, epoch, i, len(train_loader), batch_time=batch_time,
                               data_time=data_time, loss=losses[i_task], top1=top1[i_task])
                              + time_string(), log)

    for i_task in range(args.n_tasks):
        if i_task in args.active_tasks:
            print_log(
                '  **Train** Task: {} Prec@1 {top1.avg:.3f} Error@1 {error1:.3f}'.format(
                    i_task, top1=top1[i_task], error1=100 - top1[i_task].avg),
                log)
    return [x.avg for x in top1], [x.avg for x in losses]


def validate(val_loader, model, criterion, log, args):
    losses = [AverageMeter() for i in range(args.n_tasks)]
    top1 = [AverageMeter() for i in range(args.n_tasks)]

    loss_task = [0] * args.n_tasks
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(val_loader):
            input_var = sample[0]
            target_var = sample[1:]
            if USE_CUDA:
                target_var = [x.cuda(non_blocking=True) for x in target_var]
                input_var = input_var.cuda()

            # compute output
            for i_task in range(args.n_tasks):
                if i_task in args.active_tasks:
                    output = model(input_var, i_task)
                    curr_loss = criterion(output, target_var[i_task])
                    loss_task = curr_loss.item()

                    # measure accuracy and record loss
                    prec1 = accuracy(output.data, target_var[i_task], topk=(1,))
                    losses[i_task].update(loss_task, input_var.size(0))
                    top1[i_task].update(prec1.item(), input_var.size(0))

    for i_task in range(args.n_tasks):
        if i_task in args.active_tasks:
            print_log('  **Test** Task: {} Prec@1 {top1.avg:.3f} Error@1 {error1:.3f}'.format(
                      i_task, top1=top1[i_task], error1=100 - top1[i_task].avg), log)

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


def adjust_learning_rate(optimizer, epoch, gammas, schedule, args, generic=True):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr
    assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
    for (gamma, step) in zip(gammas, schedule):
        if epoch >= step:
            lr = lr * gamma
        else:
            break
    for param_group in optimizer.param_groups:
        if generic:
            param_group['lr'] = lr * args.lr_gen
        else:
            param_group['lr'] = lr
    return lr


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

    if len(res) == 1:
        res = res[0]

    return res


if __name__ == '__main__':
    main()
