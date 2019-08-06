# https://github.com/pytorch/vision/blob/master/torchvision/models/__init__.py
import argparse
import os, copy
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from fblib.util.classification.utils import convert_secs2time, time_string, time_file_str, AverageMeter
from fblib.networks import resnet_imagenet, resnext_imagenet, posenet, se_resnet_imagenet, mobilenet_v2
from fblib.util.mypath import Path


def parse_args():
    def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1")

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--data', default=Path.db_root_dir('Imagenet'),
                        help='path to dataset')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='x50')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',
                        help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR',
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=200, type=int, metavar='N',
                        help='print frequency (default: 100)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--n_gpu', type=int, default=8,
                        help='number of GPUs')
    parser.add_argument('--group_norm', type=str2bool, default=False,
                        help='Group Normalization')

    args = parser.parse_args()
    args.prefix = time_file_str()

    return args


def main():
    args = parse_args()
    best_prec1 = 0

    if not args.group_norm:
        save_dir = os.path.join('/private/home/kmaninis/Experiments/imagenet', args.arch)
    else:
        save_dir = os.path.join('/private/home/kmaninis/Experiments/imagenet', args.arch + '-GN')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    log = open(os.path.join(save_dir, '{}.{}.log'.format(args.arch, args.prefix)), 'w')

    # create model
    print_log("=> creating model '{}'".format(args.arch), log)

    resol = 224
    if args.arch == 'res26':
        model = resnet_imagenet.resnet26(pretrained=False, group_norm=args.group_norm)
    elif args.arch == 'res50':
        model = resnet_imagenet.resnet50(pretrained=False, group_norm=args.group_norm)
    elif args.arch == 'res101':
        model = resnet_imagenet.resnet101(pretrained=False, group_norm=args.group_norm)
    elif args.arch == 'x50':
        model = resnext_imagenet.resnext50_32x4d(pretrained=False)
    elif args.arch == 'x101':
        model = resnext_imagenet.resnext101_32x4d(pretrained=False)
    elif args.arch == 'res26-se':
        model = se_resnet_imagenet.se_resnet26(num_classes=1000)
    elif args.arch == 'res50-se':
        model = se_resnet_imagenet.se_resnet50(num_classes=1000)
    elif args.arch == 'res101-se':
        model = se_resnet_imagenet.se_resnet101(num_classes=1000)
    elif args.arch == 'mobilenet-v2':
        model = mobilenet_v2.mobilenet_v2(pretrained=False, n_class=1000, last_channel=2048)
    elif args.arch == 'posenet':
        model = posenet.posenet_imagenet()
        resol = 256

    print_log("=> Model : {}".format(model), log)
    print_log("=> parameter : {}".format(args), log)

    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features, device_ids=list(range(args.n_gpu)))
        model.cuda()
    else:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.n_gpu))).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print_log("=> loading checkpoint '{}'".format(args.resume), log)
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print_log("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']), log)
        else:
            raise ValueError("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(resol),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True, sampler=None)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(resol),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    filename = os.path.join(save_dir, 'checkpoint.{}.{}.pth.tar'.format(args.arch, args.prefix))
    bestname = os.path.join(save_dir, 'best.{}.{}.pth.tar'.format(args.arch, args.prefix))

    start_time = time.time()
    epoch_time = AverageMeter()
    for epoch in range(args.start_epoch, args.epochs):
        lr = adjust_learning_rate(optimizer, epoch, args)

        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.val * (args.epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
        print_log(' [{:s}] :: {:3d}/{:3d} ----- [{:s}] {:s} LR={:}'.format(args.arch, epoch, args.epochs, time_string(),
                                                                           need_time, lr), log)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, log, args)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, log, args)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
            'args': copy.deepcopy(args),
        }, is_best, filename, bestname)
        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()
    log.close()


def train(train_loader, model, criterion, optimizer, epoch, log, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target_var = target.cuda(non_blocking=True)
        input_var = input.requires_grad_()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target_var, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print_log('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5), log)


def validate(val_loader, model, criterion, log, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    with torch.no_grad():
        for i, (input_var, target) in enumerate(val_loader):
            target_var = target.cuda(non_blocking=True)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target_var, topk=(1, 5))
            losses.update(loss.item(), input_var.size(0))
            top1.update(prec1.item(), input_var.size(0))
            top5.update(prec5.item(), input_var.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print_log('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5), log)

    print_log(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5,
                                                                                           error1=100 - top1.avg), log)

    return top1.avg


def save_checkpoint(state, is_best, filename, bestname):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, bestname)


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
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
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()


if __name__ == '__main__':
    main()
