# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import socket
import timeit
import cv2
from datetime import datetime
import imageio
import numpy as np

# PyTorch includes
import torch
import torch.optim as optim
from torch.nn.functional import interpolate

# Custom includes
from fblib.util.helpers import generate_param_report
from fblib.util.dense_predict.utils import lr_poly
from experiments.dense_predict import common_configs
from fblib.util.mtl_tools.multitask_visualizer import TBVisualizer
from fblib.util.model_resources.flops import compute_gflops
from fblib.util.model_resources.num_parameters import count_parameters
from fblib.util.dense_predict.utils import AverageMeter

# Custom optimizer
from fblib.util.optimizer_mtl.select_used_modules import make_closure

# Configuration
from experiments.dense_predict.pascal_mnet import config

# Tensorboard include
from tensorboardX import SummaryWriter


def main():
    p = config.create_config()

    gpu_id = 0
    device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")
    p.TEST.BATCH_SIZE = 32

    # Setting parameters
    n_epochs = p['epochs']

    print("Total training epochs: {}".format(n_epochs))
    print(p)
    print('Training on {}'.format(p['train_db_name']))

    snapshot = 10  # Store a model every snapshot epochs
    test_interval = p.TEST.TEST_INTER  # Run on test set every test_interval epochs
    torch.manual_seed(p.SEED)

    exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]
    if not os.path.exists(os.path.join(p['save_dir'], 'models')):
        if p['resume_epoch'] == 0:
            os.makedirs(os.path.join(p['save_dir'], 'models'))
        else:
            if not config.check_downloaded(p):
                print('Folder does not exist.No checkpoint to resume from. Exiting')
                exit(1)

    net = config.get_net_mnet(p)

    gflops = compute_gflops(net, in_shape=(p['trBatch'], 3, p.TRAIN.SCALE[0], p.TRAIN.SCALE[1]),
                            tasks=p.TASKS.NAMES[0])
    print('GFLOPS per task: {}'.format(gflops / p['trBatch']))

    print('\nNumber of parameters (in millions): {0:.3f}'.format(count_parameters(net) / 1e6))
    print('Number of parameters (in millions) for decoder: {0:.3f}\n'.format(count_parameters(net.decoder) / 1e6))

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
        train_params = config.get_train_params(net, p, p['lr'])

        optimizer = optim.SGD(train_params, lr=p['lr'], momentum=p.TRAIN.MOMENTUM, weight_decay=p['wd'])

        for task in p.TASKS.NAMES:
            # Losses
            criteria_tr[task] = config.get_loss(p, task)
            criteria_ts[task] = config.get_loss(p, task)
            criteria_tr[task].to(device)
            criteria_ts[task].to(device)

        # Preparation of the data loaders
        transforms_tr, transforms_ts, _ = config.get_transformations(p)
        trainloader = config.get_train_loader(p, db_name=p['train_db_name'], transforms=transforms_tr)
        testloader = config.get_test_loader(p, db_name=p['test_db_name'], transforms=transforms_ts)

        # TensorBoard Image Visualizer
        tb_vizualizer = TBVisualizer(tasks=p.TASKS.NAMES, min_ranges=p.TASKS.TB_MIN, max_ranges=p.TASKS.TB_MAX,
                                     batch_size=p['trBatch'])

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

            if p['dscr_type'] is not None:
                print('Value of alpha: {}'.format(alpha))

            for ii, sample in enumerate(trainloader):
                curr_loss_dscr = 0

                # Grab the input
                inputs = sample['image']
                inputs.requires_grad_()
                inputs = inputs.to(device)

                task_gts = list(sample.keys())
                tasks = net.tasks

                gt_elems = {x: sample[x].to(device, non_blocking=True) for x in tasks}
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
                        = net.compute_losses(output, features, criteria_tr, gt_elems, alpha, p)

                    loss_tasks = losses_tasks[task]
                    running_loss_tr[task] += losses_tasks[task].item()
                    curr_loss_task[task] = losses_tasks[task].item()

                    counter_tr[task] += 1

                    # Store output for logging
                    outputs[task] = output[task].detach()

                    if p['dscr_type'] is not None:
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

                    # Backward pass inside make_closure to update only weights that were used during fw pass
                    optimizer.zero_grad()
                    optimizer.step(closure=make_closure(loss=loss, net=net))

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

                    if p['dscr_type'] is not None:
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
                    if p['dscr_type'] is not None:
                        writer.add_scalar('data/train_loss_dscr_iter', curr_loss_dscr, ii + num_img_tr * epoch)
                        curr_loss_dscr = 0.

                # Log train images to Tensorboard
                if p.TRAIN.TENS_VIS and ii % p.TRAIN.TENS_VIS_INTER == 0:
                    curr_iter = ii + num_img_tr * epoch
                    tb_vizualizer.visualize_images_tb(writer, sample, outputs,
                                                      global_step=curr_iter, tag=ii, phase='train')

                if p['poly'] and ii % num_img_tr == num_img_tr - 1:
                    lr_ = lr_poly(p['lr'], iter_=epoch, max_iter=n_epochs)
                    print('(poly lr policy) learning rate: {0:.6f}'.format(lr_))
                    train_params = config.get_train_params(net, p, lr_)
                    optimizer = optim.SGD(train_params, lr=lr_, momentum=p.TRAIN.MOMENTUM, weight_decay=p['wd'])
                    optimizer.zero_grad()

            # Save the model
            if (epoch % snapshot) == snapshot - 1 and epoch != 0:
                torch.save(net.state_dict(), os.path.join(p['save_dir'], 'models',
                                                          'model_epoch-' + str(epoch) + '.pth'))

            # One testing epoch
            if p.TEST.USE_TEST and epoch % test_interval == (test_interval - 1):
                print('Testing Phase')
                top1_dscr = AverageMeter()
                net.eval()
                start_time = timeit.default_timer()
                for ii, sample in enumerate(testloader):

                    inputs = sample['image'].to(device)
                    task_gts = list(sample.keys())

                    tasks = net.tasks
                    gt_elems = {x: sample[x].to(device, non_blocking=True) for x in tasks}
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
                            = net.compute_losses(output, features, criteria_tr, gt_elems, alpha, p)

                        running_loss_ts[task] += losses_tasks[task].item()
                        counter_ts[task] += 1

                        # For logging
                        outputs[task] = output[task].detach()

                        if p['dscr_type'] is not None:
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

                        if p['dscr_type'] is not None:
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

    # Generate Results
    net.eval()
    _, _, transforms_infer = config.get_transformations(p)
    for db_name in p['infer_db_names']:

        testloader = config.get_test_loader(p, db_name=db_name, transforms=transforms_infer, infer=True)
        save_dir_res = os.path.join(p['save_dir'], 'Results_' + db_name)

        print('Testing network')
        # Main Testing Loop
        with torch.no_grad():
            for ii, sample in enumerate(testloader):

                img, meta = sample['image'], sample['meta']

                # Forward pass of the mini-batch
                inputs = img.to(device)
                tasks = net.tasks

                for task in tasks:
                    output, _ = net.forward(inputs, task=task)

                    save_dir_task = os.path.join(save_dir_res, task)
                    if not os.path.exists(save_dir_task):
                        os.makedirs(save_dir_task)

                    output = interpolate(output, size=(inputs.size()[-2], inputs.size()[-1]),
                                         mode='bilinear', align_corners=False)
                    output = common_configs.get_output(output, task)

                    for jj in range(int(inputs.size()[0])):
                        if len(sample[task][jj].unique()) == 1 and sample[task][jj].unique() == 255:
                            continue

                        fname = meta['image'][jj]

                        result = cv2.resize(output[jj], dsize=(meta['im_size'][1][jj], meta['im_size'][0][jj]),
                                            interpolation=p.TASKS.INFER_FLAGVALS[task])

                        imageio.imwrite(os.path.join(save_dir_task, fname + '.png'), result.astype(np.uint8))

    if p.EVALUATE:
        common_configs.eval_all_results(p)


if __name__ == '__main__':
    main()


