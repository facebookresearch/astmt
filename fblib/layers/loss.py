# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
import numpy as np


class SoftMaxwithLoss(Module):
    """
    This function returns cross entropy loss for semantic segmentation
    """

    def __init__(self):
        super(SoftMaxwithLoss, self).__init__()
        self.softmax = nn.LogSoftmax(dim=1)
        self.criterion = nn.NLLLoss(ignore_index=255)

    def forward(self, out, label):
        assert not label.requires_grad
        # out shape  batch_size x channels x h x w
        # label shape batch_size x 1 x h x w
        label = label[:, 0, :, :].long()
        loss = self.criterion(self.softmax(out), label)

        return loss


class BalancedCrossEntropyLoss(Module):
    """
    Balanced Cross Entropy Loss with optional ignore regions
    """

    def __init__(self, size_average=True, batch_average=True, pos_weight=None):
        super(BalancedCrossEntropyLoss, self).__init__()
        self.size_average = size_average
        self.batch_average = batch_average
        self.pos_weight = pos_weight

    def forward(self, output, label, void_pixels=None):
        assert (output.size() == label.size())
        labels = torch.ge(label, 0.5).float()

        # Weighting of the loss, default is HED-style
        if self.pos_weight is None:
            num_labels_pos = torch.sum(labels)
            num_labels_neg = torch.sum(1.0 - labels)
            num_total = num_labels_pos + num_labels_neg
            w = num_labels_neg / num_total
        else:
            w = self.pos_weight

        output_gt_zero = torch.ge(output, 0).float()
        loss_val = torch.mul(output, (labels - output_gt_zero)) - torch.log(
            1 + torch.exp(output - 2 * torch.mul(output, output_gt_zero)))

        loss_pos_pix = -torch.mul(labels, loss_val)
        loss_neg_pix = -torch.mul(1.0 - labels, loss_val)

        if void_pixels is not None and not self.pos_weight:
            w_void = torch.le(void_pixels, 0.5).float()
            loss_pos_pix = torch.mul(w_void, loss_pos_pix)
            loss_neg_pix = torch.mul(w_void, loss_neg_pix)
            num_total = num_total - torch.ge(void_pixels, 0.5).float().sum()
            w = num_labels_neg / num_total

        loss_pos = torch.sum(loss_pos_pix)
        loss_neg = torch.sum(loss_neg_pix)

        final_loss = w * loss_pos + (1 - w) * loss_neg

        if self.size_average:
            final_loss /= float(np.prod(label.size()))
        elif self.batch_average:
            final_loss /= label.size()[0]

        return final_loss


class BinaryCrossEntropyLoss(Module):
    """
    Binary Cross Entropy with ignore regions, not balanced.
    """

    def __init__(self, size_average=True, batch_average=True):
        super(BinaryCrossEntropyLoss, self).__init__()
        self.size_average = size_average
        self.batch_average = batch_average

    def forward(self, output, label, void_pixels=None):
        assert (output.size() == label.size())

        labels = torch.ge(label, 0.5).float()

        output_gt_zero = torch.ge(output, 0).float()
        loss_val = torch.mul(output, (labels - output_gt_zero)) - torch.log(
            1 + torch.exp(output - 2 * torch.mul(output, output_gt_zero)))

        loss_pos_pix = -torch.mul(labels, loss_val)
        loss_neg_pix = -torch.mul(1.0 - labels, loss_val)

        if void_pixels is not None:
            w_void = torch.le(void_pixels, 0.5).float()
            loss_pos_pix = torch.mul(w_void, loss_pos_pix)
            loss_neg_pix = torch.mul(w_void, loss_neg_pix)

        loss_pos = torch.sum(loss_pos_pix)
        loss_neg = torch.sum(loss_neg_pix)
        final_loss = loss_pos + loss_neg

        if self.size_average:
            final_loss /= float(np.prod(label.size()))
        elif self.batch_average:
            final_loss /= label.size()[0]

        return final_loss


class ImGrad(nn.Module):
    """
    Compute the spatial gradients of input with Sobel filter, in order to penalize gradient mismatch.
    Used for depth prediction
    """
    def __init__(self):
        super(ImGrad, self).__init__()
        self.convx = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.convy = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)

        fx = np.array([[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]])
        fy = np.array([[1,  2,  1],
                       [0,  0,  0],
                       [-1, -2, -1]])

        weight_x = torch.from_numpy(fx).float().unsqueeze(0).unsqueeze(0)
        weight_y = torch.from_numpy(fy).float().unsqueeze(0).unsqueeze(0)

        self.convx.weight.data = weight_x
        self.convy.weight.data = weight_y

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        grad_x = self.convx(x)
        grad_y = self.convy(x)

        return grad_x, grad_y


class GradLoss(nn.Module):
    """
    Compute gradient loss using ImGrad
    """
    def __init__(self, ignore_label=255):
        super(GradLoss, self).__init__()
        self.imgrad = ImGrad()
        self.ignore_label = ignore_label

    def forward(self, out, label):

        if self.ignore_label:
            n_valid = torch.sum(label != self.ignore_label).item()
            label[label == self.ignore_label] = 0

        out_grad_x, out_grad_y = self.imgrad(out)
        label_grad_x, label_grad_y = self.imgrad(label)

        out_grad = torch.cat((out_grad_y, out_grad_x), dim=1)
        label_grad = torch.cat((label_grad_y, label_grad_x), dim=1)

        # L1 norm
        loss = torch.abs(out_grad - label_grad)

        if self.ignore_label:
            loss = torch.sum(loss) / n_valid
        else:
            loss = torch.mean(loss)

        return loss


class RMSE_log(nn.Module):
    def __init__(self, ignore_label=255):
        super(RMSE_log, self).__init__()
        self.ignore_label = ignore_label

    def forward(self, out, label):
        out[out <= 0] = 1e-6
        log_mse = (torch.log(label) - torch.log(out)) ** 2

        # Only inside valid pixels
        if self.ignore_label:
            n_valid = torch.sum(label != self.ignore_label).item()
            log_mse[label == self.ignore_label] = 0
            log_mse = torch.sum(log_mse) / n_valid
        else:
            log_mse = torch.mean(log_mse)

        loss = torch.sqrt(log_mse)

        return loss


class L1loss(nn.Module):
    """
    L1 loss with ignore labels
    """
    def __init__(self, ignore_label=255):
        super(L1loss, self).__init__()
        self.loss_func = F.l1_loss
        self.ignore_label = ignore_label

    def forward(self, out, label):

        if self.ignore_label:
            n_valid = torch.sum(label != self.ignore_label).item()

        loss = torch.abs(out - label)
        loss[label == self.ignore_label] = 0
        loss = loss.sum()

        if self.ignore_label:
            loss.div_(max(n_valid, 1e-6))
        else:
            loss.div(float(np.prod(label.size())))

        return loss


class DepthLoss(nn.Module):
    """
    Loss for depth prediction. Combination of L1 loss and Gradient loss
    """
    def __init__(self):
        super(DepthLoss, self).__init__()
        self.diff_loss = L1loss(ignore_label=255)
        self.grad_loss = GradLoss(ignore_label=255)

    def forward(self, out, label):

        loss_diff = self.diff_loss(out, label)
        loss_grad = self.grad_loss(out, label)

        loss = loss_diff + loss_grad

        return loss


def normal_ize(bottom, dim=1):
    qn = torch.norm(bottom, p=2, dim=dim).unsqueeze(dim=dim) + 1e-12

    return bottom.div(qn)


class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()

    def forward(self, bottom):
        qn = torch.norm(bottom, p=2, dim=1).unsqueeze(dim=1) + 1e-12
        top = bottom.div(qn)

        return top


class NormalsLoss(Module):
    """
    L1 loss with ignore labels
    normalize: normalization for surface normals
    """

    def __init__(self, size_average=True, normalize=False, norm=1):

        super(NormalsLoss, self).__init__()

        self.size_average = size_average

        if normalize:
            self.normalize = Normalize()
        else:
            self.normalize = None

        if norm == 1:
            print('Using L1 loss for surface normals')
            self.loss_func = F.l1_loss
        elif norm == 2:
            print('Using L2 loss for surface normals')
            self.loss_func = F.mse_loss
        else:
            raise NotImplementedError

    def forward(self, out, label, ignore_label=255):
        assert not label.requires_grad

        if ignore_label:
            n_valid = torch.sum(label != ignore_label).item()
            out[label == ignore_label] = 0
            label[label == ignore_label] = 0

        if self.normalize is not None:
            out = self.normalize(out)

        loss = self.loss_func(out, label, reduction='sum')

        if self.size_average:
            if ignore_label:
                loss.div_(max(n_valid, 1e-6))
            else:
                loss.div(float(np.prod(label.size())))

        return loss


def normals_test():
    from fblib.dataloaders.pascal_context import PASCALContext

    flagvals = {'image': cv2.INTER_CUBIC,
                 'edge': cv2.INTER_NEAREST,
                 'semseg': cv2.INTER_NEAREST,
                 'human_parts': cv2.INTER_NEAREST,
                 'normals': cv2.INTER_CUBIC}

    transform = Compose([tr.RandomHorizontalFlip(),
                        tr.ScaleNRotate(rots=(-90, 90), scales=(1., 1.),
                                        flagvals=flagvals),
                        tr.FixedResize(resolutions={x: (512, 512) for x in flagvals},
                                       flagvals=flagvals),
                        tr.AddIgnoreRegions(),
                        tr.ToTensor()])
    dataset_human = PASCALContext(split=['train', 'val'], transform=transform, retname=True,
                                  do_edge=True, do_human_parts=True, do_semseg=True, do_normals=True)

    dataloader = torch.utils.data.DataLoader(dataset_human, batch_size=2, shuffle=False, num_workers=0)

    criterion = NormalsLoss(normalize=True)
    for i, sample in enumerate(dataloader):
        assert (sample['normals'].size()[2:] == sample['image'].size()[2:])
        loss = criterion(sample['normals'], sample['normals'])
        print('Sample number: {}. Loss: {} (should be very close to 0)'.format(i, loss.item()))


def depth_test():
    from fblib.dataloaders.nyud import NYUD_MT

    flagvals = {'image': cv2.INTER_CUBIC,
                'edge': cv2.INTER_NEAREST,
                'semseg': cv2.INTER_NEAREST,
                'normals': cv2.INTER_LINEAR,
                'depth': cv2.INTER_LINEAR}

    transform = Compose([tr.RandomHorizontalFlip(),
                        tr.ScaleNRotate(rots=(-90, 90), scales=(1., 1.),
                                        flagvals=flagvals),
                        tr.FixedResize(resolutions={x: (512, 512) for x in flagvals},
                                       flagvals=flagvals),
                        tr.AddIgnoreRegions(),
                        tr.ToTensor()])
    dataset_human = NYUD_MT(split=['train', 'val'], transform=transform, retname=True,
                            do_edge=True, do_semseg=True, do_normals=True, do_depth=True)

    dataloader = torch.utils.data.DataLoader(dataset_human, batch_size=2, shuffle=False, num_workers=0)

    criterion = DepthLoss()
    for i, sample in enumerate(dataloader):
        loss = criterion(sample['depth'], sample['depth'])
        print('Sample number: {}. Loss: {} (should be 0)'.format(i, loss.item()))


if __name__ == '__main__':
    import cv2
    from torchvision.transforms import Compose
    import fblib.dataloaders.custom_transforms as tr
    normals_test()
