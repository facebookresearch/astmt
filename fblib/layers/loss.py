import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
import numpy as np


class DiffLoss(nn.Module):
    """
    Orthogonality Loss from Bousmalis et al. NIPS 2016
    """
    def __init__(self, size_average=True):
        super(DiffLoss, self).__init__()
        self.size_average = size_average

    def forward(self, private_samples, shared_samples):
        # print(input1[0, :, 0, 0])
        # print(input2[0, :, 0, 0])
        pr_sz = private_samples.shape
        if pr_sz[2] > 1:
            private_samples = private_samples.permute(0, 2, 3, 1)
            shared_samples = shared_samples.permute(0, 2, 3, 1)
            n_observations = private_samples.size(0) * private_samples.size(1) * private_samples.size(2)
        else:
            n_observations = private_samples.size(0)

        private_samples = private_samples.contiguous().view(n_observations, -1)
        shared_samples = shared_samples.contiguous().view(n_observations, -1)
        # print(input1[0, :])
        # print(input2[0, :])

        private_samples = private_samples.sub(private_samples.mean(dim=0))
        shared_samples = shared_samples.sub(shared_samples.mean(dim=0))

        private_samples_l2_norm = torch.norm(private_samples, p=2, dim=1, keepdim=True)
        private_samples = private_samples.div(private_samples_l2_norm + 1e-6)

        shared_samples_l2_norm = torch.norm(shared_samples, p=2, dim=1, keepdim=True)
        shared_samples = shared_samples.div(shared_samples_l2_norm + 1e-6)

        correlation = (private_samples.t().mm(shared_samples).pow(2))
        # print(correlation)

        if self.size_average:
            diff_loss = torch.mean(correlation) / (pr_sz[2] * pr_sz[3])
        else:
            diff_loss = torch.sum(correlation)

        diff_loss *= (diff_loss > 0).float()

        return diff_loss


def test_orthogonality():

    criterion = DiffLoss()

    a = torch.rand(2, 16, 4, 4)
    b = torch.rand(2, 4, 4, 4)

    print(criterion(a, b))
    print(criterion(a, a))


class HeatmapLoss(Module):
    """
    loss for detection heatmap
    """
    def __init__(self):
        super(HeatmapLoss, self).__init__()

    def forward(self, pred, gt):
        assert pred.size() == gt.size()

        loss = ((pred - gt) ** 2).expand_as(pred)
        loss = loss.mean(dim=3).mean(dim=2).mean(dim=1)

        return loss


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
        # print("Loss positives: {}".format(loss_pos))
        # print("Loss negatives: {}".format(loss_neg))
        # print("Loss: {}".format(final_loss))

        if self.size_average:
            final_loss /= float(np.prod(label.size()))
        elif self.batch_average:
            final_loss /= label.size()[0]

        return final_loss


class MILLoss(Module):
    """
    Multi-Instance Learning Loss
    """

    def __init__(self, size_average=True, debug=False, visualize=False, pos_weight=None):
        super(MILLoss, self).__init__()
        self.size_average = size_average
        self.counter = 0
        self.bignum = 1e6
        self.debug = debug
        self.visualize = visualize
        self.bce = BinaryCrossEntropyLoss(size_average=self.size_average)
        self.pos_weight = pos_weight

    def forward(self, output, pos, idxs):
        self.counter += 1
        # print(self.counter)

        loss_negatives = self.bce(output, pos, void_pixels=pos)

        # Iterate over all batches of MIL
        loss_positives = 0
        n_batches = len(idxs) if isinstance(idxs, list) else 1
        for i_batch in range(n_batches):

            curr_idxs = idxs[i_batch][0, :, :] if isinstance(idxs, list) else idxs[i_batch, 0, :, :]

            # In case there are no MILs.
            if curr_idxs[0, 0] < 0:
                print('Instance without MILs. Not contributing to the loss')
                continue

            prediction_pos = output[i_batch, 0, :, :]
            prediction_pos = prediction_pos.view(-1)
            iH, iW = curr_idxs.size()
            idxs_mask = self.bignum * (curr_idxs > 0).float()  # For masking curr_idxs == 0
            idxs_mask[idxs_mask == 0] = -self.bignum

            curr_idxs = curr_idxs.contiguous().view(-1)
            idxs_values = prediction_pos[curr_idxs.long()]
            idxs_values = idxs_values.view(iH, iW)
            idxs_values = torch.min(idxs_values, idxs_mask)

            maxScore, maxIdx = torch.max(idxs_values, 1)
            minmaxScore = maxScore.detach().min()

            # Handle case that all indices in a row are 0
            if minmaxScore == -self.bignum:
                maxScore_mask = 1. - 2. * (maxScore == -self.bignum).float()
                maxScore = torch.mul(maxScore, maxScore_mask)

            # Handle weird case: Pos at maxIdx is 0
            if self.debug:
                pos_at_indices = pos[0, 0, :, :].view(-1)[curr_idxs.long()].view(iH, iW)
                test = self._test_values(pos_at_indices, idxs_mask, maxIdx)
                if not test:
                    print("Check MILs..!")

            exps = torch.exp(-maxScore)
            logs = torch.log(1 + exps)
            curr_loss_positives = torch.sum(logs)

            if self.size_average:
                curr_loss_positives /= iH

            loss_positives += curr_loss_positives

        loss_positives /= n_batches

        # Proper weighting of the losses, default is HED-style
        if self.pos_weight is None:
            labels = torch.ge(pos, 0.5).float()
            num_labels_pos = torch.sum(labels)
            num_labels_neg = torch.sum(1.0 - labels)
            num_total = num_labels_pos + num_labels_neg
            w = num_labels_neg / num_total
        else:
            w = self.pos_weight

        loss_final = w * loss_positives + (1 - w) * loss_negatives

        # print("Loss positives: {}".format(loss_positives))
        # print("Loss negatives: {}".format(loss_negatives))

        if self.visualize:
            import matplotlib.pyplot as plt
            im = output.detach().cpu().data.numpy()[0, 0, :, :]
            sigm = 1 / (1 + np.exp(-im))
            plt.imshow(sigm)
            plt.show()

            plt.imshow(pos[0, 0, :, :].detach().cpu().numpy())
            plt.show()

        return loss_final

    def _test_values(self, pos, mask, maxIdx, show_message=True):
        pos = pos.cpu().numpy()
        mask = mask.cpu().numpy()
        maxIdx = maxIdx.cpu().numpy()

        idxs_pos = np.where(pos == 1)
        idxs_mask = np.where(mask == self.bignum)

        test_mask = (mask[idxs_pos[0], idxs_pos[1]] == self.bignum).all()
        test_pos = (pos[idxs_mask[0], idxs_mask[1]] == 1).all()
        test_maxIdx = (pos[list(range(pos.shape[0])), maxIdx] == 1).all()
        test_maxIdx_2 = (mask[list(range(pos.shape[0])), maxIdx] == self.bignum).all()

        if show_message:
            if not test_mask:
                print("Not a problem: where pos == 1, Mask == 1e6: {}".format(test_mask))
            if not test_pos:
                print("Problem found: where Mask == 1e6, pos == 1: {}".format(test_pos))
            if not test_maxIdx:
                print("Problem found: Pos at maxIdx == 1: {}".format(test_maxIdx))
            if not test_maxIdx_2:
                print("Problem found: Mask at maxIdx == 1e6: {}".format(test_maxIdx_2))
                print("Don't worry. Handled through masking the scores")

        return test_maxIdx and test_pos


class ImGrad(nn.Module):
    """
    Compute gradients of Prediction and Label with Sobel filter, in order to penalize gradient mismatch.
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
        N, C, _, _ = x.size()
        grad_x = self.convx(x)
        grad_y = self.convy(x)

        return grad_x, grad_y


class GradLoss(nn.Module):
    """Compute gradient loss using ImGrad
    """
    def __init__(self, ignore_label=255):
        super(GradLoss, self).__init__()
        self.imgrad = ImGrad()
        self.ignore_label = ignore_label

    def forward(self, out, label):
        N, C, _, _ = out.size()

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
    """Loss for depth prediction. Combination of L1 loss and Gradient loss
    """
    def __init__(self):
        super(DepthLoss, self).__init__()
        self.diff_loss = L1loss(ignore_label=255)
        self.grad_loss = GradLoss(ignore_label=255)

    def forward(self, out, label):

        loss_diff = self.diff_loss(out, label)
        loss_grad = self.grad_loss(out, label)
        # print('loss_diff: {}, loss_grad: {}'.format(loss_diff.item(), loss_grad))

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


def mil_test():
    transforms_tr = Compose(
        [tr.RandomHorizontalFlip(), tr.FixedResizeWithMIL(resolutions=[609, 512, 321]), tr.ToTensor()])

    dataset = PASCALContext(split=['train', 'val'], transform=transforms_tr, retname=True, do_edge=True, use_mil=True)
    loss = MILLoss()

    for i in range(len(dataset)):
        if i < 1102:
            continue
        sample = dataset[i]
        img, pos, idxs = sample['image'], sample['edge'], sample['edgeidx']
        out_pos = torch.cat([100 - 100 * pos, 100 * pos])
        print(out_pos.min(), out_pos.max())
        pos.unsqueeze_(0)
        out_pos.unsqueeze_(0)
        idxs.unsqueeze_(0)
        l = loss(out_pos, pos, idxs)
        print(l)


def normals_test():
    transform = Compose([tr.RandomHorizontalFlip(),
                        tr.ScaleNRotate(rots=(-90, 90), scales=(1., 1.),
                                        flagvals={'image': cv2.INTER_CUBIC,
                                                  'edge': cv2.INTER_NEAREST,
                                                  'semseg': cv2.INTER_NEAREST,
                                                  'human_parts': cv2.INTER_NEAREST,
                                                  'normals': cv2.INTER_CUBIC}),
                        tr.FixedResize(resolutions={'image': (512, 512),
                                                    'edge': (512, 512),
                                                    'semseg': (512, 512),
                                                    'human_parts': (512, 512),
                                                    'normals': (512, 512)},
                                       flagvals={'image': cv2.INTER_CUBIC,
                                                 'edge': cv2.INTER_NEAREST,
                                                 'semseg': cv2.INTER_NEAREST,
                                                 'human_parts': cv2.INTER_NEAREST,
                                                 'normals': cv2.INTER_CUBIC},
                                       use_mil=False, mildil=False),
                        tr.AddIgnoreRegions(),
                        tr.ToTensor()])
    dataset_human = PASCALContext(split=['train', 'val'], transform=transform, retname=True, use_mil=False,
                                  do_edge=True, do_human_parts=True, do_semseg=True, do_normals=True,
                                  all_tasks_present=False)

    dataloader = torch.utils.data.DataLoader(dataset_human, batch_size=2, shuffle=False, num_workers=0)

    loss = NormalsLoss(normalize=True)
    for i, sample in enumerate(dataloader):
        print(i)
        assert (sample['normals'].size()[2:] == sample['image'].size()[2:])
        curr_loss = loss(sample['normals'], sample['normals'])
        print(curr_loss.item())


def test_depth():
    flagvals = {'image': cv2.INTER_CUBIC,
                'edge': cv2.INTER_NEAREST,
                'semseg': cv2.INTER_NEAREST,
                'normals': cv2.INTER_LINEAR,
                'depth': cv2.INTER_LINEAR}

    transform = Compose([tr.RandomHorizontalFlip(),
                        tr.ScaleNRotate(rots=(-90, 90), scales=(1., 1.),
                                        flagvals=flagvals),
                        tr.FixedResize(resolutions={'image': (512, 512),
                                                    'edge': (512, 512),
                                                    'semseg': (512, 512),
                                                    'normals': (512, 512),
                                                    'depth': (512, 512)},
                                       flagvals=flagvals,
                                       use_mil=False, mildil=False),
                        tr.AddIgnoreRegions(),
                        tr.ToTensor()])
    dataset_human = NYUD_MT(split=['train', 'val'], transform=transform, retname=True,
                            do_edge=True, do_semseg=True, do_normals=True, do_depth=True)

    dataloader = torch.utils.data.DataLoader(dataset_human, batch_size=2, shuffle=False, num_workers=0)

    criterion = DepthLoss()
    for i, sample in enumerate(dataloader):
        print(i)
        loss = criterion(sample['depth'], sample['depth'])
        print(loss.item())


if __name__ == '__main__':
    import cv2
    from fblib.dataloaders.pascal_context import PASCALContext
    from fblib.dataloaders.nyud import NYUD_MT
    from torchvision.transforms import Compose
    import fblib.dataloaders.custom_transforms as tr
    test_depth()
