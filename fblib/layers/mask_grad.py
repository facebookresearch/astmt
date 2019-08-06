import torch
import torch.nn as nn
from torch.autograd import Function
from fblib.layers.loss import DiffLoss


class MaskGradLayerF(Function):
    """
    Gradient Masking core function for multiple tasks
    """

    @staticmethod
    def forward(ctx, x, mask, mask_fw=False, task=None):
        ctx.mask = mask
        ctx.task = task
        if mask_fw:
            x = torch.mul(x, ctx.mask)

        if ctx.task is not None:
            print('Task: {}'.format(ctx.task))
            print('Mask Forward pass? {}'.format(mask_fw))
            print('Size of Mask: {}\n'
                  'Size of fw pass Output {}\n'.format(x.shape, ctx.mask.shape))
            print('Mask: \n{}\n'
                  'Output: \n{}\n'.format(ctx.mask.detach().cpu().numpy(), x[0, :, :2, :2].detach().cpu().numpy()))

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = torch.mul(grad_output, ctx.mask)
        if ctx.task is not None:  # Debug
            print('Task: {}'.format(ctx.task))
            print('Size of incoming Gradient: {}\n'
                  'Size of Mask: {}\n'
                  'Size of output Gradient {}\n'.format(grad_output.shape, ctx.mask.shape, output.shape))
            print(grad_output.dtype, ctx.mask.dtype, output.dtype)
            print('Incoming Gradient: \n{}\n'
                  'Mask: \n{}\n'
                  'Output Gradient: \n{}\n'.format(
                grad_output[0, :, :2, :2].detach().cpu().numpy(),
                ctx.mask.detach().cpu().numpy(),
                output[0, :, :2, :2].detach().cpu().numpy()))

        return output, None, None, None


class MaskGradPerm(nn.Module):
    """
    Gradient mask layer with permanent mask
    """

    def __init__(self, mask, mask_fw=False):
        super(MaskGradPerm, self).__init__()
        self.mask_fw = mask_fw
        self.mask = mask
        self.maskgrad = MaskGradLayerF
        if self.mask_fw:
            print('Will be masking forward pass')

    def forward(self, x, task=None):
        x = self.maskgrad.apply(x, self.mask, self.mask_fw, task)
        return x


class ConvCoupledMaskGrad(nn.Module):
    """
    Convolution operations along multiple tasks, coupled with Masks of Gradients
    """

    def __init__(self,
                 tasks,
                 conv_layer_kwargs=None,
                 norm=None,
                 is_cuda=True,
                 use_mask=True,
                 generic_path=False,
                 generic_channels_mult=2,
                 mask_fw=False,
                 use_diff_loss=False,
                 debug=False):

        super(ConvCoupledMaskGrad, self).__init__()

        self.tasks = tasks
        self.use_mask = use_mask
        self.generic_path = generic_path
        self.mask_fw = mask_fw
        self.use_diff_loss = use_diff_loss
        self.debug = debug

        # Assertions
        if self.use_diff_loss:
            assert self.generic_path

        self.convs = nn.ModuleDict({task: nn.Conv2d(**conv_layer_kwargs) for task in tasks})

        out_channels = conv_layer_kwargs['out_channels']
        in_channels = conv_layer_kwargs['in_channels']

        if not self.generic_path:
            in_per_task = in_channels // len(tasks)
        else:
            in_per_task = in_channels // (len(tasks) + generic_channels_mult)  # Assume common gen chan mult for all

        if self.use_mask:
            # Initialize masks
            print('Initializing Gradient Masks')
            self.grad_masks = nn.ModuleDict()
            idx = 0
            mask = torch.zeros(1, in_channels, 1, 1)
            for task in tasks:
                tmp = mask.clone()
                tmp[:, idx:idx + in_per_task, :, :] = 1

                if self.generic_path:
                    tmp[:, -generic_channels_mult * in_per_task:, :, :] = 1  # Activate generic path gradients

                if is_cuda:
                    tmp = tmp.cuda()

                self.grad_masks[task] = MaskGradPerm(mask=tmp, mask_fw=self.mask_fw)
                idx += in_per_task

        self.norm = nn.ModuleDict({task: norm(out_channels) for task in tasks})

        # Initialize optional generic path
        if self.generic_path:
            conv_layer_generic_kwargs = conv_layer_kwargs
            conv_layer_generic_kwargs['out_channels'] *= generic_channels_mult
            self.convs['generic'] = nn.Conv2d(**conv_layer_generic_kwargs)
            self.norm['generic'] = norm(generic_channels_mult * out_channels)

            if self.use_mask:
                # Initialize masks
                print('Initializing Gradient Mask for generic path')
                mask = torch.zeros(1, in_channels, 1, 1)
                tmp = mask.clone()
                tmp[:, -generic_channels_mult * in_per_task:, :, :] = 1

                if is_cuda:
                    tmp = tmp.cuda()

                self.grad_masks['generic'] = MaskGradPerm(mask=tmp, mask_fw=self.mask_fw)

        self.relu = nn.ReLU(inplace=True)

        if self.use_diff_loss:
            print('Using Orthogonality Constraint on Private and Shared features')
            self.diff_loss = DiffLoss(size_average=True)

    def forward(self, x):
        out = []

        tasks = self.tasks if not self.generic_path else self.tasks + ['generic']

        for task in tasks:
            tmp = x
            if self.use_mask:
                if not self.debug:
                    tmp = self.grad_masks[task](tmp)
                else:
                    tmp = self.grad_masks[task](tmp, task)  # Debugging prints the gradients

            tmp = self.convs[task](tmp)
            tmp = self.relu(self.norm[task](tmp))
            out.append(tmp)

        # Orthogonality constraint: Will only execute if there is a generic path, see assertion at __init__
        diff_loss = 0
        if self.use_diff_loss:
            for i in range(len(out) - 1):
                diff_loss += self.diff_loss(out[i], out[-1])
            # print('Diff Loss: {}'.format(diff_loss))

        return torch.cat(out, dim=1), diff_loss


class ConvCoupledMaskGradClassifier(nn.Module):
    """
    Convolution operations along multiple tasks, coupled with Masks of Gradients
    """

    def __init__(self,
                 tasks,
                 in_channels=None,
                 n_classes=None,
                 is_cuda=True,
                 use_mask=True,
                 generic_path=False,
                 generic_channels_mult=2,
                 mask_fw=False,
                 debug=False):

        super(ConvCoupledMaskGradClassifier, self).__init__()

        self.tasks = tasks
        self.use_mask = use_mask
        self.generic_path = generic_path
        self.mask_fw = mask_fw
        self.debug = debug

        self.convs = nn.ModuleDict({task: nn.Conv2d(in_channels, n_classes[task], kernel_size=1, bias=True)
                                    for task in tasks})

        if not self.generic_path:
            in_per_task = in_channels // len(tasks)
        else:
            in_per_task = in_channels // (len(tasks) + generic_channels_mult)  # Assume common gen chan mult for all

        if self.use_mask:
            # Initialize masks
            print('Initializing Gradient Masks for classifiers')
            self.grad_masks = nn.ModuleDict()
            idx = 0
            mask = torch.zeros(1, in_channels, 1, 1)
            for task in tasks:
                tmp = mask.clone()
                tmp[:, idx:idx + in_per_task, :, :] = 1

                if self.generic_path:
                    tmp[:, -generic_channels_mult * in_per_task:, :, :] = 1  # Activate generic path gradients

                if is_cuda:
                    tmp = tmp.cuda()
                self.grad_masks[task] = MaskGradPerm(mask=tmp, mask_fw=self.mask_fw)
                idx += in_per_task

    def forward(self, x, task=None):
        if self.use_mask:
            if not self.debug:
                x = self.grad_masks[task](x)
            else:
                x = self.grad_masks[task](x, task)
        x = self.convs[task](x)

        return x


class MaskGradAssembler(nn.Module):
    """
    Assemble features from different tasks so that they are compatible with ConvCoupledMaskGrad
    """

    def __init__(self, n_feats_a, n_feats_b, tasks=None, generic_path=False, generic_channels_mult=2):
        super(MaskGradAssembler, self).__init__()

        self.n_feats_a = n_feats_a
        self.n_feats_b = n_feats_b
        self.generic_path = generic_path
        if self.generic_path:
            self.n_feats_generic_a = self.n_feats_a * generic_channels_mult
            self.n_feats_generic_b = self.n_feats_b * generic_channels_mult

        self.tasks = tasks
        self.n_tasks = len(self.tasks)

    def forward(self, a, b):
        if not self.generic_path:
            assert (a.size(1) == self.n_feats_a * self.n_tasks)
            assert (b.size(1) == self.n_feats_b * self.n_tasks)
        else:
            assert (a.size(1) == self.n_feats_a * self.n_tasks + self.n_feats_generic_a)
            assert (b.size(1) == self.n_feats_b * self.n_tasks + self.n_feats_generic_b)

        out = []
        idx_a = 0
        idx_b = 0
        for _ in self.tasks:
            out.append(a[:, idx_a:idx_a + self.n_feats_a, :, :])
            out.append(b[:, idx_b:idx_b + self.n_feats_b, :, :])

            idx_a += self.n_feats_a
            idx_b += self.n_feats_b

        # Append generic features if present
        if self.generic_path:
            out.append(a[:, -self.n_feats_generic_a:, :, :])
            out.append(b[:, -self.n_feats_generic_b:, :, :])

        out = torch.cat(out, dim=1)
        return out


def test_mask_grad_assembler():
    print('Testing class: MaskGradAssembler')

    # Create inputs
    a, b = torch.rand(1, 8, 2, 2).requires_grad_().cuda(), torch.rand(1, 4, 2, 2).requires_grad_().cuda()

    net = MaskGradAssembler(n_feats_a=2, n_feats_b=1, tasks=['edge', 'semseg'], generic_path=True).cuda()

    out = net(a, b)

    print('Features A:')
    print('{}\n'.format(a.detach().cpu().numpy()))

    print('Features B:')
    print('{}\n'.format(b.detach().cpu().numpy()))

    print('Features Assembled:')
    print('{}\n'.format(out.detach().cpu().numpy()))


def test_mask_grad_perm():
    print('Testing class: MaskGradPerm')
    # Create mask
    mask = torch.zeros(1, 5, 1, 1)
    mask[:, :2, :, :] = 1.

    # Create input
    x = torch.rand(1, 5, 4, 4).requires_grad_()

    # To GPU (must be done outside class for MaskGradPermMask, unfortunately.
    mask = mask.cuda()
    x = x.cuda()
    net = MaskGradPerm(mask=mask).cuda()

    out = net(x, task='dummy')
    loss = out.sum()
    loss.backward()


def test_conv_coupled_mask_grad_and_classifier():
    print('Testing class: ConvCoupledMaskGrad')
    tasks = ['edge', 'semseg', 'human_parts', 'sal']
    n_classes = {'edge': 1, 'semseg': 21, 'human_parts': 7, 'sal': 1}
    in_channels = 6
    mask_fw = False
    use_diff_loss = True
    debug = False

    conv1 = ConvCoupledMaskGrad(tasks=tasks,
                                conv_layer_kwargs={'in_channels': in_channels,
                                                   'out_channels': int(in_channels / 6), 'kernel_size': 3, 'padding': 1},
                                norm=nn.BatchNorm2d,
                                is_cuda=True,
                                generic_path=True,
                                mask_fw=mask_fw,
                                use_diff_loss=use_diff_loss,
                                debug=debug)
    conv2 = ConvCoupledMaskGrad(tasks=tasks,
                                conv_layer_kwargs={'in_channels': in_channels,
                                                   'out_channels': int(in_channels / 6), 'kernel_size': 3, 'padding': 1},
                                norm=nn.BatchNorm2d,
                                is_cuda=True,
                                generic_path=True,
                                mask_fw=mask_fw,
                                use_diff_loss=use_diff_loss,
                                debug=debug)
    classifiers = ConvCoupledMaskGradClassifier(tasks=tasks,
                                                in_channels=in_channels,
                                                n_classes=n_classes,
                                                is_cuda=True,
                                                use_mask=True,
                                                generic_path=True,
                                                mask_fw=mask_fw,
                                                debug=debug)

    x = torch.rand(16, in_channels, 2, 2).requires_grad_()

    conv1.cuda()
    conv2.cuda()
    classifiers.cuda()
    x = x.cuda()

    x, df1 = conv1(x)
    x, df2 = conv2(x)
    out = {}
    for task in tasks:
        out[task] = classifiers(x, task=task)

    loss = 0
    for task in tasks:
        loss += out[task].sum()

    loss.backward()


if __name__ == '__main__':
    # test_mask_grad_perm()
    test_conv_coupled_mask_grad_and_classifier()
    # test_mask_grad_assembler()
