import torch
import torch.nn as nn
import torch.nn.functional as F

from fblib.layers.mask_grad import ConvCoupledMaskGrad, ConvCoupledMaskGradClassifier, MaskGradAssembler
from fblib.networks.classifiers_multitask import AtrousSpatialPyramidPoolingModule
from fblib.layers.squeeze import ConvCoupledSE


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     dilation=dilation, padding=dilation, bias=False)


class UbernetDecoder(nn.Module):
    """
    Simple Shallow decoder (like Ubernet)
    """

    def __init__(self,
                 tasks,
                 in_channels_low,
                 in_channels_high,
                 n_classes,
                 norm=nn.BatchNorm2d,
                 ret_features=True):
        super(UbernetDecoder, self).__init__()

        self.tasks = tasks
        self.ret_features = ret_features

        self.high_level = nn.ModuleDict()
        self.low_level = nn.ModuleDict()
        self.predict = nn.ModuleDict()

        for task in tasks:
            self.high_level[task] = nn.Sequential(nn.Conv2d(in_channels=in_channels_high,
                                                            out_channels=n_classes[task],
                                                            kernel_size=1,
                                                            bias=False),
                                                  norm(n_classes[task]),
                                                  nn.ReLU(inplace=True))

            self.low_level[task] = nn.Sequential(nn.Conv2d(in_channels=in_channels_low,
                                                           out_channels=n_classes[task],
                                                           kernel_size=1,
                                                           bias=False),
                                                 norm(n_classes[task]),
                                                 nn.ReLU(inplace=True))

            self.predict[task] = nn.Conv2d(in_channels=2 * n_classes[task],
                                           out_channels=n_classes[task],
                                           kernel_size=1,
                                           bias=True)

    def forward(self, x_low, x_high, task=None):
        # Reduce dimensionality of low-level features
        x_low = self.low_level[task](x_low)

        # Reduce dimensionality of high-level features and upscale
        x_high = self.high_level[task](x_high)
        x_high = F.interpolate(x_high, size=(x_low.shape[2], x_low.shape[3]), mode='bilinear', align_corners=False)

        # Concatenate features
        x = torch.cat([x_low, x_high], dim=1)

        features = x

        # Make final prediction
        x = self.predict[task](x)

        if self.ret_features:
            return x, features
        else:
            return x


def test_ubernet():
    tasks = ['edge', 'semseg', 'human_parts']
    out_channels = {'edge': 1, 'semseg': 21, 'human_parts': 7}

    in_channels_low = 256
    in_channels_high = 2048

    x_low = torch.rand(2, in_channels_low, 128, 128)
    x_high = torch.rand(2, in_channels_high, 64, 64)

    net = UbernetDecoder(tasks=tasks,
                         in_channels_low=in_channels_low,
                         in_channels_high=in_channels_high,
                         n_classes=out_channels)

    x_low, x_high, net = x_low.cuda(), x_high.cuda(), net.cuda()

    for task in tasks:
        out, _ = net(x_low, x_high, task=task)
        print('Task: {}, Output Shape: {}'.format(task, out.shape))


class MultiPathDecoder(nn.Module):
    """
    Multi-path decoder with the following functionalities:
    - Masking of Gradients
    - Masking of features during forward pass
    - Use of Orthogonality Constraint
    """

    def __init__(self,
                 tasks,
                 n_classes,
                 classifier='conv',
                 in_channels_low=256,
                 in_channels_high=2048,
                 out_f_classifier=64,
                 norm=nn.BatchNorm2d,
                 use_mask=True,
                 gen_path=False,
                 gen_mult=2,
                 mask_fw=False,
                 use_diff_loss=False
                 ):
        super(MultiPathDecoder, self).__init__()
        print('Multi-path Decoder')

        out_f_low = 48 * out_f_classifier / 256  # Adapt in case of thinner classifiers
        assert (int(out_f_low) == out_f_low)
        out_f_low = int(out_f_low)

        self.tasks = tasks
        self.use_diff_loss = use_diff_loss
        self.mask_fw = mask_fw

        if classifier == 'conv':
            print('Using simple convolutional classifier')
            self.layer5 = ConvCoupledMaskGrad(tasks=self.tasks,
                                              conv_layer_kwargs={'in_channels': in_channels_high,
                                                                 'out_channels': out_f_classifier,
                                                                 'kernel_size': 1,
                                                                 'bias': False},
                                              norm=norm,
                                              use_mask=False,  # Common reading feature layer
                                              generic_path=gen_path,
                                              generic_channels_mult=gen_mult,
                                              mask_fw=mask_fw,
                                              use_diff_loss=False)
        else:
            raise NotImplementedError('Choose one of the available classifiers')

        self.low_level_reduce = ConvCoupledMaskGrad(tasks=self.tasks,
                                                    conv_layer_kwargs={'in_channels': in_channels_low,
                                                                       'out_channels': out_f_low,
                                                                       'kernel_size': 3,
                                                                       'padding': 1,
                                                                       'bias': False},
                                                    norm=norm,
                                                    use_mask=False,  # Common reading feature layer
                                                    generic_path=gen_path,
                                                    generic_channels_mult=gen_mult,
                                                    mask_fw=self.mask_fw,
                                                    use_diff_loss=False)

        self.assembler = MaskGradAssembler(n_feats_a=out_f_classifier,
                                           n_feats_b=out_f_low,
                                           tasks=self.tasks,
                                           generic_path=gen_path,
                                           generic_channels_mult=gen_mult)

        if not gen_path:
            in_channels_mult = len(tasks)
        else:
            in_channels_mult = len(tasks) + gen_mult

        concat_in_channels = in_channels_mult * (out_f_classifier + out_f_low)
        self.conv_concat = ConvCoupledMaskGrad(tasks=self.tasks,
                                               conv_layer_kwargs={'in_channels': concat_in_channels,
                                                                  'out_channels': out_f_classifier,
                                                                  'kernel_size': 3,
                                                                  'padding': 1,
                                                                  'bias': False},
                                               norm=norm,
                                               use_mask=use_mask,
                                               generic_path=gen_path,
                                               generic_channels_mult=gen_mult,
                                               mask_fw=mask_fw,
                                               use_diff_loss=use_diff_loss)

        process_in_channels = in_channels_mult * out_f_classifier
        self.conv_process = ConvCoupledMaskGrad(tasks=self.tasks,
                                                conv_layer_kwargs={'in_channels': process_in_channels,
                                                                   'out_channels': out_f_classifier,
                                                                   'kernel_size': 3,
                                                                   'padding': 1,
                                                                   'bias': False},
                                                norm=norm,
                                                use_mask=use_mask,
                                                generic_path=gen_path,
                                                generic_channels_mult=gen_mult,
                                                mask_fw=mask_fw,
                                                use_diff_loss=use_diff_loss)

        classifier_in_channels = in_channels_mult * out_f_classifier
        self.conv_predict = ConvCoupledMaskGradClassifier(tasks=tasks,
                                                          in_channels=classifier_in_channels,
                                                          n_classes=n_classes,
                                                          use_mask=use_mask,
                                                          generic_path=gen_path,
                                                          generic_channels_mult=gen_mult,
                                                          mask_fw=mask_fw)

    def forward(self, x_low, x_high, task_gts=None):
        out = {}

        # Process high-level features
        x_high, diff_loss_1 = self.layer5(x_high)
        x_high = F.interpolate(x_high, size=(x_low.shape[2], x_low.shape[3]), mode='bilinear', align_corners=False)

        # Process low-level features
        x_low, diff_loss_2 = self.low_level_reduce(x_low)

        # Assemble low- and high-level features to be compatible with GradMask
        x = self.assembler(x_high, x_low)

        # Process all features together
        x, diff_loss_3 = self.conv_concat(x)
        x, diff_loss_4 = self.conv_process(x)

        # Classifiers
        for task in self.tasks:
            if self._chk_forward(task, task_gts):
                out[task] = self.conv_predict(x, task=task)

        diff_loss = 0
        if self.use_diff_loss:
            diff_loss = diff_loss_1 + 0.01 * diff_loss_2 + diff_loss_3 + diff_loss_4

        return out, diff_loss

    @staticmethod
    def _chk_forward(task, meta):
        if meta is None:
            return True
        else:
            return task in meta


def test_multipath():
    import fblib.util.visualize as viz

    tasks = ['edge', 'semseg', 'human_parts']
    n_classes = {'edge': 1, 'semseg': 21, 'human_parts': 7}

    in_channels_low = 256
    in_channels_high = 2048
    out_f_classifier = 64
    use_mask = True
    gen_path = True
    gen_mult = 2
    mask_fw = False

    x_low = torch.rand(2, in_channels_low, 128, 128).requires_grad_()
    x_high = torch.rand(2, in_channels_high, 64, 64).requires_grad_()

    net = MultiPathDecoder(tasks=tasks,
                           n_classes=n_classes,
                           classifier='conv',
                           in_channels_high=in_channels_high,
                           in_channels_low=in_channels_low,
                           out_f_classifier=out_f_classifier,
                           norm=nn.BatchNorm2d,
                           use_mask=use_mask,
                           gen_path=gen_path,
                           gen_mult=gen_mult,
                           mask_fw=mask_fw,
                           use_diff_loss=True)

    x_low, x_high, net = x_low.cuda(), x_high.cuda(), net.cuda()

    out, _ = net(x_low, x_high, task_gts=tasks)
    for task in out:
        print('Task: {}, Output Shape: {}'.format(task, out[task].shape))

    g = viz.make_dot(out, net.state_dict())
    g.view(directory='/private/home/kmaninis/')


class ASPPv3Plus(nn.Module):
    """
    ASPP-v3 decoder
    """

    def __init__(self,
                 tasks,
                 n_classes,
                 classifier='atrous-v3',
                 in_channels_low=256,
                 in_channels_high=2048,
                 out_f_classifier=64,
                 atrous_rates=None,
                 norm=nn.BatchNorm2d,
                 norm_per_task=True,
                 squeeze=False,
                 adapters=False,
                 smooth=False,
                 casc_type=None
                 ):
        super(ASPPv3Plus, self).__init__()
        print('Initializing ASPP v3 Decoder for multiple tasks')

        if atrous_rates is None:
            atrous_rates = [6, 12, 18]

        out_f_low = 48 * out_f_classifier / 256  # Adapt in case of thinner classifiers
        assert (int(out_f_low) == out_f_low)
        out_f_low = int(out_f_low)

        kwargs_low = {"num_features": int(out_f_low), "affine": True}
        kwargs_out = {"num_features": out_f_classifier, "affine": True}

        self.tasks = tasks

        if classifier == 'atrous-v3':
            print('Initializing classifier: ASPP with global features (Deeplab-v3+)')
            self.layer5 = AtrousSpatialPyramidPoolingModule(in_f=in_channels_high,
                                                            depth=out_f_classifier,
                                                            dilation_series=atrous_rates,
                                                            sync_bnorm=(norm != nn.BatchNorm2d),
                                                            tasks=self.tasks,
                                                            norm_per_task=norm_per_task,
                                                            squeeze=squeeze,
                                                            adapters=adapters,
                                                            smooth=smooth)
        elif classifier == 'conv':
            self.layer5 = ConvCoupledSE(tasks=tasks,
                                        process_layers=nn.Conv2d(in_channels_high, out_f_classifier, kernel_size=1,
                                                                 bias=False),
                                        norm=norm,
                                        norm_kwargs=kwargs_low,
                                        norm_per_task=norm_per_task,
                                        squeeze=squeeze,
                                        reduction=4)  # Too few features if we keep default reduction (16)
        else:
            raise NotImplementedError('Choose one of the available classifiers')

        self.low_level_reduce = ConvCoupledSE(tasks=tasks,
                                              process_layers=nn.Conv2d(in_channels_low, int(out_f_low), kernel_size=1,
                                                                       bias=False),
                                              norm=norm,
                                              norm_kwargs=kwargs_low,
                                              norm_per_task=norm_per_task,
                                              squeeze=squeeze,
                                              adapters=adapters,
                                              reduction=4)  # Too few features if we keep default reduction (16)

        self.conv_concat = ConvCoupledSE(tasks=tasks,
                                         process_layers=conv3x3(out_f_classifier + int(out_f_low),
                                                                out_f_classifier),
                                         norm=norm,
                                         norm_kwargs=kwargs_out,
                                         norm_per_task=norm_per_task,
                                         squeeze=squeeze,
                                         adapters=adapters)

        self.conv_process = ConvCoupledSE(tasks=tasks,
                                          process_layers=conv3x3(out_f_classifier, out_f_classifier),
                                          norm=norm,
                                          norm_kwargs=kwargs_out,
                                          norm_per_task=norm_per_task,
                                          squeeze=squeeze,
                                          adapters=adapters)

        self.conv_predict = nn.ModuleDict(
            {task: nn.Conv2d(out_f_classifier, n_classes[task], kernel_size=1, bias=True) for task in tasks})

        self.casc_type = casc_type
        if self.casc_type == 'late':
            print('Initializing Late cascading on decoder')
            n_latent = 32  # magic number
            self.residual_cascade = nn.Sequential(nn.Conv2d(out_f_classifier, n_latent, kernel_size=1),
                                                  nn.ReLU(inplace=True),
                                                  nn.Conv2d(n_latent, n_latent, kernel_size=3, padding=1),
                                                  nn.ReLU(inplace=True),
                                                  nn.Conv2d(n_latent, out_f_classifier, kernel_size=1),
                                                  )

    def forward(self, x_low, x, task=None):
        x_low = self.low_level_reduce(x_low, task)

        x = self.layer5(x, task)

        x = F.interpolate(x, size=(x_low.shape[2], x_low.shape[3]), mode='bilinear', align_corners=False)

        x = torch.cat([x, x_low], dim=1)
        x = self.conv_concat(x, task)
        x = self.conv_process(x, task)

        features = x

        x = self.conv_predict[task](x)

        return x, features


def test_aspp():
    import fblib.util.visualize as viz

    tasks = ['edge']
    n_classes = {'edge': 1}

    in_channels_low = 256
    in_channels_high = 2048
    out_f_classifier = 64

    x_low = torch.rand(2, in_channels_low, 128, 128).requires_grad_()
    x_high = torch.rand(2, in_channels_high, 64, 64).requires_grad_()

    net = ASPPv3Plus(tasks=tasks,
                     n_classes=n_classes,
                     classifier='atrous-v3',
                     in_channels_high=in_channels_high,
                     in_channels_low=in_channels_low,
                     out_f_classifier=out_f_classifier,
                     norm=nn.BatchNorm2d,
                     squeeze=True)

    x_low, x_high, net = x_low.cuda(), x_high.cuda(), net.cuda()

    out = {}
    for task in tasks:
        out[task], _ = net(x_low, x_high, task=task)
        print('Task: {}, Output Shape: {}'.format(task, out[task].shape))

    g = viz.make_dot(out, net.state_dict())
    g.view(directory=Path.exp_dir())


if __name__ == '__main__':
    from fblib.util.mypath import Path
    test_aspp()
