# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad

from fblib.util.custom_container import SequentialMultiTask
import fblib.networks.classification.se_mobilenet_v2 as se_mobilenet_v2_imagenet
from fblib.networks.deeplab_multi_task.discriminators import FullyConvDiscriminator
from fblib.layers.reverse_grad import ReverseLayerF


def conv3x3_mnet(planes, stride=1, dilation=1):
    """3x3 depth-wiseconvolution with padding"""
    return nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                     dilation=dilation, padding=dilation, bias=False,
                     groups=planes)


class ConvBNMTL(nn.Module):
    """Simple 3x3 convolution, batchnorm and relu for MobileNet"""
    def __init__(self, inp, oup, stride):
        super(ConvBNMTL, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True))

    def forward(self, x, task=None):

        return self.conv(x)


class SEMobileMultiTaskDict(nn.Module):
    """SE for multiple tasks, for MobileNet"""
    def __init__(self, channel, reduction=4, tasks=None):
        super(SEMobileMultiTaskDict, self).__init__()
        self.tasks = tasks

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        if self.tasks is None:
            self.fc = nn.Sequential(nn.Linear(channel, channel // reduction),
                                    nn.ReLU6(inplace=True),
                                    nn.Linear(channel // reduction, channel),
                                    nn.Sigmoid())
        else:
            print('Initializing Mobile Squeeze and Excitation modules:')
            self.fc = nn.ModuleDict()
            for task in self.tasks:
                print('SE Mobile for task: {}'.format(task))
                self.fc[task] = SequentialMultiTask(nn.Linear(channel, channel // reduction),
                                                    nn.ReLU6(inplace=True),
                                                    nn.Linear(channel // reduction, channel),
                                                    nn.Sigmoid())

    def forward(self, x, task=None):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)

        if self.tasks:
            y = self.fc[task](y).view(b, c, 1, 1)
        else:
            y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ConvCoupledSEMnet(nn.Module):
    """
    SE-layer per task, coupled with convolutions and batchnorm.
    """

    def __init__(self, tasks,
                 process_layers=None,
                 norm_kwargs=None,
                 norm_per_task=False,
                 squeeze=False,
                 reduction=4):

        super(ConvCoupledSEMnet, self).__init__()

        self.norm_per_task = norm_per_task
        self.squeeze = squeeze

        if not isinstance(process_layers, list):
            process_layers = [process_layers]

        self.process = nn.Sequential(*process_layers)

        if self.squeeze:
            print('Initializing SE on decoder')
            self.se = SEMobileMultiTaskDict(process_layers[-1].out_channels, tasks=tasks, reduction=reduction)

        if self.norm_per_task:
            print('Initializing batchnorm per task on decoder')
            self.norm = nn.ModuleDict({task: nn.BatchNorm2d(**norm_kwargs) for task in tasks})
        else:
            self.norm = nn.BatchNorm2d(**norm_kwargs)

        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x, task):

        x = self.process(x)

        if self.norm_per_task:
            x = self.norm[task](x)
        else:
            x = self.norm(x)

        x = self.relu(x)

        if self.squeeze:
            x = self.se(x, task)

        return x


class InvResidualCommon(nn.Module):
    """Common Inverted Residual block for Mobilenet
    """
    def __init__(self, tasks, norm_per_task, hidden_dim, oup, stride, dilation=1):
        super(InvResidualCommon, self).__init__()

        self.norm_per_task = norm_per_task

        self.conv1 = nn.Conv2d(hidden_dim, hidden_dim, 3, stride, padding=dilation,
                               groups=hidden_dim, bias=False, dilation=dilation)
        if self.norm_per_task:
            self.bn1 = nn.ModuleDict({task: nn.BatchNorm2d(hidden_dim) for task in tasks})
        else:
            self.bn1 = nn.BatchNorm2d(hidden_dim)

        self.conv2 = nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)
        if self.norm_per_task:
            print('Initializing Batchnorm per task on encoder')
            self.bn2 = nn.ModuleDict({task: nn.BatchNorm2d(oup) for task in tasks})
        else:
            self.bn2 = nn.BatchNorm2d(oup)

        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x, task=None):
        x = self.conv1(x)
        if self.norm_per_task:
            x = self.bn1[task](x)
        else:
            x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        if self.norm_per_task:
            x = self.bn2[task](x)
        else:
            x = self.bn2(x)
        return x


class InvResidualExpand(nn.Module):
    """Expanding inverted residual block for Mobilenet
    """
    def __init__(self, tasks, norm_per_task, inp, hidden_dim, oup, stride, dilation=1):
        super(InvResidualExpand, self).__init__()

        self.norm_per_task = norm_per_task

        self.conv1 = nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False)
        if self.norm_per_task:
            self.bn1 = nn.ModuleDict({task: nn.BatchNorm2d(hidden_dim) for task in tasks})
        else:
            self.bn1 = nn.BatchNorm2d(hidden_dim)

        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, stride, padding=dilation,
                               groups=hidden_dim, bias=False, dilation=dilation)
        if self.norm_per_task:
            self.bn2 = nn.ModuleDict({task: nn.BatchNorm2d(hidden_dim) for task in tasks})
        else:
            self.bn2 = nn.BatchNorm2d(hidden_dim)

        self.conv3 = nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)
        if self.norm_per_task:
            print('Initializing batchnorm per task on encoder')
            self.bn3 = nn.ModuleDict({task: nn.BatchNorm2d(oup) for task in tasks})
        else:
            self.bn3 = nn.BatchNorm2d(oup)

        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x, task=None):
        x = self.conv1(x)
        if self.norm_per_task:
            x = self.bn1[task](x)
        else:
            x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        if self.norm_per_task:
            x = self.bn2[task](x)
        else:
            x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        if self.norm_per_task:
            x = self.bn3[task](x)
        else:
            x = self.bn3(x)

        return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, tasks, dilation=1,
                 norm_per_task=False, use_modulation=False):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.norm_per_task = norm_per_task

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = InvResidualCommon(tasks=tasks,
                                          norm_per_task=self.norm_per_task,
                                          hidden_dim=hidden_dim,
                                          oup=oup,
                                          stride=stride,
                                          dilation=dilation)
        else:
            self.conv = InvResidualExpand(tasks=tasks,
                                          norm_per_task=self.norm_per_task,
                                          inp=inp,
                                          hidden_dim=hidden_dim,
                                          oup=oup,
                                          stride=stride,
                                          dilation=dilation)

        if use_modulation:
            print('Initializing SE per task on encoder')
            self.se = SEMobileMultiTaskDict(tasks=tasks, channel=oup, reduction=4)
        else:
            self.se = SEMobileMultiTaskDict(tasks=None, channel=oup, reduction=4)

    def forward(self, x, task=None):
        if self.use_res_connect:
            out = self.conv(x, task)
            out = self.se(out, task)
            return x + out
        else:
            out = self.conv(x, task)
            out = self.se(out, task)
            return out


class ASPPMnet(nn.Module):
    """
    Atrous Spatial Pyramid Pooling Module (DeepLab-v3+) for mobilenet
    """

    def __init__(self, dilation_series=None, depth=64, in_f=320, tasks=None, squeeze=False,
                 norm_per_task=True):
        super(ASPPMnet, self).__init__()

        if dilation_series is None:
            dilation_series = [6, 12, 18]
        padding_series = dilation_series

        self.bnorm = nn.BatchNorm2d
        self.squeeze = squeeze

        kwargs = {"num_features": depth, "affine": True}

        self.conv2d_list = nn.ModuleList()

        # 1x1 convolution
        self.conv2d_list.append(
            ConvCoupledSEMnet(tasks=tasks,
                              process_layers=nn.Conv2d(in_f, depth, kernel_size=1, stride=1, bias=False),
                              norm_kwargs=kwargs,
                              norm_per_task=norm_per_task,
                              squeeze=self.squeeze))

        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                ConvCoupledSEMnet(tasks=tasks,
                                  process_layers=nn.Conv2d(in_f, depth, kernel_size=3, stride=1, padding=padding,
                                                           dilation=dilation, bias=False, groups=depth),
                                  norm_kwargs=kwargs,
                                  norm_per_task=norm_per_task,
                                  squeeze=self.squeeze))

        # Global features
        self.conv2d_list.append(
            ConvCoupledSEMnet(tasks=tasks,
                              process_layers=[nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                                              nn.Conv2d(in_f, depth, kernel_size=1, stride=1,
                                                        bias=False, groups=depth)],
                              norm_kwargs=kwargs,
                              norm_per_task=norm_per_task))

        self.conv2d_final = ConvCoupledSEMnet(tasks=tasks,
                                              process_layers=nn.Conv2d(depth * 5, depth, kernel_size=1,
                                                                       stride=1, bias=False, groups=depth),
                                              norm_kwargs=kwargs,
                                              norm_per_task=norm_per_task,
                                              squeeze=self.squeeze)

    def forward(self, x, task=None):
        h, w = x.size(2), x.size(3)

        interm = []
        for i in range(len(self.conv2d_list)):
            interm.append(self.conv2d_list[i](x, task))

        # Upsample the global features
        interm[-1] = F.interpolate(input=interm[-1], size=(h, w), mode='bilinear', align_corners=False)

        # Concatenate the parallel streams
        out = torch.cat(interm, dim=1)

        # Final convolutional layer of the classifier
        out = self.conv2d_final(out, task)

        return out


class ASPPDecoderMnet(nn.Module):
    """
    ASPP-v3 decoder for Mobilenet
    """

    def __init__(self,
                 tasks,
                 n_classes,
                 in_channels_high=320,
                 in_channels_low=24,
                 out_f_classifier=64,
                 atrous_rates=None,
                 norm_per_task=True,
                 squeeze=False,
                 up=4,
                 ):

        super(ASPPDecoderMnet, self).__init__()
        print('Initializing Mobilenet ASPP v3 Decoder for multiple tasks')

        if atrous_rates is None:
            atrous_rates = [6, 12, 18]

        out_f_low = int(48 * out_f_classifier / 256)
        kwargs_out = {"num_features": out_f_classifier, "affine": True}
        kwargs_low = {"num_features": out_f_low, "affine": True}

        self.up = up
        self.tasks = tasks

        print('Initializing classifier: ASPP with global features (Deeplab-v3+)')
        self.layer5 = ASPPMnet(in_f=in_channels_high,
                               depth=out_f_classifier,
                               dilation_series=atrous_rates,
                               tasks=self.tasks,
                               norm_per_task=norm_per_task,
                               squeeze=squeeze)

        self.low_level_reduce = ConvCoupledSEMnet(tasks=tasks,
                                                  process_layers=nn.Conv2d(in_channels_low, out_f_low, kernel_size=1,
                                                                           stride=1, bias=False,
                                                                           groups=math.gcd(in_channels_low, out_f_low)),
                                                  norm_kwargs=kwargs_low,
                                                  norm_per_task=norm_per_task,
                                                  squeeze=squeeze)

        self.conv_concat = ConvCoupledSEMnet(tasks=tasks,
                                             process_layers=nn.Conv2d(out_f_classifier + out_f_low, out_f_classifier,
                                                                      kernel_size=3, stride=1, bias=False, padding=1,
                                                                      groups=math.gcd(out_f_classifier + out_f_low,
                                                                                      out_f_classifier)),
                                             norm_kwargs=kwargs_out,
                                             norm_per_task=norm_per_task,
                                             squeeze=squeeze)

        self.conv_process = ConvCoupledSEMnet(tasks=tasks,
                                              process_layers=conv3x3_mnet(out_f_classifier),
                                              norm_kwargs=kwargs_out,
                                              norm_per_task=norm_per_task,
                                              squeeze=squeeze)

        self.conv_predict = nn.ModuleDict(
            {task: nn.Conv2d(out_f_classifier, n_classes[task], kernel_size=1, bias=True) for task in tasks})

    def forward(self, x_low, x, task=None):

        x_low = self.low_level_reduce(x_low, task)

        x = self.layer5(x, task)

        x = F.interpolate(x, scale_factor=self.up, mode='bilinear', align_corners=False)

        x = torch.cat((x_low, x), dim=1)

        x = self.conv_concat(x, task)
        x = self.conv_process(x, task)

        features = x

        x = self.conv_predict[task](x)

        return x, features


class SEMobileNetV2(nn.Module):
    def __init__(self, n_classes, width_mult=1., output_stride=16,
                 tasks=None, train_norm_layers=False, mod_enc=False, mod_dec=False,
                 use_dscr=False, dscr_k=1, dscr_d=2):

        super(SEMobileNetV2, self).__init__()

        self.use_dscr = use_dscr

        self.norm_per_task_enc = train_norm_layers and tasks and mod_enc
        self.norm_per_task_dec = train_norm_layers and tasks and mod_dec

        self.tasks = tasks
        self.task_dict = {x: i for i, x in enumerate(self.tasks)}

        atrous_rates = [6, 12, 18]

        if output_stride == 8:
            dilations = (2, 4)
            strides = (2, 2, 2, 1, 1)
            atrous_rates = [x * 2 for x in atrous_rates]
        elif output_stride == 16:
            dilations = (1, 2)
            strides = (2, 2, 2, 2, 1)
        else:
            raise ValueError('Choose between output_stride 8 and 16')

        block = InvertedResidual
        input_channel = 32
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16,  1, 1,          1],
            [6, 24,  2, strides[1], 1],
            [6, 32,  3, strides[2], 1],
            [6, 64,  4, strides[3], dilations[0]],
            [6, 96,  3, 1,          dilations[0]],
            [6, 160, 3, strides[4], dilations[1]],
            [6, 320, 1, 1,          dilations[1]],
        ]

        input_channel = int(input_channel * width_mult)

        # build first layer
        self.features = [ConvBNMTL(3, input_channel, strides[0])]

        # build inverted residual blocks
        for t, c, n, s, dil in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s,
                                               expand_ratio=t,
                                               dilation=dil,
                                               tasks=self.tasks,
                                               norm_per_task=self.norm_per_task_enc,
                                               use_modulation=mod_enc))
                else:
                    self.features.append(block(input_channel, output_channel, 1,
                                               expand_ratio=t,
                                               dilation=dil,
                                               tasks=self.tasks,
                                               norm_per_task=self.norm_per_task_enc,
                                               use_modulation=mod_enc))
                input_channel = output_channel

        # make it nn.Sequential
        self.features = SequentialMultiTask(*self.features)
        self.features_low = self.features[:4]
        self.features_high = self.features[4:]

        self.decoder = ASPPDecoderMnet(n_classes=n_classes,
                                       in_channels_high=320,
                                       in_channels_low=24,
                                       out_f_classifier=64,
                                       atrous_rates=atrous_rates,
                                       tasks=self.tasks,
                                       norm_per_task=self.norm_per_task_dec,
                                       squeeze=mod_dec)

        if self.use_dscr:
            self.dscr_k = dscr_k
            self.dscr_d = dscr_d
            self.task_label_shape = (128, 128)
            print('Using Discriminator with kernel size: {} and depth: {}'.format(self.dscr_k, self.dscr_d))
            self.discriminator = self._get_discriminator(width_decoder=64)
            self.rev_layer = ReverseLayerF()
            self.criterion_classifier = torch.nn.CrossEntropyLoss(ignore_index=255)

        self._initialize_weights()
        self._verify_bnorm_params()

    def forward(self, x, task=None):

        in_shape = x.shape[2:]

        x_low = self.features_low(x, task)
        x = self.features_high(x_low, task)

        x, features = self.decoder(x_low, x, task)

        x = F.interpolate(x, size=(in_shape[-2], in_shape[-1]), mode='bilinear', align_corners=False)

        return x, features

    def compute_losses(self, outputs, features, criteria, gt_elems, alpha, p):
        """
        Computes losses for tasks, losses for discriminator, output of discriminator, and gradients
        """
        # Compute classification losses and gradients wrt features
        tasks = outputs.keys()

        grads = {}
        losses_tasks = {}
        task_labels = {}
        outputs_dscr = {}
        losses_dscr = {}

        with torch.enable_grad():
            for task in tasks:

                curr_loss = p.TASKS.LOSS_MULT[task] * criteria[task](outputs[task], gt_elems[task])
                losses_tasks[task] = curr_loss

                if self.use_dscr:
                    # Create task labels
                    task_labels[task] = self._create_task_labels(gt_elems, task).to(outputs[task].device)

                    # Compute Gradients
                    grads[task] = grad(curr_loss, features[task], create_graph=True)[0]
                    grads_norm = grads[task].norm(p=2, dim=1).unsqueeze(1) + 1e-10
                    input_dscr = grads[task] / grads_norm
                    input_dscr = self.rev_layer.apply(input_dscr, alpha)

                    outputs_dscr[task] = self.discriminator(input_dscr)

                    losses_dscr[task] = self.criterion_classifier(outputs_dscr[task], task_labels[task])

        return losses_tasks, losses_dscr, outputs_dscr, grads, task_labels

    def _create_task_labels(self, gt_elems, task):

        valid = gt_elems[task].detach().clone()
        valid = F.interpolate(valid, size=self.task_label_shape, mode='nearest')
        valid[valid != 255] = self.task_dict[task]
        valid = valid[:, 0, :, :]

        return valid.long()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def _verify_bnorm_params(self):
        verify_trainable = True
        a = 0
        for x in self.modules():
            if isinstance(x, nn.BatchNorm2d):
                for y in x.parameters():
                    verify_trainable = (verify_trainable and y.requires_grad)
                a += isinstance(x, nn.BatchNorm2d)

        print("\nVerification: Trainable batchnorm parameters? Answer: {}\n".format(verify_trainable))
        print("Asynchronous bnorm layers: {}".format(a))

    def _get_discriminator(self, width_decoder):
        discriminator = FullyConvDiscriminator(in_channels=width_decoder, n_classes=len(self.tasks),
                                               kernel_size=self.dscr_k, depth=self.dscr_d)

        return discriminator

    def _define_if_copyable(self, module):
        is_copyable = isinstance(module, nn.Conv2d) \
                      or isinstance(module, nn.Linear) \
                      or isinstance(module, nn.BatchNorm2d) or \
                      isinstance(module, nn.BatchNorm2d)
        return is_copyable

    def _exists_task_in_name(self, layer_name):
        for task in self.tasks:
            if layer_name.find(task) > 0:
                return task

        return None

    def load_pretrained(self, base_network):

        copy_trg = {}
        for (name_trg, module_trg) in self.named_modules():
            if self._define_if_copyable(module_trg):
                copy_trg[name_trg] = module_trg

        copy_src = {}
        for (name_src, module_src) in base_network.named_modules():
            if self._define_if_copyable(module_src):
                copy_src[name_src] = module_src

        task_specific_counter = 0
        mapping = {}
        for name_trg in copy_trg:
            # Handle first layers
            if 'decoder' in name_trg:
                continue
            elif 'discriminator' in name_trg:
                continue
            elif name_trg in copy_src:
                map_trg = name_trg
            elif 'features.0' in name_trg:
                map_trg = name_trg.replace('.conv', '')
            elif '.conv1' in name_trg:
                map_trg = name_trg.replace('.conv1', '.0')
            elif '.bn1' in name_trg:
                map_trg = name_trg.replace('.bn1', '.1')
            elif '.conv2' in name_trg:
                map_trg = name_trg.replace('.conv2', '.3')
            elif '.bn2' in name_trg:
                map_trg = name_trg.replace('.bn2', '.4')
            elif '.conv3' in name_trg:
                map_trg = name_trg.replace('.conv3', '.6')
            elif '.bn3' in name_trg:
                map_trg = name_trg.replace('.bn3', '.7')

            elif self._exists_task_in_name(name_trg):
                # Handle SE layers
                task = self._exists_task_in_name(name_trg)
                name_src = name_trg.replace('.' + task, '')

                if name_src in copy_src:
                    map_trg = name_src
                    task_specific_counter += 1

            else:
                raise ValueError('Unknown module name found: {}'.format(name_trg))
            # Handle BatchNom2d layers
            task = self._exists_task_in_name(map_trg)
            if task:
                map_trg = map_trg.replace('.' + task, '')

            mapping[name_trg] = map_trg

        i = 0
        for name in mapping:
            module_trg = copy_trg[name]
            module_src = copy_src[mapping[name]]

            if module_trg.weight.data.shape != module_src.weight.data.shape:
                print('skipping layer with size: {} and target size: {}'
                      .format(module_trg.weight.data.shape, module_src.weight.data.shape))
                continue

            if isinstance(module_trg, nn.Conv2d) and isinstance(module_src, nn.Conv2d):
                module_trg.weight.data = module_src.weight.data.clone()
                if module_src.bias is not None:
                    module_trg.bias = module_src.bias.clone()
                i += 1

            elif isinstance(module_trg, nn.BatchNorm2d) and isinstance(module_src, nn.BatchNorm2d):

                # copy running mean and variance of batchnorm layers!
                module_trg.running_mean.data = module_src.running_mean.data.clone()
                module_trg.running_var.data = module_src.running_var.data.clone()

                module_trg.weight.data = module_src.weight.data.clone()
                module_trg.bias.data = module_src.bias.data.clone()
                i += 1

            elif isinstance(module_trg, nn.Linear) and (isinstance(module_src, nn.Linear)):
                module_trg.weight.data = module_src.weight.data.clone()
                module_trg.bias.data = module_src.bias.data.clone()
                i += 1

        print('\nContents of {} out of {} layers successfully copied, including {} task-specific layers\n'
              .format(i, len(mapping), task_specific_counter))


def get_lr_params(model, part='all', tasks=None):
    """
    This generator returns all the parameters of the backbone
    """

    def ismember(layer_txt, tasks):
        exists = False
        for task in tasks:
            exists = exists or layer_txt.find(task) > 0
        return exists

    # for multi-GPU training
    if hasattr(model, 'module'):
        model = model.module

    if part == 'all':
        b = [model]
    elif part == 'backbone':
        b = [model.features]
    elif part == 'decoder':
        b = [model.decoder]
    elif part == 'generic':
        b = [model]
    elif part == 'task_specific':
        b = [model]
    elif part == 'discriminator':
        b = [model.discriminator]

    for i in range(len(b)):
        for name, k in b[i].named_parameters():
            if k.requires_grad:
                if part == 'generic' or part == 'decoder' or part == 'backbone':
                    if ismember(name, tasks):
                        continue
                elif part == 'task_specific':
                    if not ismember(name, tasks):
                        continue
                yield k


def se_mobilenet_v2(pretrained='scratch', **kwargs):

    model = SEMobileNetV2(**kwargs)

    if pretrained == 'imagenet':

        print('loading pre-trained imagenet model')
        model_full = se_mobilenet_v2_imagenet.se_mobilenet_v2(pretrained=True)
        model.load_pretrained(model_full)
    elif pretrained == 'scratch':
        print('using imagenet initialized from scratch')
    else:
        raise NotImplementedError('select either imagenet or scratch for pre-training')

    return model


def main():

    import fblib.util.pdf_visualizer as viz

    elems = ['image', 'semseg', 'edge']
    tasks = ['semseg', 'edge']
    net = se_mobilenet_v2(pretrained='imagenet',
                          n_classes={'edge': 1, 'semseg': 21},
                          tasks=tasks,
                          mod_enc=True,
                          mod_dec=True,
                          train_norm_layers=True)

    net.cuda()
    net.eval()

    img = torch.rand(2, 3, 512, 512)
    img = img.cuda()
    y = {}
    for task in tasks:
        y[task], _ = net.forward(img, task=elems[-1])

    g = viz.make_dot(y, net.state_dict())
    g.view(directory='./')


if __name__ == '__main__':
    main()
