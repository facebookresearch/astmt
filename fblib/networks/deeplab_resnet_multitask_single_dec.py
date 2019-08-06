import os
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
from torch.autograd import grad
import torchvision.models.resnet as resnet

from fblib.networks.classifiers_multitask import AtrousSpatialPyramidPoolingModule, ConvClassifier
import fblib.networks.resnet_imagenet as resnet_imagenet
import fblib.networks.mobilenet_v2 as mobilenet_v2
from fblib.layers.image_features import ImageFeatures
from fblib.layers.misc_layers import interp_surgery
from fblib.networks import resnet as custom_resnet
from encoding.nn import BatchNorm2d as SyncBatchNorm2d
from fblib.layers.squeeze import ConvCoupledSE
from fblib.networks.discriminators import FullyConvDiscriminator, AvePoolDiscriminator, ConvDiscriminator
from fblib.layers.reverse_grad import ReverseLayerF


try:
    from itertools import izip
except ImportError:  # python3.x
    izip = zip

affine_par = True


def get_ngroups_gn(dim):
    """
    Get number of groups used by groupnorm, based on number of channels
    """
    n_lay_per_group_low = 16
    n_lay_per_group = 32
    if dim <= 256:
        assert (dim % n_lay_per_group_low == 0)
        return int(dim / n_lay_per_group_low)
    else:
        assert (dim % n_lay_per_group == 0)
        return int(dim / n_lay_per_group)


def outS(i):
    i = int(i)
    i = (i + 1) / 2
    i = int(np.ceil((i + 1) / 2.0))
    i = (i + 1) / 2
    return i


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     dilation=dilation, padding=dilation, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation_=1, downsample=None, train_norm_layers=False,
                 sync_bnorm=False):
        super(BasicBlock, self).__init__()
        self.bnorm = nn.BatchNorm2d if not sync_bnorm else SyncBatchNorm2d

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = self.bnorm(planes)
        for i in self.bn1.parameters():
            i.requires_grad = train_norm_layers
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation_)
        self.bn2 = self.bnorm(planes)
        for i in self.bn2.parameters():
            i.requires_grad = train_norm_layers
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation_=1, downsample=None, train_norm_layers=False,
                 sync_bnorm=False):
        super(Bottleneck, self).__init__()

        self.bnorm = nn.BatchNorm2d if not sync_bnorm else SyncBatchNorm2d

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)  # change
        self.bn1 = self.bnorm(planes, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = train_norm_layers

        padding = dilation_

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,  # change
                               padding=padding, bias=False, dilation=dilation_)
        self.bn2 = self.bnorm(planes, affine=affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = train_norm_layers
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = self.bnorm(planes * 4, affine=affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = train_norm_layers
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, n_classes, nInputChannels=3, classifier='atrous-v3',
                 output_stride=8, groupnorm=False, tasks=None, train_norm_layers=False,
                 sync_bnorm=False, out_f_classifier=256, dscr_type='conv',
                 squeeze=False, norm_per_task=False, se_after_relu=False, conv_se=False,
                 im_features=None, trans='add', train_im_features=False):

        super(ResNet, self).__init__()

        print("Constructing ResNet model...")
        print("Output stride: {}".format(output_stride))
        print("Number of classes: {}".format(n_classes))
        print("Number of Input Channels: {}".format(nInputChannels))

        v3_atrous_rates = [6, 12, 18]

        if output_stride == 8:
            dilations = (2, 4)
            strides = (2, 2, 2, 1, 1)
            v3_atrous_rates = [x * 2 for x in v3_atrous_rates]
        elif output_stride == 16:
            dilations = (1, 2)
            strides = (2, 2, 2, 2, 1)
        else:
            raise ValueError('Choose between output_stride 8 and 16')

        self.inplanes = 64
        self.classifier = classifier
        self.groupnorm = groupnorm
        self.train_norm_layers = train_norm_layers
        self.sync_bnorm = sync_bnorm
        self.bnorm = nn.BatchNorm2d if not self.sync_bnorm else SyncBatchNorm2d
        self.squeeze = squeeze
        self.conv_se = conv_se

        # Image Features parameters
        self.im_features = im_features
        self.trans = trans
        self.train_im_features = train_im_features

        self.tasks = tasks
        self.task_dict = {x: i for i, x in enumerate(self.tasks)}

        # Network structure
        self.conv1 = nn.Conv2d(nInputChannels, 64, kernel_size=7, stride=strides[0], padding=3,
                               bias=False)
        self.bn1 = self.bnorm(64, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = self.train_norm_layers
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=strides[1], padding=1, ceil_mode=False)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[2])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[3], dilation__=dilations[0])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[4], dilation__=dilations[1])

        out_f_low = 48 * out_f_classifier / 256  # Adapt in case of thinner classifiers
        assert (int(out_f_low) == out_f_low)

        if not self.groupnorm:
            NormModule = self.bnorm
            kwargs_low = {"num_features": int(out_f_low), "affine": affine_par}
            kwargs_out = {"num_features": out_f_classifier, "affine": affine_par}
        else:
            NormModule = nn.GroupNorm
            kwargs_low = {"num_groups": get_ngroups_gn(out_f_low), "num_channels": out_f_low, "affine": affine_par}
            kwargs_out = {"num_groups": get_ngroups_gn(out_f_classifier), "num_channels": out_f_classifier,
                          "affine": affine_par}

        print('Using decoder')
        self.decoder = nn.Module()
        if classifier == 'atrous-v3':
            print('Initializing classifier: A-trous with global features (Deeplab-v3+)')
            self.decoder.layer5 = AtrousSpatialPyramidPoolingModule(depth=out_f_classifier,
                                                                    groupnorm=self.groupnorm,
                                                                    dilation_series=v3_atrous_rates,
                                                                    sync_bnorm=self.sync_bnorm,
                                                                    tasks=self.tasks,
                                                                    squeeze=self.squeeze,
                                                                    se_after_relu=se_after_relu,
                                                                    norm_per_task=norm_per_task,
                                                                    conv_se=self.conv_se)
        elif classifier == 'conv':
            print('Using simple convolutional classifier')
            self.decoder.layer5 = ConvClassifier(depth=out_f_classifier,
                                                 groupnorm=self.groupnorm,
                                                 sync_bnorm=self.sync_bnorm,
                                                 tasks=self.tasks,
                                                 squeeze=self.squeeze,
                                                 se_after_relu=se_after_relu,
                                                 norm_per_task=norm_per_task,
                                                 conv_se=self.conv_se)
        else:
            raise NotImplementedError('Choose one of the 3 available classifiers')

        self.decoder.low_level_reduce = ConvCoupledSE(tasks=tasks,
                                                      process_layers=nn.Conv2d(256, int(out_f_low), kernel_size=1,
                                                                               bias=False),
                                                      norm=NormModule,
                                                      norm_kwargs=kwargs_low,
                                                      norm_per_task=norm_per_task,
                                                      squeeze=self.squeeze,
                                                      se_after_relu=se_after_relu,
                                                      reduction=2,  # Too few features if we keep default reduction (16)
                                                      fully_conv=self.conv_se)
        self.decoder.conv_concat = ConvCoupledSE(tasks=tasks,
                                                 process_layers=conv3x3(out_f_classifier + int(out_f_low),
                                                                        out_f_classifier),
                                                 norm=NormModule,
                                                 norm_kwargs=kwargs_out,
                                                 norm_per_task=norm_per_task,
                                                 squeeze=self.squeeze,
                                                 se_after_relu=se_after_relu,
                                                 fully_conv=self.conv_se)

        self.decoder.conv_process = ConvCoupledSE(tasks=tasks,
                                                  process_layers=conv3x3(out_f_classifier, out_f_classifier),
                                                  norm=NormModule,
                                                  norm_kwargs=kwargs_out,
                                                  norm_per_task=norm_per_task,
                                                  squeeze=self.squeeze,
                                                  se_after_relu=se_after_relu,
                                                  fully_conv=self.conv_se)

        self.decoder.conv_predict = nn.ModuleDict(
            {task: nn.Conv2d(out_f_classifier, n_classes[task], kernel_size=1, bias=True) for task in tasks})

        if self.use_dscr:
            print('Using Discriminator type: {}'.format(dscr_type))
            self.task_label_shape = None
            if dscr_type == 'conv':
                self.discriminator = ConvDiscriminator(in_channels=out_f_classifier, n_classes=len(tasks))
            elif dscr_type == 'avepool':
                self.discriminator = AvePoolDiscriminator(in_channels=out_f_classifier, n_classes=len(tasks))
            elif dscr_type == 'fconv':
                self.discriminator = FullyConvDiscriminator(in_channels=out_f_classifier, n_classes=len(tasks))
                self.task_label_shape = (128, 128)
            else:
                raise NotImplementedError

            self.rev_layer = ReverseLayerF()
            self.criterion_classifier = torch.nn.CrossEntropyLoss(ignore_index=255)

        # Initialize weights
        self._initialize_weights()

        # Load pre-trained global network after weight Initialization
        if self.im_features:
            print('\nUsing global features for conditional modulation:')
            if self.im_features == 'res26':
                print('ResNet-26')
                imagenet_network = resnet_imagenet.resnet26
            elif self.im_features == 'res50':
                print('ResNet-50')
                imagenet_network = resnet_imagenet.resnet50
            elif self.im_features == 'res101':
                print('ResNet-101')
                imagenet_network = resnet_imagenet.resnet101
            elif self.im_features == 'mobilenet':
                imagenet_network = mobilenet_v2.mobilenet_v2
                self.decoder.reduce_features = nn.Conv2d(1280, 2048, kernel_size=1, bias=True)
            else:
                raise NotImplementedError

            global_net = imagenet_network(pretrained=True, features=True)
            self.global_net = ImageFeatures(global_net)

            for i in self.global_net.parameters():
                i.requires_grad = self.train_im_features

        # Check if batchnorm parameters are trainable
        self._verify_bnorm_params()

    def _make_layer(self, block, planes, blocks, stride=1, dilation__=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation__ == 2 or dilation__ == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                self.bnorm(planes * block.expansion, affine=affine_par),
            )

            # Train batchnorm?
            for i in downsample._modules['1'].parameters():
                i.requires_grad = self.train_norm_layers

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation_=dilation__, downsample=downsample,
                            train_norm_layers=self.train_norm_layers, sync_bnorm=self.sync_bnorm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation_=dilation__,
                                train_norm_layers=self.train_norm_layers, sync_bnorm=self.sync_bnorm))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, self.bnorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.zero_()
                m.weight.data = interp_surgery(m)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def _verify_bnorm_params(self):
        verify_trainable = True
        a = s = 0
        for x in self.modules():
            if isinstance(x, nn.BatchNorm2d) or isinstance(x, SyncBatchNorm2d):
                for y in x.parameters():
                    verify_trainable = (verify_trainable and y.requires_grad)
                a += isinstance(x, nn.BatchNorm2d)
                s += isinstance(x, SyncBatchNorm2d)

        print("\nVerification: Trainable batchnorm parameters? Answer: {}\n".format(verify_trainable))
        print("Asynchronous bnorm layers: {}".format(a))
        print("Synchronous bnorm layers: {}".format(s))

    def forward(self, x_in, task_gts=None):

        x = self.conv1(x_in)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)

        x_low = x

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.im_features:
            # Very important: batchnorms must be in eval() mode during inference!
            self.global_net.eval()

            # Compute global image features and modulate the signal
            if not self.train_im_features:
                # Save some memory
                with torch.no_grad():
                    _, image_features = self.global_net(x_in)
            else:
                _, image_features = self.global_net(x_in)

            image_features = image_features.unsqueeze(2).unsqueeze(3)

            if hasattr(self.decoder, 'reduce_features'):
                # image_features = image_features.requires_grad_()
                image_features = self.decoder.reduce_features(image_features)
                print(self.decoder.reduce_features.weight[-1].detach().cpu().numpy())

            if self.trans == 'mul':
                x = torch.mul(x, image_features)
            elif self.trans == 'add':
                x = torch.add(x, image_features)

        out = {}

        dec = self.decoder
        if not self.squeeze:
            x = dec.layer5(x)
            x = F.interpolate(x,
                              size=(x_low.shape[2], x_low.shape[3]),
                              mode='bilinear',
                              align_corners=False)
            x_low = dec.low_level_reduce(x_low, task=None)
            x = torch.cat([x, x_low], dim=1)
            x = dec.conv_concat(x, task=None)
            x = dec.conv_process(x, task=None)
            for task in self.tasks:
                if self._chk_forward(task, task_gts):
                    out[task] = dec.conv_predict[task](x)

        else:
            for task in self.tasks:
                if self._chk_forward(task, task_gts):
                    x_task = dec.layer5(x, task)
                    x_task = F.interpolate(x_task,
                                           size=(x_low.shape[2], x_low.shape[3]),
                                           mode='bilinear',
                                           align_corners=False)
                    x_low_task = dec.low_level_reduce(x_low, task)
                    x_task = torch.cat([x_task, x_low_task], dim=1)
                    x_task = dec.conv_concat(x_task, task)
                    x_task = dec.conv_process(x_task, task)
                    out[task] = dec.conv_predict[task](x_task)

        return out

    @staticmethod
    def _chk_forward(task, meta):
        if meta is None:
            return True
        else:
            return task in meta

    def compute_losses_dscr(self, outputs, features, criteria, gt_elems, alpha, p):
        """
        Computes losses for tasks, losses for discriminator, output of discriminator, and gradients
        """
        # Compute classification losses and gradients wrt features
        tasks = outputs.keys()

        grads = {}
        losses_tasks = {}
        task_labels = {}
        with torch.enable_grad():
            for task in tasks:

                # Create task labels
                task_labels[task] = self._create_task_labels(gt_elems, task).to(p['device'])

                curr_loss = p.TASKS.LOSS_MULT[task] * criteria[task](outputs[task], gt_elems[task])
                losses_tasks[task] = curr_loss

                grads[task] = grad(curr_loss, features[task], create_graph=True)[0]

            # Compute norm of gradients
            grads_norm = {task: grads[task].norm(p=2, dim=1) + 1e-10 for task in tasks}

            if p['avenrm']:
                # print('Dividing by the average norm accross all tasks')
                grads_norm = torch.cat([grads_norm[task] for task in tasks]).mean(dim=0)
                input_dscr = {task: grads[task] / grads_norm.unsqueeze(0).unsqueeze(0) for task in tasks}
            else:
                input_dscr = {task: grads[task] / grads_norm[task].unsqueeze(1) for task in tasks}

            # Compute discriminator loss
            outputs_dscr = {}
            losses_dscr = {}
            for task in tasks:
                input_dscr[task] = self.rev_layer.apply(input_dscr[task], alpha)

                output_dscr = self.discriminator(input_dscr[task])

                curr_loss_dscr = self.criterion_classifier(output_dscr, task_labels[task])
                losses_dscr[task] = curr_loss_dscr
                outputs_dscr[task] = output_dscr

        # print(input_dscr)
        # print(outputs_dscr)
        return losses_tasks, losses_dscr, outputs_dscr, grads, task_labels

    def load_pretrained(self, base_network, nInputChannels=3):
        flag = 0
        i = 0
        for module, module_ori in izip(self.modules(), base_network.modules()):
            if isinstance(module, nn.Conv2d) and isinstance(module_ori, nn.Conv2d):
                if not flag and nInputChannels != 3:
                    module.weight[:, :3, :, :].data = deepcopy(module_ori.weight.data)
                    module.bias = deepcopy(module_ori.bias)
                    for i in range(3, int(module.weight.data.shape[1])):
                        module.weight[:, i, :, :].data = deepcopy(
                            module_ori.weight[:, -1, :, :][:, np.newaxis, :, :].data)
                    flag = 1
                elif module.weight.data.shape == module_ori.weight.data.shape:
                    i += 1
                    module.weight.data = deepcopy(module_ori.weight.data)
                    module.bias = deepcopy(module_ori.bias)
                else:
                    print('Skipping Conv layer with size: {} and target size: {}'
                          .format(module.weight.data.shape, module_ori.weight.data.shape))
            elif isinstance(module, self.bnorm) and (isinstance(module_ori, nn.BatchNorm2d)
                                                     or isinstance(module_ori, self.bnorm)):
                if module.weight.data.shape == module_ori.weight.data.shape:
                    i += 1

                    # Copy running mean and variance of batchnorm layers!
                    module.running_mean.data = deepcopy(module_ori.running_mean.data)
                    module.running_var.data = deepcopy(module_ori.running_var.data)

                    module.weight.data = deepcopy(module_ori.weight.data)
                    module.bias.data = deepcopy(module_ori.bias.data)
                else:
                    print('Skipping Batchnorm layer with size: {} and target size: {}'
                          .format(module.weight.data.shape, module_ori.weight.data.shape))
        print("Content of {} layers successfully copied.".format(i))

    def load_pretrained_ms(self, base_network, nInputChannels=3):
        flag = 0
        for module, module_ori in izip(self.modules(), base_network.Scale.modules()):
            if isinstance(module, nn.Conv2d) and isinstance(module_ori, nn.Conv2d):
                if not flag and nInputChannels != 3:
                    module.weight[:, :3, :, :].data = deepcopy(module_ori.weight.data)
                    module.bias = deepcopy(module_ori.bias)
                    for i in range(3, int(module.weight.data.shape[1])):
                        module.weight[:, i, :, :].data = deepcopy(
                            module_ori.weight[:, -1, :, :][:, np.newaxis, :, :].data)
                    flag = 1
                elif module.weight.data.shape == module_ori.weight.data.shape:
                    module.weight.data = deepcopy(module_ori.weight.data)
                    module.bias = deepcopy(module_ori.bias)
                else:
                    print('Skipping Conv layer with size: {} and target size: {}'
                          .format(module.weight.data.shape, module_ori.weight.data.shape))
            elif isinstance(module, nn.BatchNorm2d) and isinstance(module_ori, nn.BatchNorm2d) \
                    and module.weight.data.shape == module_ori.weight.data.shape:

                # Copy running mean and variance of batchnorm layers!
                module.running_mean.data = deepcopy(module_ori.running_mean.data)
                module.running_var.data = deepcopy(module_ori.running_var.data)

                module.weight.data = deepcopy(module_ori.weight.data)
                module.bias.data = deepcopy(module_ori.bias.data)


class MS_Deeplab(nn.Module):
    def __init__(self, block, NoLabels, nInputChannels=3):
        super(MS_Deeplab, self).__init__()
        self.Scale = ResNet(block, [3, 4, 23, 3], NoLabels, nInputChannels=nInputChannels)

    def forward(self, x):
        input_size = x.size()[2]
        self.interp1 = nn.Upsample(size=(int(input_size * 0.75) + 1, int(input_size * 0.75) + 1), mode='bilinear')
        self.interp2 = nn.Upsample(size=(int(input_size * 0.5) + 1, int(input_size * 0.5) + 1), mode='bilinear')
        self.interp3 = nn.Upsample(size=(outS(input_size), outS(input_size)), mode='bilinear')
        out = []
        x2 = self.interp1(x)
        x3 = self.interp2(x)
        out.append(self.Scale(x))  # for original scale
        out.append(self.interp3(self.Scale(x2)))  # for 0.75x scale
        out.append(self.Scale(x3))  # for 0.5x scale

        x2Out_interp = out[1]
        x3Out_interp = self.interp3(out[2])
        temp1 = torch.max(out[0], x2Out_interp)
        out.append(torch.max(temp1, x3Out_interp))
        return out[-1]


def Res_Deeplab(n_classes=21, pretrained=None):
    model = MS_Deeplab(Bottleneck, n_classes)
    if pretrained is not None:
        if pretrained == 'voc':
            pth_model = 'MS_DeepLab_resnet_trained_VOC.pth'
        elif pretrained == 'ms_coco':
            pth_model = 'MS_DeepLab_resnet_pretrained_COCO_init.pth'
        saved_state_dict = torch.load(os.path.join(Path.models_dir(), pth_model),
                                      map_location=lambda storage, loc: storage)
        if n_classes != 21:
            for i in saved_state_dict:
                i_parts = i.split('.')
                if i_parts[1] == 'layer5':
                    saved_state_dict[i] = model.state_dict()[i]
        model.load_state_dict(saved_state_dict)
    return model


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
        b = [model.conv1, model.bn1, model.layer1, model.layer2, model.layer3, model.layer4]
    elif part == 'decoder':
        b = [model.decoder]
    elif part == 'generic':
        b = [model]
    elif part == 'task_specific':
        b = [model]
    elif part == 'global_feat':
        b = [model.global_net]

    for i in range(len(b)):
        for name, k in b[i].named_parameters():
            if k.requires_grad:
                if part == 'generic' or part == 'decoder':
                    if ismember(name, tasks):
                        continue
                elif part == 'task_specific':
                    if not ismember(name, tasks):
                        continue
                yield k


def resnet26(n_classes, pretrained='scratch', nInputChannels=3, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (str): If 'imagenet', returns a model pre-trained on ImageNet
    """

    print('Constructing ResNet18')
    model = ResNet(Bottleneck, [2, 2, 2, 2], n_classes, nInputChannels=nInputChannels, **kwargs)

    if pretrained == 'imagenet':
        model_full = custom_resnet.resnet26(pretrained=True)
        model.load_pretrained(model_full, nInputChannels=nInputChannels)
    elif pretrained == 'scratch':
        print('Training from scratch')
    else:
        raise NotImplementedError('Please specify scratch or imagenet for pre-training')
    return model


def resnet50(n_classes, pretrained='scratch', nInputChannels=3, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (str): If 'imagenet', returns a model pre-trained on ImageNet
    """

    print('Constructing ResNet50')
    model = ResNet(Bottleneck, [3, 4, 6, 3], n_classes, nInputChannels=nInputChannels, **kwargs)
    if pretrained == 'imagenet':
        model_full = resnet.resnet50(pretrained=True)
        model.load_pretrained(model_full, nInputChannels=nInputChannels)
    return model


def resnet101(n_classes, pretrained='scratch', nInputChannels=3, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained ('imagenet', 'voc', 'ms_coco): Select model trained on respective dataset.
    """

    print('Constructing ResNet101')

    model = ResNet(Bottleneck, [3, 4, 23, 3], n_classes, nInputChannels=nInputChannels, **kwargs)
    if pretrained == 'imagenet':
        print('Initializing from pre-trained ImageNet model..')
        model_full = resnet.resnet101(pretrained=True)
        model.load_pretrained(model_full, nInputChannels=nInputChannels)
    elif pretrained == 'ms_coco' or pretrained == 'voc':
        model_full = Res_Deeplab(n_classes, pretrained=pretrained)
        model.load_pretrained_ms(model_full, nInputChannels=nInputChannels)
    return model


def test_vis_net(net, elems):

    net.cuda()

    # Define the transformations
    transform = transforms.Compose(
        [tr.RandomHorizontalFlip(),
         tr.FixedResize(resolutions={x: (512, 512) for x in elems},
                        flagvals={x: cv2.INTER_NEAREST for x in elems}),
         tr.ToTensor()])

    # Define dataset, tasks, and the dataloader
    dataset = PASCALContext(split=['train'], transform=transform, retname=True,
                            do_edge=True,
                            do_human_parts=True,
                            do_semseg=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=False, num_workers=0)

    net.train()
    for ii, sample in enumerate(dataloader):
        img = sample['image']
        task_gts = list(sample.keys())
        img = img.cuda()
        y = net.forward(img, task_gts=task_gts)
        g = viz.make_dot(y, net.state_dict())
        g.view()

        break


def test_gflops(net, elems):
    from fblib.util.model_resources.flops import compute_gflops

    n = len(elems) - 1
    gflops_n_tasks = compute_gflops(net, in_shape=(2, 3, 256, 256), tasks=elems[1:])
    gflops_1_task = compute_gflops(net, in_shape=(2, 3, 256, 256), tasks=elems[1])
    gflops_decoder = (gflops_n_tasks - gflops_1_task) / (n - 1)
    gflops_encoder = gflops_1_task - gflops_decoder

    print('GFLOPS for {} tasks: {}'.format(n, gflops_n_tasks))
    print('GFLOPS for 1 task: {}'.format(gflops_1_task))
    print('GFLOPS for encoder: {}'.format(gflops_encoder))
    print('GFLOPS for task specific layers (decoder): {}'.format(gflops_decoder))


def test_lr_params(net, tasks):
    params = get_lr_params(net, part='generic', tasks=tasks)
    print(params)


def test_imagenet_features(net):
    import os
    import torch
    import pickle
    import cv2
    import numpy as np
    import urllib.request
    from fblib import PROJECT_ROOT_DIR
    classes = pickle.load(urllib.request.urlopen(
        'https://gist.githubusercontent.com/yrevar/6135f1bd8dcf2e0cc683/raw/d133d61a09d7e5a3b36b8c111a8dd5c4b5d560ee'
        '/imagenet1000_clsid_to_human.pkl'))

    img = cv2.imread(os.path.join(PROJECT_ROOT_DIR, 'util/img/cat.jpg')).astype(np.float32)

    img = img[:, :, :, np.newaxis]
    img = img.transpose((3, 2, 0, 1))
    img = torch.from_numpy(img.astype(np.float32))

    net.eval()
    with torch.no_grad():
        outputs, features = net.global_net(img)
        outputs = torch.nn.functional.softmax(outputs, dim=1)

        for j in range(outputs.shape[0]):
            output = outputs[j, :]
            print(output.max())
            print(output.argmax())
            print(classes[np.asscalar(output.argmax().numpy())])


if __name__ == '__main__':
    import cv2
    from fblib.util.mypath import Path
    from fblib.dataloaders import custom_transforms as tr
    from torchvision import transforms
    from fblib.dataloaders.pascal_context import PASCALContext
    from torch.utils.data import DataLoader
    import fblib.util.visualize as viz

    elems = ['image', 'edge', 'human_parts']
    squeeze = False
    out_f_classifier = 64
    norm_per_task = False
    se_after_relu = False
    im_features = 'res101'

    # Load Network
    net = resnet26(n_classes={'edge': 1, 'human_parts': 7}, pretrained='imagenet', nInputChannels=3,
                   classifier='atrous-v3', output_stride=8, groupnorm=False, tasks=elems[1:],
                   out_f_classifier=out_f_classifier,
                   squeeze=squeeze,
                   norm_per_task=norm_per_task,
                   se_after_relu=se_after_relu,
                   conv_se=False,
                   im_features=im_features)

    # test_imagenet_features(net)
    test_vis_net(net, elems)
    # test_gflops(net, elems)
    # test_lr_params(net, tasks)
