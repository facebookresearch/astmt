# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from copy import deepcopy

import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
from torch.autograd import grad

import fblib.networks.classification.se_resnet as se_resnet
from fblib.networks.deeplab_multi_task.decoders import ASPPv3Plus, UbernetDecoder
from fblib.networks.deeplab_multi_task.discriminators import FullyConvDiscriminator
from fblib.layers.reverse_grad import ReverseLayerF
from fblib.layers.squeeze import SELayerMultiTaskDict
from fblib.util.custom_container import SequentialMultiTask

try:
    from itertools import izip
except ImportError:  # python3.x
    izip = zip

affine_par = True


class ConvCoupledBatchNormMT(nn.Module):
    def __init__(self,
                 tasks=None,
                 process_layer=None,
                 norm=nn.BatchNorm2d,
                 norm_kwargs=None,
                 train_norm=False):
        super(ConvCoupledBatchNormMT, self).__init__()

        self.tasks = tasks

        # Processing layer
        self.process = process_layer

        # Batch Norm layer(s)
        if tasks is not None:
            print('Using per-task batchnorm parameters in Encoder: Downsampling')
            self.norm = nn.ModuleDict({task: norm(**norm_kwargs) for task in self.tasks})
        else:
            self.norm = norm(**norm_kwargs)

        # Define whether batchnorm parameters are trainable
        for i in self.norm.parameters():
            i.requires_grad = train_norm

    def forward(self, x, task=None):
        x = self.process(x)

        if self.tasks is None:
            x = self.norm(x)
        else:
            x = self.norm[task](x)

        return x


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, train_norm_layers=False,
                 reduction=16, tasks=None, squeeze_enc=True, adapters=False):
        super(SEBottleneck, self).__init__()
        self.adapters = adapters
        self.per_task_norm_layers = train_norm_layers and tasks
        padding = dilation
        self.bnorm = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)  # change
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,  # change
                               padding=padding, bias=False, dilation=dilation)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)

        if self.adapters:
            print('Using parallel adapters in Encoder')
            self.adapt = nn.ModuleDict({task: nn.Conv2d(planes, planes, kernel_size=1, bias=False) for task in tasks})

        if self.per_task_norm_layers:
            print('Using per-task batchnorm parameters in Encoder')
            self.bn1 = nn.ModuleDict({task: self.bnorm(planes, affine=affine_par) for task in tasks})
            self.bn2 = nn.ModuleDict({task: self.bnorm(planes, affine=affine_par) for task in tasks})
            self.bn3 = nn.ModuleDict({task: self.bnorm(planes * 4, affine=affine_par) for task in tasks})
        else:
            self.bn1 = self.bnorm(planes, affine=affine_par)
            self.bn2 = self.bnorm(planes, affine=affine_par)
            self.bn3 = self.bnorm(planes * 4, affine=affine_par)

        for i in self.bn1.parameters():
            i.requires_grad = train_norm_layers
        for i in self.bn2.parameters():
            i.requires_grad = train_norm_layers
        for i in self.bn3.parameters():
            i.requires_grad = train_norm_layers

        self.relu = nn.ReLU(inplace=True)

        if squeeze_enc:
            self.se = SELayerMultiTaskDict(channel=planes * 4, reduction=reduction, tasks=tasks)
        else:
            self.se = SELayerMultiTaskDict(channel=planes * 4, reduction=reduction, tasks=None)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x, task=None):
        residual = x

        out = self.conv1(x)
        if self.per_task_norm_layers:
            out = self.bn1[task](out)
        else:
            out = self.bn1(out)
        out = self.relu(out)

        if self.adapters:
            out = self.adapt[task](out) + self.conv2(out)
        else:
            out = self.conv2(out)

        if self.per_task_norm_layers:
            out = self.bn2[task](out)
        else:
            out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        if self.per_task_norm_layers:
            out = self.bn3[task](out)
        else:
            out = self.bn3(out)

        out = self.se(out, task)

        if self.downsample is not None:
            residual = self.downsample(x, task)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, n_classes, classifier='atrous-v3', output_stride=16, tasks=None,
                 train_norm_layers=False, width_decoder=256, squeeze_enc=True, squeeze_dec=False, adapters=False,
                 norm_per_task=False, dscr_type='fconv', dscr_d=2, dscr_k=1):

        super(ResNet, self).__init__()

        print("Constructing ResNet model...")
        print("Output stride: {}".format(output_stride))
        print("Number of classes: {}".format(n_classes))

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
        self.train_norm_layers = train_norm_layers
        self.bnorm = nn.BatchNorm2d
        self.squeeze_enc = squeeze_enc
        self.adapters = adapters

        self.tasks = tasks
        self.task_dict = {x: i for i, x in enumerate(self.tasks)}

        self.per_task_norm_layers = self.train_norm_layers and self.tasks

        self.use_dscr = True if dscr_type is not None else False

        # Network structure
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=strides[0], padding=3, bias=False)
        self.bn1 = self.bnorm(64, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = self.train_norm_layers
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=strides[1], padding=1, ceil_mode=False)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[2])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[3], dilation=dilations[0])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[4], dilation=dilations[1])

        width_low = 48 * width_decoder / 256  # Adapt in case of thinner classifiers
        assert (int(width_low) == width_low)

        print('Using decoder')
        if classifier == 'atrous-v3' or classifier == 'conv':
            print('Initializing classifier: A-trous with global features (Deeplab-v3+)')
            self.decoder = ASPPv3Plus(tasks=self.tasks,
                                      n_classes=n_classes,
                                      classifier=classifier,
                                      in_channels_low=256,
                                      in_channels_high=2048,
                                      out_f_classifier=width_decoder,
                                      atrous_rates=v3_atrous_rates,
                                      norm=self.bnorm,
                                      norm_per_task=norm_per_task,
                                      squeeze=squeeze_dec,
                                      adapters=self.adapters,
                                      )
        elif classifier == 'uber':
            print('Initializing Ubernet classifier')
            self.decoder = UbernetDecoder(tasks=self.tasks,
                                          n_classes=n_classes,
                                          in_channels_low=256,
                                          in_channels_high=2048,
                                          norm=self.bnorm)
        else:
            raise NotImplementedError('Choose one of the available classifiers')

        if self.use_dscr:
            print('Using Discriminator')
            self.dscr_d = dscr_d
            self.dscr_k = dscr_k
            self.task_label_shape = (128, 128)
            self.discriminator = self._get_discriminator(width_decoder)
            self.rev_layer = ReverseLayerF()
            self.criterion_classifier = torch.nn.CrossEntropyLoss(ignore_index=255)

        self._initialize_weights()

        # Check if batchnorm parameters are trainable
        self._verify_bnorm_params()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = ConvCoupledBatchNormMT(tasks=self.tasks if self.per_task_norm_layers else None,
                                                process_layer=nn.Conv2d(self.inplanes, planes * block.expansion,
                                                                        kernel_size=1, stride=stride, bias=False),
                                                norm=self.bnorm,
                                                norm_kwargs={'num_features': planes * block.expansion,
                                                             'affine': affine_par},
                                                train_norm=self.train_norm_layers)

        layers = [block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample,
                        train_norm_layers=self.train_norm_layers, tasks=self.tasks,
                        squeeze_enc=self.squeeze_enc, adapters=self.adapters)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation,
                                train_norm_layers=self.train_norm_layers, tasks=self.tasks,
                                squeeze_enc=self.squeeze_enc, adapters=self.adapters))

        return SequentialMultiTask(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, self.bnorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

        # zero-initialization for residual adapters
        for name, m in self.named_modules():
            if name.find('adapt') >= 0 and isinstance(m, nn.Conv2d):
                m.weight.data.fill_(0)

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

    def forward(self, x_in, task=None):

        x = x_in
        in_shape = x.shape[2:]

        # First Layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Stage #1 and low-level features
        x = self.layer1(x, task)
        x_low = x

        # Stages #2 - #4
        x = self.layer2(x, task=task)
        x = self.layer3(x, task=task)
        x = self.layer4(x, task=task)

        # Decoder
        x, features = self.decoder(x_low, x, task=task)

        out = F.interpolate(x, size=(in_shape[-2], in_shape[-1]), mode='bilinear', align_corners=False)

        return out, features

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

    def _get_discriminator(self, width_decoder):
        discriminator = FullyConvDiscriminator(in_channels=width_decoder, n_classes=len(self.tasks),
                                               kernel_size=self.dscr_k, depth=self.dscr_d)

        return discriminator

    def _create_task_labels(self, gt_elems, task):

        valid = deepcopy(gt_elems[task].detach())
        valid = F.interpolate(valid, size=self.task_label_shape, mode='nearest')
        valid[valid != 255] = self.task_dict[task]
        valid = valid[:, 0, :, :]

        return valid.long()

    def _define_if_copyable(self, module):
        is_copyable = isinstance(module, nn.Conv2d) \
                      or isinstance(module, nn.Linear) \
                      or isinstance(module, nn.BatchNorm2d) or \
                      isinstance(module, self.bnorm)
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
            if name_trg in copy_src:
                mapping[name_trg] = name_trg

            # Copy ImageNet SE layers to each task-specific layer
            elif self.tasks is not None:
                task = self._exists_task_in_name(name_trg)
                if task:
                    name_src = name_trg.replace('.' + task, '')

                    if name_src in copy_src:
                        mapping[name_trg] = name_src
                        task_specific_counter += 1

        # Handle downsampling layers
        for name_trg in copy_trg:
            name_src = None
            if name_trg.find('downsample') > 0:
                if name_trg.find('process') > 0:
                    name_src = name_trg.replace('process', '0')
                elif name_trg.find('norm'):
                    if self.per_task_norm_layers is not None:
                        task = self._exists_task_in_name(name_trg)
                        if task:
                            name_src = name_trg.replace('norm.' + task, '1')
                    else:
                        name_src = name_trg.replace('norm', '1')

            if name_src in copy_src:
                mapping[name_trg] = name_src

        i = 0
        for name in mapping:
            module_trg = copy_trg[name]
            module_src = copy_src[mapping[name]]

            if module_trg.weight.data.shape != module_src.weight.data.shape:
                print('Skipping layer with size: {} and target size: {}'
                      .format(module_trg.weight.data.shape, module_src.weight.data.shape))
                continue

            if isinstance(module_trg, nn.Conv2d) and isinstance(module_src, nn.Conv2d):
                module_trg.weight.data = deepcopy(module_src.weight.data)
                module_trg.bias = deepcopy(module_src.bias)
                i += 1

            elif isinstance(module_trg, self.bnorm) and (isinstance(module_src, nn.BatchNorm2d)
                                                         or isinstance(module_src, self.bnorm)):

                # Copy running mean and variance of batchnorm layers!
                module_trg.running_mean.data = deepcopy(module_src.running_mean.data)
                module_trg.running_var.data = deepcopy(module_src.running_var.data)

                module_trg.weight.data = deepcopy(module_src.weight.data)
                module_trg.bias.data = deepcopy(module_src.bias.data)
                i += 1

            elif isinstance(module_trg, nn.Linear) and (isinstance(module_src, nn.Linear)):
                module_trg.weight.data = deepcopy(module_src.weight.data)
                module_trg.bias.data = deepcopy(module_src.bias.data)
                i += 1

        print('\nContent of {} out of {} layers successfully copied, including {} task-specific layers\n'
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
        b = [model.conv1, model.bn1, model.layer1, model.layer2, model.layer3, model.layer4]
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


def se_resnet26(n_classes, pretrained='scratch', **kwargs):
    """Constructs a SE-ResNet-18 model.
    Args:
        pretrained (str): If True, returns a model pre-trained on ImageNet
    """

    print('Constructing ResNet18')
    model = ResNet(SEBottleneck, [2, 2, 2, 2], n_classes, **kwargs)

    if pretrained == 'imagenet':
        print('Loading pre-trained ImageNet model')
        model_full = se_resnet.se_resnet26(pretrained=True)
        model.load_pretrained(model_full)
    elif pretrained == 'scratch':
        print('Training from scratch!')
    else:
        raise NotImplementedError('Select between scratch and imagenet for pre-training')

    return model


def se_resnet50(n_classes, pretrained='scratch', **kwargs):
    """Constructs a SE-ResNet-50 model.
    Args:
        pretrained (str): If True, returns a model pre-trained on ImageNet
    """

    print('Constructing ResNet50')
    model = ResNet(SEBottleneck, [3, 4, 6, 3], n_classes, **kwargs)
    if pretrained == 'imagenet':
        print('Loading pre-trained ImageNet model')
        model_full = se_resnet.se_resnet50(pretrained=True)
        model.load_pretrained(model_full)
    elif pretrained == 'scratch':
        print('Training from scratch!')
    else:
        raise NotImplementedError('Select between scratch and imagenet for pre-training')

    return model


def se_resnet101(n_classes, pretrained='scratch', **kwargs):
    """Constructs a SE-ResNet-101 model.
    Args:
        pretrained (str): Select model trained on respective dataset.
    """

    print('Constructing ResNet101')

    model = ResNet(SEBottleneck, [3, 4, 23, 3], n_classes, **kwargs)
    if pretrained == 'imagenet':
        print('Loading pre-trained ImageNet model')
        model_full = se_resnet.se_resnet101(pretrained=True)
        model.load_pretrained(model_full)
    elif pretrained == 'scratch':
        print('Training from scratch!')
    else:
        raise NotImplementedError('Select between scratch and imagenet for pre-training')

    return model


def test_vis_net(net, elems):
    import cv2
    from torchvision import transforms
    from fblib.dataloaders import custom_transforms as tr
    from fblib.dataloaders.pascal_context import PASCALContext
    from torch.utils.data import DataLoader
    import fblib.util.visualizepy as viz

    tasks = elems[1:]
    net.cuda()

    # Define the transformations
    transform = transforms.Compose(
        [tr.FixedResize(resolutions={x: (512, 512) for x in elems},
                        flagvals={x: cv2.INTER_NEAREST for x in elems}),
         tr.ToTensor()])

    # Define dataset, tasks, and the dataloader
    dataset = PASCALContext(split=['train'],
                            transform=transform,
                            do_edge=True,
                            do_human_parts=True,
                            do_semseg=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

    net.eval()

    sample = next(iter(dataloader))

    img = sample['image']
    task_gts = list(sample.keys())
    img = img.cuda()
    y = {}
    for task in task_gts:
        if task in tasks:
            y[task], _ = net.forward(img, task=task)

    g = viz.make_dot(y, net.state_dict())
    g.view(directory='./')


def test_gflops(net, elems):
    from fblib.util.model_resources.flops import compute_gflops

    batch_size = 2
    gflops_1_task = compute_gflops(net, in_shape=(batch_size, 3, 256, 256), tasks=elems[1])

    print('GFLOPS for 1 task: {}'.format(gflops_1_task / batch_size))


def test_lr_params(net, tasks):
    params = get_lr_params(net, part='generic', tasks=tasks)
    for p in params:
        print(p)


def main():
    elems = ['image', 'edge', 'semseg']
    tasks = elems[1:]
    squeeze_enc = False
    squeeze_dec = False
    adapters = False
    width_decoder = 256
    norm_per_task = False

    # Load Network
    net = se_resnet26(n_classes={'edge': 1, 'semseg': 21},
                      pretrained='imagenet',
                      classifier='atrous-v3',
                      output_stride=8,
                      tasks=tasks,
                      width_decoder=width_decoder,
                      squeeze_enc=squeeze_enc,
                      squeeze_dec=squeeze_dec,
                      adapters=adapters,
                      norm_per_task=norm_per_task,
                      train_norm_layers=True,
                      dscr_type=None)

    test_vis_net(net, elems)
    test_gflops(net, elems)
    test_lr_params(net, tasks)


if __name__ == '__main__':
    main()
