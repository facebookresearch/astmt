import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import grad
import math

from fblib.util.custom_container import SequentialMultiTask
from fblib.layers.squeeze import SELayerMultiTask
from fblib.layers.attention import Conv2dAttentionAdapters, XPathLayer
from fblib.networks.discriminators import Discriminator, MiniDiscriminator
from fblib.layers.reverse_grad import ReverseLayerF


class ResNeXtBottleneck(nn.Module):
    expansion = 4
    """
    RexNeXt bottleneck type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
    """

    def __init__(self, inplanes, planes, cardinality, base_width, stride=1, downsample=None, n_tasks=1,
                 adapters=False, attention=False, squeeze=False, bn_per_task=False, binary_attention=False,
                 xpath=False):

        super(ResNeXtBottleneck, self).__init__()

        D = int(math.floor(planes * (base_width / 64.0)))
        C = cardinality

        self.adapters = adapters
        self.attention = attention
        self.squeeze = squeeze
        self.binary_attention = binary_attention
        self.xpath = xpath

        self.conv_reduce = nn.Conv2d(inplanes, D * C, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_reduce = nn.BatchNorm2d(D * C)

        if self.xpath:
            self.xp = XPathLayer(in_channels=inplanes,
                                 interm_channels=D,
                                 out_channels=planes * 4,
                                 stride=stride,
                                 n_tasks=n_tasks)

        self.conv_conv_bn = Conv2dAttentionAdapters(in_channels=D * C,
                                                    out_channels=D * C,
                                                    kernel_size=3,
                                                    stride=stride,
                                                    n_tasks=n_tasks,
                                                    padding=1,
                                                    groups=C,
                                                    bias=False,
                                                    adapters=self.adapters,
                                                    attention=self.attention,
                                                    bn_per_task=bn_per_task,
                                                    binary_attention=binary_attention)

        self.conv_expand = nn.Conv2d(D * C, planes * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_expand = nn.BatchNorm2d(planes * 4)

        if self.squeeze:
            self.se = SELayerMultiTask(planes * 4, n_tasks=n_tasks)

        self.downsample = downsample

    def forward(self, x, task=None):
        residual = x
        # print("Task: {}".format(task))

        if self.xpath:
            xpath_feat = self.xp(x, task)

        bottleneck = self.conv_reduce(x)
        bottleneck = F.relu(self.bn_reduce(bottleneck), inplace=True)

        bottleneck = self.conv_conv_bn(bottleneck, task)
        bottleneck = F.relu(bottleneck, inplace=True)

        bottleneck = self.conv_expand(bottleneck)
        bottleneck = self.bn_expand(bottleneck)

        if self.xpath:
            bottleneck += xpath_feat

        if self.squeeze:
            bottleneck = self.se(bottleneck, task)

        if self.downsample is not None:
            residual = self.downsample(x)

        return F.relu(residual + bottleneck, inplace=True)


class CifarResNeXtAbstract(nn.Module):
    def __init__(self, block, depth, cardinality, base_width, num_classes, n_tasks=1, use_orig=False,
                 bn_per_task=False, adapters=False, attention=False, squeeze=False, binary_attention=False,
                 xpath=False, use_discriminator=False, reverse_grad=False, ret_features=False, n_gpu=1,
                 onlynorm=False):

        super(CifarResNeXtAbstract, self).__init__()

        # Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
        assert (depth - 2) % 9 == 0, 'depth should be one of 20, 29, 38, 47, 56, 101'

        # Generic params
        self.depth = depth
        self.block = block
        self.layer_blocks = (depth - 2) // 9
        self.cardinality = cardinality
        self.base_width = base_width
        self.inplanes = 64

        # Multi-tasking params
        self.num_classes = num_classes
        self.use_orig = use_orig
        self.n_aux_tasks = n_tasks
        self.n_tasks = self.n_aux_tasks + 1 if self.use_orig else self.n_aux_tasks

        # Forward pass modifiers
        self.adapters = adapters  # per-task parallel adapters
        self.attention = attention  # per-task Attention modules
        self.squeeze = squeeze  # per-task SE modules
        self.xpath = xpath  # per-task ResNeXt path
        self.binary_attention = binary_attention
        self.bn_per_task = bn_per_task

        # Task-discriminator
        self.use_discriminator = use_discriminator
        self.ret_features = ret_features
        self.reverse_grad = reverse_grad
        self.n_gpu = n_gpu
        self.onlynorm = onlynorm

    def _initialize_layers(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, n_tasks=1,
                    adapters=False, attention=False, squeeze=False,
                    binary_attention=False, xpath=False):

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, self.cardinality, self.base_width, stride, downsample=downsample,
                        bn_per_task=self.bn_per_task, binary_attention=binary_attention, n_tasks=n_tasks,
                        adapters=adapters, attention=attention, squeeze=squeeze, xpath=xpath)]

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.cardinality, self.base_width,
                                bn_per_task=self.bn_per_task, binary_attention=binary_attention, n_tasks=n_tasks,
                                adapters=adapters, attention=attention, squeeze=squeeze, xpath=xpath))

        return SequentialMultiTask(*layers)

    def forward(self, *args):
        raise NotImplementedError


class Body(CifarResNeXtAbstract):
    def __init__(self, **kwargs):
        super(Body, self).__init__(**kwargs)

        # Network definition
        self.conv_1_3x3 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)

        self.inplanes = 64
        self.stage_1 = self._make_layer(self.block, 64, self.layer_blocks, stride=1, n_tasks=self.n_tasks)
        self.stage_2 = self._make_layer(self.block, 128, self.layer_blocks, stride=2, n_tasks=self.n_tasks)

    def forward(self, x):
        x = self.conv_1_3x3(x)
        x = F.relu(self.bn_1(x), inplace=True)
        x = self.stage_1(x)
        x = self.stage_2(x)
        return x


class Features(CifarResNeXtAbstract):
    def __init__(self, **kwargs):
        super(Features, self).__init__(**kwargs)

        self.inplanes = 512  # Computed!
        self.stage_3 = self._make_layer(self.block, 256, self.layer_blocks, stride=2, n_tasks=self.n_tasks,
                                        adapters=self.adapters, attention=self.attention, squeeze=self.squeeze,
                                        binary_attention=self.binary_attention, xpath=self.xpath)
        self.avgpool = nn.AvgPool2d(8)

    def forward(self, x, task):
        feat = self.stage_3(x, task)
        feat = self.avgpool(feat)

        feat = feat.view(feat.size(0), -1)

        return feat


class CifarResNeXt(CifarResNeXtAbstract):
    """
  ResNext optimized for the Cifar dataset, as specified in
  https://arxiv.org/pdf/1611.05431.pdf
  """

    def __init__(self, **kwargs):

        super(CifarResNeXt, self).__init__(**kwargs)

        self.body = Body(depth=self.depth,
                         block=self.block,
                         cardinality=self.cardinality,
                         base_width=self.base_width,
                         num_classes=self.num_classes,
                         n_tasks=self.n_tasks,
                         bn_per_task=self.bn_per_task,
                         adapters=self.adapters,
                         attention=self.attention,
                         squeeze=self.squeeze,
                         binary_attention=self.binary_attention,
                         xpath=self.xpath)

        self.features = Features(depth=self.depth,
                                 block=self.block,
                                 cardinality=self.cardinality,
                                 base_width=self.base_width,
                                 num_classes=self.num_classes,
                                 n_tasks=self.n_tasks,
                                 bn_per_task=self.bn_per_task,
                                 adapters=self.adapters,
                                 attention=self.attention,
                                 squeeze=self.squeeze,
                                 binary_attention=self.binary_attention,
                                 xpath=self.xpath)

        print('Using discriminator')
        if not self.onlynorm:
            self.discriminator = Discriminator(in_channels=256 * 4, n_classes=self.n_tasks)
        else:
            self.discriminator = MiniDiscriminator(in_channels=1, n_classes=self.n_tasks)

        if self.reverse_grad:
            self.rev_layer = ReverseLayerF()
        self.classifier = nn.ModuleList()

        if self.use_orig:
            self.classifier.append(nn.Linear(256 * self.block.expansion, 100))
        for i in range(self.n_aux_tasks):
            self.classifier.append(nn.Linear(256 * self.block.expansion, self.num_classes))

        if self.n_gpu > 0:
            self.body = torch.nn.DataParallel(self.body, device_ids=list(range(self.n_gpu)))
            self.features = torch.nn.DataParallel(self.features, device_ids=list(range(self.n_gpu)))

    def forward(self, x, tasks=None, train=True):

        if not train:
            with torch.no_grad():
                x = self.body(x)
        else:
            x = self.body(x)

        outputs = []
        feats = []

        for task in range(self.n_tasks):
            if task in tasks:
                feat = self.features(x, task)
                feats.append(feat)
                outputs.append(self.classifier[task](feat))

        return outputs, feats

    def compute_losses_discr(self, output, features, criterion, gt_cls, gt_task, alpha, args):
        """
        Computes losses for tasks, losses for discriminator, output of discriminator, and gradients
        """
        # Compute classification losses and gradients wrt features
        grads = []
        losses_tasks = []
        for task in range(args.n_tasks):
            curr_loss = args.main_w[task] * criterion(output[task], gt_cls[task])
            losses_tasks.append(curr_loss)

            grads.append(grad(curr_loss, features[task], create_graph=True)[0])

        # Compute gradient norm and discriminator input depending on the objective
        grads_norm = [x.norm(p=2, dim=1) + 1e-10 for x in grads]
        if args.onlynorm:
            grads_norm = [x.unsqueeze(1) for x in grads_norm]
            input_discr = grads_norm
        else:
            grads_norm = [torch.cat(grads_norm).mean().item() for i in range(len(grads_norm))]
            input_discr = [grads[task] / grads_norm[task] for task in range(args.n_tasks)]

        # Compute discriminator loss
        outputs_discr = []
        losses_discr = []
        for task in range(args.n_tasks):
            if args.reverse:
                input_discr[task] = self.rev_layer.apply(input_discr[task], alpha)

            output_discr = self.discriminator(input_discr[task])
            curr_loss_discr = criterion(output_discr, gt_task[task])
            losses_discr.append(curr_loss_discr)
            outputs_discr.append(output_discr)

        # print(input_discr)
        # print(outputs_discr)
        return losses_tasks, losses_discr, outputs_discr, grads


def resnext20(cardinality=8, base_width=64, **kwargs):
    """Constructs a ResNeXt-18, 16*64d model for CIFAR-10 (by default)
    """
    model = CifarResNeXt(block=ResNeXtBottleneck, depth=20, cardinality=cardinality, base_width=base_width, **kwargs)
    return model


def resnext29(cardinality=8, base_width=64, **kwargs):
    """Constructs a ResNeXt-29, 8*64d model for CIFAR-10 (by default)
    """
    model = CifarResNeXt(block=ResNeXtBottleneck, depth=29, cardinality=cardinality, base_width=base_width, **kwargs)
    return model


def visualize_network(net, n_tasks=1):
    import fblib.util.visualize as viz
    net.eval()
    x = torch.randn(1, 3, 32, 32)
    x.requires_grad_()

    # pdf visualizer
    y, feats = net.forward(x, list(range(n_tasks + 1)))
    g = viz.make_dot(y, net.state_dict())
    g.view(directory='/private/home/kmaninis')


if __name__ == '__main__':
    import torchvision.transforms as transforms

    n_tasks = 1

    # Transformations
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.RandomCrop(32, padding=4),
                                          transforms.ToTensor()])

    # Load network
    net = resnext20(8, 64, num_classes=10, n_tasks=n_tasks, use_orig=True, bn_per_task=False,
                    adapters=False, attention=False, squeeze=True, binary_attention=False, xpath=False, n_gpu=0)

    visualize_network(net, n_tasks=n_tasks)
