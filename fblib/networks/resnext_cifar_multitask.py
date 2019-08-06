import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torch.nn import init
import math

from fblib.util.custom_container import SequentialMultiTask
from fblib.layers.squeeze import SELayerMultiTask
from fblib.layers.attention import Conv2dAttentionAdapters, XPathLayer


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

        self.conv_conv_bn = Conv2dAttentionAdapters(in_channels=D*C,
                                                    out_channels=D*C,
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
                 onlynorm=False, task_stages=1):

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
        self.task_stages = task_stages
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

        self.gen_stages = 3 - self.task_stages  # Total stages for CIFAR ResNeXt is 3

        self.stages = []

        if self.gen_stages >= 1:
            self.inplanes = 64
            self.stages.append(self._make_layer(self.block, 64, self.layer_blocks, stride=1, n_tasks=self.n_tasks))
        if self.gen_stages >= 2:
            self.inplanes = 256
            self.stages.append(self._make_layer(self.block, 128, self.layer_blocks, stride=2, n_tasks=self.n_tasks))
        if self.gen_stages == 3:
            self.inplanes = 512
            self.stages.append(self._make_layer(self.block, 256, self.layer_blocks, stride=2, n_tasks=self.n_tasks))

        if len(self.stages) > 0:
            self.stages = nn.ModuleList(self.stages)

    def forward(self, x):
        x = self.conv_1_3x3(x)
        x = F.relu(self.bn_1(x), inplace=True)

        for i in range(len(self.stages)):
            x = self.stages[i](x)

        return x


class Features(CifarResNeXtAbstract):
    def __init__(self, **kwargs):
        super(Features, self).__init__(**kwargs)

        self.stages = []

        if self.task_stages == 3:
            self.inplanes = 64
            self.stages.append(self._make_layer(self.block, 64, self.layer_blocks, stride=1, n_tasks=self.n_tasks,
                                                adapters=self.adapters, attention=self.attention, squeeze=self.squeeze,
                                                binary_attention=self.binary_attention, xpath=self.xpath))
        if self.task_stages >= 2:
            self.inplanes = 256
            self.stages.append(self._make_layer(self.block, 128, self.layer_blocks, stride=2, n_tasks=self.n_tasks,
                                                adapters=self.adapters, attention=self.attention, squeeze=self.squeeze,
                                                binary_attention=self.binary_attention, xpath=self.xpath))
        if self.task_stages >= 1:
            self.inplanes = 512
            self.stages.append(self._make_layer(self.block, 256, self.layer_blocks, stride=2, n_tasks=self.n_tasks,
                                                adapters=self.adapters, attention=self.attention, squeeze=self.squeeze,
                                                binary_attention=self.binary_attention, xpath=self.xpath))
        self.avgpool = nn.AvgPool2d(8)

        if len(self.stages) > 0:
            self.stages = nn.ModuleList(self.stages)

    def forward(self, x, task):
        for i in range(len(self.stages)):
            x = self.stages[i](x, task)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)

        return x


class CifarResNeXt(CifarResNeXtAbstract):
    """
  ResNext optimized for the Cifar dataset, as specified in
  https://arxiv.org/pdf/1611.05431.pdf
  """

    def __init__(self, **kwargs):

        super(CifarResNeXt, self).__init__(**kwargs)

        self.body = Body(task_stages=self.task_stages,
                         depth=self.depth,
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

        self.features = Features(task_stages=self.task_stages,
                                 depth=self.depth,
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

        self.classifier = nn.ModuleList()

        if self.use_orig:
            self.classifier.append(nn.Linear(256 * self.block.expansion, 100))
        for i in range(self.n_aux_tasks):
            self.classifier.append(nn.Linear(256 * self.block.expansion, self.num_classes))

        if self.n_gpu > 0:
            self.body = torch.nn.DataParallel(self.body, device_ids=list(range(self.n_gpu)))
            self.features = torch.nn.DataParallel(self.features, device_ids=list(range(self.n_gpu)))
            self.classifier = nn.ModuleList([torch.nn.DataParallel(x, device_ids=list(range(self.n_gpu))) for x in self.classifier])

    def forward(self, x, tasks=None):
        x = self.body(x)

        outputs = []
        feats = []
        for task in range(self.n_tasks):
            if task in tasks:
                feat = self.features(x, task)
                if self.ret_features:
                    feats.append(feat)
                outputs.append(self.classifier[task](feat))

        return outputs, feats

    def compute_losses(self, output, features, criterion, gt_cls, gt_task, args):
        """
        Computes losses for tasks
        """
        # Compute classification losses and gradients wrt features
        grads = []
        losses_tasks = []
        for task in range(args.n_tasks):
            curr_loss = criterion(output[task], gt_cls[task])
            losses_tasks.append(curr_loss)

            grads.append(grad(curr_loss, features[task], create_graph=True)[0])

        if args.adj_w:
            # Adjust weights of tasks according to the norm of their gradients
            grads_norms = [x.detach().norm(p=2, dim=1) + 1e-10 for x in grads]
            grads_norms = torch.cat(grads_norms)
            gt_task = torch.cat(gt_task)

            mean_norms = [0] * args.n_tasks
            for task in range(args.n_tasks):
                mean_norms[task] = grads_norms[gt_task == task].mean().item()

            for task in range(args.n_tasks):
                args.main_w[task] = (sum(mean_norms) - mean_norms[task]) / sum(mean_norms) * args.n_tasks

        return losses_tasks


def resnext20(cardinality=8, base_width=64, **kwargs):
    """Constructs a ResNeXt-20, 8*64d model for CIFAR-10 (by default)
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
    net.cuda().eval()
    x = torch.randn(1, 3, 32, 32)
    x.requires_grad_().cuda()

    # pdf visualizer
    y, _ = net.forward(x, list(range(n_tasks+1)))
    g = viz.make_dot(y, net.state_dict())
    g.view(directory='/private/home/kmaninis')


def test_resources(net):
    from fblib.util.model_resources.flops import compute_gflops
    from fblib.util.model_resources.num_parameters import count_parameters

    net.cuda()
    gflops = compute_gflops(net, in_shape=(1, 3, 32, 32), tasks=list(range(n_tasks+1)))
    print('GFLOPS: {}'.format(gflops))
    print("\nNumber of parameters (in millions): {0:.3f}\n".format(count_parameters(net) / 1e6))


if __name__ == '__main__':
    import torchvision.transforms as transforms

    n_tasks = 1

    # Transformations
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.RandomCrop(32, padding=4),
                                          transforms.ToTensor()])

    # Load network
    net = resnext20(8, 64, num_classes=10, n_tasks=n_tasks, use_orig=True, bn_per_task=False,
                    adapters=False, attention=False, squeeze=True, binary_attention=False, xpath=False, task_stages=1)
    test_resources(net)
    visualize_network(net, n_tasks=n_tasks)
