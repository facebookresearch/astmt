import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import fblib.networks.mobilenet_v2 as mobilenet_v2_imagenet
from fblib.util.mypath import Path
from collections import OrderedDict


def conv3x3_mnet(planes, stride=1, dilation=1):
    """3x3 depth-wiseconvolution with padding"""
    return nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                     dilation=dilation, padding=dilation, bias=False,
                     groups=planes)


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class ASPPMnet(nn.Module):
    """
    Atrous Spatial Pyramid Pooling Module (DeepLab-v3+) for mobilenet
    """

    def __init__(self, dilation_series=None, out_f=256, in_f=320):
        super(ASPPMnet, self).__init__()

        if dilation_series is None:
            dilation_series = [6, 12, 18]
        padding_series = dilation_series

        self.bnorm = nn.BatchNorm2d

        kwargs = {"num_features": out_f, "affine": True}
        # Reduce features in order to apply depth-wise convolutions
        self.conv_reduce = nn.Sequential(nn.Conv2d(in_f, out_f, kernel_size=1, stride=1, bias=False),
                                         self.bnorm(**kwargs),
                                         nn.ReLU6(inplace=True))

        # List of parallel convolutions
        self.conv2d_list = nn.ModuleList()

        # 1x1 convolution
        self.conv2d_list.append(nn.Sequential(nn.Conv2d(out_f, out_f, kernel_size=1, stride=1,
                                                        bias=False, groups=out_f),
                                              self.bnorm(**kwargs),
                                              nn.ReLU6(inplace=True)))
        # Dilated Convolutions
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Sequential(nn.Conv2d(out_f, out_f, kernel_size=3, stride=1, padding=padding,
                                                            dilation=dilation, bias=False, groups=out_f),
                                                  self.bnorm(**kwargs),
                                                  nn.ReLU6(inplace=True)))

        # Global features
        self.conv2d_list.append(nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                                              nn.Conv2d(out_f, out_f, kernel_size=1, stride=1,
                                                        bias=False, groups=out_f),
                                              self.bnorm(**kwargs),
                                              nn.ReLU6(inplace=True)))

        self.conv2d_final = nn.Sequential(nn.Conv2d(out_f * 5, out_f, kernel_size=1,
                                                    stride=1, bias=False, groups=out_f),
                                          self.bnorm(**kwargs),
                                          nn.ReLU6(inplace=True))

    def forward(self, x):
        h, w = x.size(2), x.size(3)

        # Reduce
        x = self.conv_reduce(x)

        # ASPP
        interm = []
        for i in range(len(self.conv2d_list)):
            interm.append(self.conv2d_list[i](x))

        # Upsample the global features
        interm[-1] = F.interpolate(input=interm[-1], size=(h, w), mode='bilinear', align_corners=False)

        # Concatenate the parallel streams
        out = torch.cat(interm, dim=1)

        # Final convolutional layer of the classifier
        out = self.conv2d_final(out)

        return out


class ASPPDecoderMnet(nn.Module):
    """
    ASPP-v3 decoder for Mobilenet
    """

    def __init__(self,
                 n_classes,
                 in_channels_high=320,
                 in_channels_low=24,
                 out_f_classifier=256,
                 atrous_rates=None,
                 up=4,
                 ):
        super(ASPPDecoderMnet, self).__init__()
        print('Initializing Mobilenet ASPP v3 Decoder')

        if atrous_rates is None:
            atrous_rates = [6, 12, 18]

        kwargs_out = {"num_features": out_f_classifier, "affine": True}
        kwargs_low = {"num_features": 48, "affine": True}

        self.up = up
        self.norm = nn.BatchNorm2d

        print('Initializing classifier: ASPP with global features (Deeplab-v3+)')
        self.layer5 = ASPPMnet(in_f=in_channels_high,
                               out_f=out_f_classifier,
                               dilation_series=atrous_rates)

        self.low_level_reduce = nn.Sequential(nn.Conv2d(in_channels_low, 48, kernel_size=1,
                                                        stride=1, bias=False, groups=2),
                                              self.norm(**kwargs_low),
                                              nn.ReLU6(inplace=True))

        self.conv_concat = nn.Sequential(nn.Conv2d(out_f_classifier + 48, out_f_classifier, kernel_size=3, padding=1,
                                                   stride=1, bias=False, groups=math.gcd(304, 256)),
                                         self.norm(**kwargs_out),
                                         nn.ReLU6(inplace=True))

        self.conv_process = nn.Sequential(conv3x3_mnet(out_f_classifier),
                                          self.norm(**kwargs_out),
                                          nn.ReLU6(inplace=True))

        self.conv_predict = nn.Conv2d(out_f_classifier, n_classes, kernel_size=1, bias=True)

    def forward(self, x_low, x):
        x_low = self.low_level_reduce(x_low)

        x = self.layer5(x)

        x = F.interpolate(x, scale_factor=self.up, mode='bilinear', align_corners=False)

        x = torch.cat([x_low, x], dim=1)

        x = self.conv_concat(x)
        x = self.conv_process(x)

        features = x

        x = self.conv_predict(x)

        return x, features


class InvResidualCommon(nn.Module):
    """Common Inverted Residual block for Mobilenet
    """

    def __init__(self, hidden_dim, oup, stride, dilation=1, train_norm_layers=False):
        super(InvResidualCommon, self).__init__()

        self.conv1 = nn.Conv2d(hidden_dim, hidden_dim, 3, stride, padding=dilation,
                               groups=hidden_dim, bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        for x in self.bn1.parameters():
            x.requires_grad = train_norm_layers

        self.conv2 = nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(oup)
        for x in self.bn2.parameters():
            x.requires_grad = train_norm_layers

        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        return x


class InvResidualExpand(nn.Module):
    """Expanding inverted residual block for Mobilenet
    """

    def __init__(self, inp, hidden_dim, oup, stride, dilation=1, train_norm_layers=False):
        super(InvResidualExpand, self).__init__()

        self.conv1 = nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        for x in self.bn1.parameters():
            x.requires_grad = train_norm_layers

        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, stride, padding=dilation,
                               groups=hidden_dim, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        for x in self.bn2.parameters():
            x.requires_grad = train_norm_layers

        self.conv3 = nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(oup)
        for x in self.bn3.parameters():
            x.requires_grad = train_norm_layers

        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, dilation=1, train_norm_layers=False):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = InvResidualCommon(hidden_dim=hidden_dim,
                                          oup=oup,
                                          stride=stride,
                                          dilation=dilation,
                                          train_norm_layers=train_norm_layers)
        else:
            self.conv = InvResidualExpand(inp=inp,
                                          hidden_dim=hidden_dim,
                                          oup=oup,
                                          stride=stride,
                                          dilation=dilation,
                                          train_norm_layers=train_norm_layers)

    def forward(self, x):
        if self.use_res_connect:
            out = self.conv(x)
            return x + out
        else:
            out = self.conv(x)
            return out


class MobileNetV2(nn.Module):
    def __init__(self, n_classes, width_mult=1., output_stride=16, train_norm_layers=False,
                 nInputChannels=3, classifier='atrous-v3', sync_bnorm=False):

        super(MobileNetV2, self).__init__()

        self.train_norm_layers = train_norm_layers
        if sync_bnorm:
            raise NotImplementedError('Sync bnorm not implemented for mobilenet')

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
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1, 1],
            [6, 24, 2, strides[1], 1],
            [6, 32, 3, strides[2], 1],
            [6, 64, 4, strides[3], dilations[0]],
            [6, 96, 3, 1, dilations[0]],
            [6, 160, 3, strides[4], dilations[1]],
            [6, 320, 1, 1, dilations[1]],
        ]

        input_channel = int(input_channel * width_mult)

        # build first layer of low level features
        self.features = [conv_bn(nInputChannels, input_channel, strides[0])]

        # build inverted residual blocks
        for t, c, n, s, dil in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s,
                                               expand_ratio=t,
                                               dilation=dil,
                                               train_norm_layers=train_norm_layers))
                else:
                    self.features.append(block(input_channel, output_channel, 1,
                                               expand_ratio=t,
                                               dilation=dil,
                                               train_norm_layers=train_norm_layers))
                input_channel = output_channel

        self.features = nn.Sequential(*self.features)

        self.features_low = self.features[:4]
        self.features_high = self.features[4:]

        if classifier == 'atrous-v3':
            self.decoder = ASPPDecoderMnet(n_classes=n_classes,
                                           in_channels_high=320,
                                           out_f_classifier=256,
                                           atrous_rates=atrous_rates)
        else:
            raise NotImplementedError('Implemented classifier: atrous-v3')

        self._initialize_weights()
        self._verify_bnorm_params()

    def forward(self, x):

        in_shape = x.shape[2:]

        x_low = self.features_low(x)
        x = self.features_high(x_low)

        x, _ = self.decoder(x_low, x)

        x = F.interpolate(x, size=(in_shape[-2], in_shape[-1]), mode='bilinear', align_corners=False)

        return x

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

    @staticmethod
    def _define_if_copyable(module):
        is_copyable = isinstance(module, nn.Conv2d) \
                      or isinstance(module, nn.Linear) \
                      or isinstance(module, nn.BatchNorm2d) or \
                      isinstance(module, nn.BatchNorm2d)
        return is_copyable

    def load_pretrained(self, base_network):

        copy_trg = {}
        for (name_trg, module_trg) in self.named_modules():
            if self._define_if_copyable(module_trg):
                copy_trg[name_trg] = module_trg

        copy_src = {}
        for (name_src, module_src) in base_network.named_modules():
            if self._define_if_copyable(module_src):
                copy_src[name_src] = module_src

        mapping = {}
        for name_trg in copy_trg:
            if 'decoder' in name_trg:
                continue
            elif 'features.' in name_trg:
                mapping[name_trg] = name_trg

        for name_trg in mapping:
            map_trg = mapping[name_trg]
            if '.conv1' in name_trg:
                map_trg = map_trg.replace('.conv1', '.0')
            elif '.bn1' in name_trg:
                map_trg = map_trg.replace('.bn1', '.1')
            elif '.conv2' in name_trg:
                map_trg = map_trg.replace('.conv2', '.3')
            elif '.bn2' in name_trg:
                map_trg = map_trg.replace('.bn2', '.4')
            elif '.conv3' in name_trg:
                map_trg = map_trg.replace('.conv3', '.6')
            elif '.bn3' in name_trg:
                map_trg = map_trg.replace('.bn3', '.7')

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

        print('\nContents of {} out of {} layers successfully copied\n'
              .format(i, len(mapping)))


def get_lr_params(model, part='all'):
    """
    This generator returns all the parameters of the network
    """
    # for multi-GPU training
    if hasattr(model, 'module'):
        model = model.module

    if part == 'all':
        b = [model]
    elif part == 'backbone':
        b = [model.features_low, model.features]
    elif part == 'decoder':
        b = [model.decoder]

    for i in range(len(b)):
        for name, k in b[i].named_parameters():
            if k.requires_grad:
                yield k


def mobilenet_v2(pretrained='scratch', **kwargs):
    model = MobileNetV2(**kwargs)

    if pretrained == 'imagenet':

        print('loading pre-trained imagenet model')
        model_full = mobilenet_v2_imagenet.mobilenet_v2(pretrained=True)
        model.load_pretrained(model_full)
    elif pretrained == 'coco':
        print('loading pre-trained COCO model')
        # Load checkpoint
        checkpoint = torch.load(
            os.path.join(Path.models_dir(), 'mobilenet_v2_coco_80.pth'), map_location=lambda storage, loc: storage)

        # handle dataparallel
        if 'module.' in list(checkpoint.keys())[0]:
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                name = k.replace('module.', '')  # remove `module.`
                new_state_dict[name] = v
        else:
            new_state_dict = checkpoint

        # Load pre-trained IN model
        model.load_state_dict(new_state_dict)

    elif pretrained == 'scratch':
        print('using imagenet initialized from scratch')
    else:
        raise NotImplementedError('select either imagenet or scratch for pre-training')

    return model


def test_flops():
    from fblib.util.model_resources.flops import compute_gflops
    net = mobilenet_v2(n_classes=21, pretrained='imagenet',
                       output_stride=16, train_norm_layers=True)
    print('GFLOPS: {}'.format(compute_gflops(net, (2, 3, 512, 512))))


def test_visualize_network():
    import fblib.util.visualize as viz
    net = mobilenet_v2(n_classes=21, pretrained='imagenet',
                       output_stride=16, train_norm_layers=True)
    net.eval()
    x = torch.randn(1, 3, 512, 512)
    x.requires_grad_()

    # pdf visualizer
    y = net.forward(x)
    g = viz.make_dot(y, net.state_dict())
    g.view(directory='.')


if __name__ == '__main__':
    test_flops()
    # test_visualize_network()
