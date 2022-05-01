import math
import copy
from functools import partial
from collections import OrderedDict
from typing import Optional, Callable
import torch
from torch import Tensor
from torch.nn import functional as f
import torch.nn as nn

def make_divisible(ch, divisor=8, min_ch=None):
    if min_ch is None:
        min_ch = 8
    new_ch = max(min_ch, int(ch+divisor//2)//divisor*divisor)
    if new_ch < 0.9*ch:
        new_ch += divisor
    return new_ch

class ConBnActivation(nn.Sequential):
    def __init__(self,
                 in_plane: int,
                 out_plane: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_layer: Optional[Callable[..., nn.Module]] = None):
        padding = (kernel_size-1)//2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.SiLU
        super(ConBnActivation, self).__init__(nn.Conv2d(in_channels=in_plane,
                                                        out_channels=out_plane,
                                                        stride=stride,
                                                        kernel_size=kernel_size,
                                                        groups=groups,
                                                        padding=padding,
                                                        bias=False),
                                              norm_layer(out_plane),
                                              activation_layer())

class Squeeze_Exception(nn.Module):
    def __init__(self,
                 input_c: int,
                 expand_c: int,
                 squeeze_factor: int = 4):
        super(Squeeze_Exception, self).__init__()
        squeeze_c = input_c//squeeze_factor
        self.f1 = nn.Conv2d(expand_c, squeeze_c, 1)
        self.a1 = nn.SiLU()
        self.f2 = nn.Conv2d(squeeze_c, expand_c, 1)
        self.a2 = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        scale = f.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.f1(scale)
        scale = self.a1(scale)
        scale = self.f2(scale)
        scale = self.a2(scale)
        return scale*x

class InvertResidualConfig:
    def __init__(self,
                 kernel: int,
                 input_c: int,
                 output_c: int,
                 expand_ratio: int,
                 stride: int,
                 use_se: bool,
                 drop_ratio: float,
                 index: str,
                 wid_confidence: float):
        self.input_c = self.adjustchannel(input_c, wid_confidence)
        self.kernel = kernel
        self.expand_c = self.input_c*expand_ratio
        self.output_c = self.adjustchannel(output_c, wid_confidence)
        self.stride = stride
        self.use_se = use_se
        self.drop_ratio = drop_ratio
        self.index = index

    @staticmethod
    def adjustchannel(channel: int, wid_confidence: float):
        return make_divisible(channel*wid_confidence, 8)

class InvertResidual(nn.Module):
    def __init__(self,
                 cnt: InvertResidualConfig,
                 norm_layer: Callable[..., nn.Module]):
        super(InvertResidual, self).__init__()
        if cnt.stride not in [1, 2]:
            raise ValueError("stride error!")
        self.use_res_connect = (cnt.stride == 1 and cnt.input_c == cnt.output_c)
        layers = OrderedDict()
        ac_layer = nn.SiLU

        if cnt.expand_c != cnt.input_c:
            layers.update({"expand_channel": ConBnActivation(cnt.input_c,
                                                             cnt.expand_c,
                                                             kernel_size=1,
                                                             norm_layer=norm_layer,
                                                             activation_layer=ac_layer
                                                             )})
        layers.update({"DwCon_channel": ConBnActivation(cnt.expand_c,
                                                        cnt.expand_c,
                                                        kernel_size=cnt.kernel,
                                                        stride=cnt.stride,
                                                        norm_layer=norm_layer,
                                                        activation_layer=ac_layer,
                                                        groups=cnt.expand_c)})
        if cnt.use_se:
            layers.update({"se": Squeeze_Exception(cnt.input_c,
                                                   cnt.expand_c)})
        layers.update({"Con_layer": ConBnActivation(cnt.expand_c,
                                                    cnt.output_c,
                                                    kernel_size=1,
                                                    norm_layer=norm_layer,
                                                    activation_layer=nn.Identity)})
        self.block = nn.Sequential(layers)
        self.out_c = cnt.output_c
        self.is_stride = cnt.stride > 1
        if cnt.drop_ratio > 0:
            self.drop = nn.Dropout2d(p=cnt.drop_ratio, inplace=True)
        else:
            self.drop = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        result = self.block(x)
        result = self.drop(result)
        if self.use_res_connect:
            result += x
        return result

class EfficientNet(nn.Module):
    def __init__(self,
                 width_confidence: float,
                 depth_confidence: float,
                 num_class: int = 1000,
                 dropout: float = 0.2,
                 drop_connect_ratio: float = 0.2,
                 block: Optional[Callable[..., nn.Module]] = None,
                 norm_layer:  Optional[Callable[..., nn.Module]] = None
                 ):
        super(EfficientNet, self).__init__()
        default_cnt = [[3, 32, 16, 1, 1, True, drop_connect_ratio, 1],
                       [3, 16, 24, 6, 2, True, drop_connect_ratio, 2],
                       [5, 24, 40, 6, 2, True, drop_connect_ratio, 2],
                       [3, 40, 80, 6, 2, True, drop_connect_ratio, 3],
                       [5, 80, 112, 6, 1, True, drop_connect_ratio, 3],
                       [5, 112, 192, 6, 2, True, drop_connect_ratio, 4],
                       [3, 192, 160, 6, 1, True, drop_connect_ratio, 1]]

        def round_repeat(repeats):
            return int(math.ceil(depth_confidence*repeats))

        if block is None:
            block = InvertResidual

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)
        adjust_c = partial(InvertResidualConfig.adjustchannel,
                           wid_confidence=width_confidence)
        bneck_conf = partial(InvertResidualConfig,
                             wid_confidence=width_confidence)
        b = 0
        num_block = float(sum(round_repeat(i[-1]) for i in default_cnt))
        InvertResidual_setting = []
        for stage, args in enumerate(default_cnt):
            cnt = copy.copy(args)
            for i in range(round_repeat(cnt.pop(-1))):
                if i > 0:
                    cnt[-3] = 1
                    cnt[1] = cnt[2]
                cnt[-1] = args[-2] * b / num_block
                index = str(stage+1) + chr(i+97)
                InvertResidual_setting.append(bneck_conf(*cnt, index))
                b += 1
            layers = OrderedDict()
            layers.update({"Conv": ConBnActivation(in_plane=3,
                                                   out_plane=adjust_c(32),
                                                   kernel_size=3,
                                                   stride=2,
                                                   norm_layer=norm_layer)})
        for cnt in InvertResidual_setting:
            layers.update({cnt.index: block(cnt, norm_layer)})
        last_con_input = InvertResidual_setting[-1].output_c
        last_con_output = adjust_c(1280)
        layers.update({"top": ConBnActivation(
            in_plane=last_con_input,
            out_plane=last_con_output,
            kernel_size=1,
            norm_layer=norm_layer
        )})
        self.feature = nn.Sequential(layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        classifier = []
        if dropout > 0:
            classifier.append(nn.Dropout2d(p=dropout, inplace=True))
        classifier.append(nn.Linear(last_con_output, num_class))
        self.classifier = nn.Sequential(*classifier)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


    def _forward(self, x: Tensor)->Tensor:
        x = self.feature(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


    def forward(self, x:Tensor)->Tensor:
        return self._forward(x)

def Efficiention_b0(num_class=1000):
    return EfficientNet(width_confidence=0.1,
                        depth_confidence=0.1,
                        dropout=0.2,
                        num_class=num_class)

def Efficiention_b1(num_class=1000):
    return EfficientNet(width_confidence=1.0,
                        depth_confidence=1.1,
                        dropout=0.2,
                        num_class=num_class)



