import torch.nn as nn
import torch

def _make_divisible(ch, divisor=8, min_ch=None):                       # 调整通道数为8（divisor）的倍数，即可以被8整除
    if min_ch == None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch+divisor/2)//divisor*divisor)
    if new_ch < 0.9*ch:
        new_ch += divisor
    return new_ch

class ConBnRelu(nn.Sequential):                                       # 定义基本卷积层  卷积+正则化+激活函数
    def __init__(self, inchannel, outchannle, kernel_size=3, stride=1, group=1):
        padding = (kernel_size-1)//2
        super(ConBnRelu, self).__init__(
            nn.Conv2d(in_channels=inchannel, out_channels=outchannle,
                      kernel_size=kernel_size, stride=stride, padding=padding, groups=group, bias=False),
            nn.BatchNorm2d(outchannle),
            nn.ReLU6(inplace=True)
        )

class InvertResidual(nn.Module):                                     # 定义倒残差
    def __init__(self, inchannle, outchannel, stride, expand_ratio):
        super(InvertResidual, self).__init__()
        hidden_channle = inchannle * expand_ratio
        self.use_shotcut = stride == 1 and inchannle == outchannel
        layers = []
        if expand_ratio != 1:
            layers.append(ConBnRelu(inchannle, hidden_channle, kernel_size=1))
        layers.extend([
            ConBnRelu(hidden_channle, hidden_channle, stride=stride, group=hidden_channle),
            nn.Conv2d(hidden_channle, outchannel, kernel_size=1, bias=False),
            nn.BatchNorm2d(outchannel)]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shotcut:
            return x+self.conv(x)
        else:
            return self.conv(x)

class MobileNet_V2(nn.Module):
    def __init__(self, num_class=1000, alpht=1.0, round_nearest=8):
        super(MobileNet_V2,self).__init__()
        block = InvertResidual
        input_channel = _make_divisible(32*alpht, round_nearest)
        last_channel = _make_divisible(1280*alpht, round_nearest)
        inverted_residual_setting = [
            [1, 16, 1, 1],                                          #t c n s
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 1],
            [6, 320, 1, 1],
        ]
        features = []
        features.append(ConBnRelu(3, input_channel, stride=2))
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c*alpht, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        features.append(ConBnRelu(input_channel, last_channel, 1))
        self.feature = nn.Sequential(*features)
        self.average = nn.AdaptiveAvgPool2d((1, 1))
        self.classer = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_class)
        )
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.feature(x)
        x = self.average(x)
        x = torch.flatten(x, 1)
        x = self.classer(x)
        return x




