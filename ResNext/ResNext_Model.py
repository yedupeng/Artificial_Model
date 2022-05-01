import torch.nn as nn
import torch

class BasicBlock(nn.Module):                                                           # 18-layers、34-layers
    exception = 1
    def __init__(self, in_channels, out_channles, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channles,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(out_channles)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channles, out_channels=out_channles,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(out_channles)
        self.downsample = downsample

    def forward(self, x):
        identify = x
        if self.downsample is not None:
            identify = self.downsample(x)
        out = self.conv1(x)
        out = self.b1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.b2(out)
        out += identify
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):                                                              # 具体操作和ResNet网络一致，不过卷积换成了层卷积
    exception = 4
    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.exception,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.exception)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out
class ResNet(nn.Module):
    def __init__(self, block, block_num, num_class=1000, include_top=True, group=1, wid_per_group=64):
        super(ResNet, self).__init__()
        self.group = group
        self.wid_per_group = wid_per_group
        self.include_top = include_top
        self.in_channel = 64
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.b1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(block, 64, block_num[0])
        self.layer2 = self.make_layer(block, 128, block_num[1], stride=2)
        self.layer3 = self.make_layer(block, 256, block_num[2], stride=2)
        self.layer4 = self.make_layer(block, 512, block_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512*block.exception, num_class)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel*block.exception:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel*block.exception,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel*block.exception))
        layer = []
        layer.append(block(self.in_channel, channel, downsample=downsample, stride=stride,
                           groups=self.group, width_per_group=self.wid_per_group))
        self.in_channel = channel*block.exception
        for _ in range(1, block_num):
            layer.append(block(self.in_channel, channel,
                               groups=self.group, width_per_group=self.wid_per_group))
        return nn.Sequential(*layer)

    def forward(self, x):
        x = self.conv1(x)
        x = self.b1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
        return x

def Resnext50(num_class=1000, include_top=True):
    group = 32
    wid_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_class=num_class,
                  include_top=include_top,
                  group=group,
                  wid_per_group=wid_per_group
                  )





