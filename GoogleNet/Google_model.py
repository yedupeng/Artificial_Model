import torch
import torch.nn as nn
import torch.nn.functional as F

# 网络
class GoogleNet(nn.Module):
    def __init__(self,num_classes=1000, aux_enable=True, init_weight=False):      # 根据网络按顺序编写结构
        super(GoogleNet, self).__init__()
        self.aux_enable = aux_enable
        self.Conv1 = Basic_Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.MaxPool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.Conv2 = Basic_Conv2d(64, 64, kernel_size=1)
        self.Conv3 = Basic_Conv2d(64, 192, kernel_size=3, padding=1)
        self.MaxPool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.Inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.Inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.MaxPool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.Inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.Inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.Inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.Inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.Inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.MaxPool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.Inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.Inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        if aux_enable:
            self.aux1a = InceptionAux(512, num_classes)
            self.aux1b = InceptionAux(528, num_classes)

        self.average = nn.AdaptiveAvgPool2d((1, 1))                              # 也可使用nn.AvgPool2d()
        self.Drop = nn.Dropout(0.4)
        self.f = nn.Linear(1024, num_classes)
        if init_weight:
            self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.Conv1(x)
        x = self.MaxPool1(x)
        x = self.Conv2(x)
        x = self.Conv3(x)
        x = self.MaxPool2(x)
        x = self.Inception3a(x)
        x = self.Inception3b(x)
        x = self.MaxPool3(x)
        x = self.Inception4a(x)                                                 # 根据网络图 再对应Inception层处进行辅助分类器操作
        if self.training and self.aux_enable:
            aux1 = self.aux1a(x)
        x = self.Inception4b(x)
        x = self.Inception4c(x)
        x = self.Inception4d(x)
        if self.training and self.aux_enable:
            aux2 = self.aux1b(x)
        x = self.Inception4e(x)
        x = self.MaxPool4(x)
        x = self.Inception5a(x)
        x = self.Inception5b(x)
        x = self.average(x)
        x = torch.flatten(x, 1)
        x = self.Drop(x)
        x = self.f(x)
        if self.training and self.aux_enable:
            return x, aux1, aux2                                               # 在train函数中，根据设定的权重加和
        return x

class Inception(nn.Module):                                                     # Inception层
    def __init__(self, in_channels, ch11, chr33red, ch33, ch55red, ch55, pool):
        super(Inception, self).__init__()
        self.batch1 = Basic_Conv2d(in_channels, ch11, kernel_size=1)           # 分支1  1×1的卷积
        self.batch2 = nn.Sequential(                                           # 分支2  降维后进行3×3的卷积
            Basic_Conv2d(in_channels, chr33red, kernel_size=1),
            Basic_Conv2d(chr33red, ch33, kernel_size=3, padding=1)
        )
        self.batch3 = nn.Sequential(                                           # 分支3  降维后进行5×5的卷积
            Basic_Conv2d(in_channels, ch55red, kernel_size=1),
            Basic_Conv2d(ch55red, ch55, kernel_size=5, padding=2)
        )
        self.batch4 = nn.Sequential(                                           # 分支4  最大池化后降维
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            Basic_Conv2d(in_channels, pool, kernel_size=1)
        )

    def forward(self, x):
        batch1 = self.batch1(x)
        batch2 = self.batch2(x)
        batch3 = self.batch3(x)
        batch4 = self.batch4(x)
        outputs = [batch1, batch2, batch3, batch4]
        return torch.cat(outputs, 1)                                            # 结果拼接

class InceptionAux(nn.Module):                                                  # 创建辅助分类器 最大池化+卷积+2*FC层+softmax
    def __init__(self, in_channels,num_classes):
        super(InceptionAux, self).__init__()
        self.AveragePool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = Basic_Conv2d(in_channels, 128, kernel_size=1)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.AveragePool(x)
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = F.dropout(x, 0.5, training=self.training)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.dropout(x, 0.5, training=self.training)
        x = self.fc2(x)
        return x

class Basic_Conv2d(nn.Module):                                                  # 创建卷积层
    def __init__(self, in_channels, outchannel, **kwargs):
        super(Basic_Conv2d, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, outchannel, **kwargs)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv2d(x)
        x = self.relu(x)
        return x
