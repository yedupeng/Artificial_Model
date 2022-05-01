import torch.nn as nn
import torch

class Alexnet(nn.Module):
    def __init__(self, num_class=1000, init_weight=False):
        super(Alexnet, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),         #  为了减少计算量，卷积核比模型中小一倍
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(48, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classfier = nn.Sequential(                                   #   定义全连接层
            nn.Dropout(p=0.5),                                            #   Dropout用于随机失活比例0.5的神经元
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_class),
        )
        if init_weight:
            self._initialize_weights()

    def forward(self, x):
        x = self.feature(x)
        x = torch.flatten(x, start_dim=1)                                 #    将维度转为1维再传入全连接层计算
        x = self.classfier(x)
        return x

    def _initialize_weights(self):                                        #    初始化权重以及偏置
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)