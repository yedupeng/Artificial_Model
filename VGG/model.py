import torch.nn as nn
import torch

class VGG(nn.Module):
    def __init__(self, features, class_num=1000, init_weights=False):
        super(VGG, self).__init__()
        self.features = features
        self.classers = nn.Sequential(           #定义全连接层
            nn.Dropout(0.5),
            nn.Linear(512*7*7, 2048),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Linear(2048, class_num)
        )
        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):               #初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classers(x)
        return x

def make_features(cfg:list):                     #如果是’M‘则进行最大池化、如果是数字则进行卷积堆叠
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(True)]
            in_channels = v
    return nn.Sequential(*layers)              #将网络模块返回

cfgs = {                                       #根据网络参数图按顺序写入参数
    'Vgg11': [64, 'M', '128', 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'Vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'Vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'Vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512,  'M'],
}

def Vgg(model_name = "Vgg16", **kwargs):
    try:
        cfg = cfgs[model_name]                 #根据model_name选择对应网络参数
    except:
        print("warning!")
        exit(-1)
    model = VGG(make_features(cfg), **kwargs)  #数据带入网络参数
    return model


Vgg_model = Vgg(model_name="Vgg13")

