from typing import List, Callable
import torch
from torch import tensor, Tensor
import torch.nn as nn

def channel_shufflle(x: Tensor, groups: int)->Tensor:
    batch_size, num_channel, height, width = x.size()
    channel_group = num_channel // groups
    x = x.view(batch_size, groups, channel_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)
    return x

class Invert_Residual(nn.Module):
    def __init__(self, input_c: int, output_c: int , stride: int):
        super(Invert_Residual, self).__init__()
        if stride not in [1, 2]:
            raise ValueError("error stride")
        self.stride = stride
        assert output_c % 2 == 0
        branch_feature = output_c//2
        assert (self.stride != 1) or (input_c == branch_feature << 1)

        if self.stride == 2:
            self.branch1 = nn.Sequential(
                self.depth_Con(input_c, input_c, kernel_s=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(input_c),
                nn.Conv2d(input_c, branch_feature, kernel_size=1, stride=self.stride, padding=0, bias=False),
                nn.BatchNorm2d(branch_feature),
                nn.ReLU(inplace=True)
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(input_c if self.stride > 1 else branch_feature, branch_feature, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_feature),
            nn.ReLU(inplace=True),
            self.depth_Con(branch_feature, branch_feature, kernel_s=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_feature),
            nn.Conv2d(branch_feature, branch_feature, kernel_size=1, stride=1, padding=0,bias=False),
            nn.BatchNorm2d(branch_feature),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depth_Con(input_c: int,
                  output_c: int,
                  kernel_s: int,
                  stride: int = 1,
                  padding: int = 0,
                  bias: bool = False) -> nn.Conv2d:
        return nn.Conv2d(in_channels=input_c, out_channels=output_c, kernel_size=kernel_s,
                         stride=stride, padding=padding, bias=bias, groups=input_c)

    def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch2(x), self.branch2(x)), dim=1)

        out = channel_shufflle(out, 2)
        return out

class ShuffleNet(nn.Module):
    def __init__(self,
                 stage_repeat:List[int],
                 stage_out_channel:List[int],
                 num_class: int = 1000,
                 inverted_residual: Callable[..., nn.Module] = Invert_Residual
                 ):
        super(ShuffleNet, self).__init__()
        if len(stage_repeat) != 3:
            raise ValueError("stage_repeat error!")
        if len(stage_out_channel) != 5:
            raise ValueError("stage_out_channel error!")
        self._stage_out_channel = stage_out_channel
        input_channel = 3
        output_channel = self._stage_out_channel[0]
        self.Con1 = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True)
        )
        input_channel = output_channel
        self.Maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage2: nn.Sequential
        self.stage3: nn.Sequential
        self.stage4: nn.Sequential
        stage_name = ["stage{}".format(i) for i in [2, 3, 4]]
        for name, repeat, output_channel in zip(stage_name, stage_repeat, self._stage_out_channel[1:]):
            seq = [Invert_Residual(input_channel, output_channel, 2)]
            for i in range(repeat-1):
                seq.append(Invert_Residual(output_channel, output_channel, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channel = output_channel

        output_channel = self._stage_out_channel[-1]
        self.Con5 = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(output_channel, num_class)

    def _forward(self, x: Tensor) -> Tensor:
        x = self.Con1(x)
        x = self.Maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.Con5(x)
        x = x.mean([2, 3])
        x = self.fc(x)
        return x

    def forward(self,  x: Tensor) -> Tensor:
        return self._forward(x)

def Shuffle_NetV2_x1_0(num_classes = 1000):
    model = ShuffleNet(
        stage_repeat=[4, 8, 4],
        stage_out_channel=[24, 116, 232, 464, 1024],
        num_class=num_classes
    )
    return model

def Shuffle_NetV2_x0_5(num_classes = 1000):
    model = ShuffleNet(
        stage_repeat=[4, 8, 4],
        stage_out_channel=[24, 48, 96, 192, 1024],
        num_class=num_classes
    )
    return model
