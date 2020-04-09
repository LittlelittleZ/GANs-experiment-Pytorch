import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


def _upsample(x):
    h, w = x.shape[2:]
    return F.interpolate(x, size=(h * 2, w * 2))


def upsample_conv(x, conv):
    return conv(_upsample(x))


class genBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 activation=F.relu, hidden_channels=None, ksize=3, pad=1, upsample=False):
        super(genBlock, self).__init__()
        self.activation = activation
        self.upsample = upsample
        self.learnable_sc = in_channels != out_channels or upsample
        hidden_channels = out_channels if hidden_channels is None else hidden_channels
        self.c1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=ksize, padding=pad)
        nn.init.xavier_uniform_(self.c1.weight.data, math.sqrt(2))
        self.c2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=ksize, padding=pad)
        nn.init.xavier_uniform_(self.c2.weight.data, math.sqrt(2))
        self.b1 = nn.BatchNorm2d(in_channels)
        self.b2 = nn.BatchNorm2d(hidden_channels)
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
            nn.init.xavier_uniform_(self.c_sc.weight.data)

    def residual(self, x):
        h = x
        h = self.b1(h)
        h = self.activation(h)
        h = upsample_conv(h, self.c1) if self.upsample else self.c1(h)
        h = self.b2(h)
        h = self.activation(h)
        h = self.c2(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = upsample_conv(x, self.c_sc) if self.upsample else self.c_sc(x)
            return x
        else:
            return x

    def forward(self, input):
        return self.residual(input) + self.shortcut(input)


class ResNetGenerator(nn.Module):
    def __init__(self, ch=256, dim_z=128, bottom_width=4, activation=F.relu, distribution="normal"):
        super(ResNetGenerator, self).__init__()
        self.bottom_width = bottom_width
        self.activation = activation
        self.distribution = distribution
        self.dim_z = dim_z
        self.l1 = nn.Linear(dim_z, (bottom_width ** 2) * ch)
        nn.init.xavier_uniform_(self.l1.weight.data)
        self.block2 = genBlock(ch, ch, activation=activation, upsample=True)
        self.block3 = genBlock(ch, ch, activation=activation, upsample=True)
        self.block4 = genBlock(ch, ch, activation=activation, upsample=True)
        self.b5 = nn.BatchNorm2d(ch)
        self.c5 = nn.Conv2d(ch, 3, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.c5.weight.data)
        self.initial()

    def initial(self):
        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                nn.init.constant_(m.bias.data, 0)
            elif classname.find('Linear') != -1:
                nn.init.constant_(m.bias.data, 0)
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

        self.apply(weights_init)

    def forward(self, input):
        h = input
        h0 = self.l1(h)
        h = h0.view(h0.size(0), -1, self.bottom_width, self.bottom_width)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.b5(h)
        h = self.activation(h)
        h = torch.tanh(self.c5(h))
        return h


# 自制ResNet50代码块，cifar10数据库Discriminator，z_dim=128
def _downsample(input):
    return F.avg_pool2d(input, 2)


class Dis_Bottleneck(nn.Module):
    # 前面1x1和3x3卷积的filter个数相等，最后1x1卷积是其expansion倍
    expansion = 4

    def __init__(self, in_channels, out_channels, downsample=False, stride=1):
        super(Dis_Bottleneck, self).__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, self.expansion * out_channels,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        if self.downsample:
            out = _downsample(out)
        return out


class ResNet_Discriminator(nn.Module):
    def __init__(self, block, num_blocks, ch=128):
        super(ResNet_Discriminator, self).__init__()
        self.ch = ch

        self.conv1 = nn.Conv2d(3, ch, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.layer1 = self._make_layer(block, ch, num_blocks[0], downsample=True, stride=1)
        self.layer2 = self._make_layer(block, ch, num_blocks[1], downsample=False, stride=2)
        self.layer3 = self._make_layer(block, ch, num_blocks[2], downsample=False, stride=2)
        self.layer4 = self._make_layer(block, ch, num_blocks[3], downsample=False, stride=2)
        self.linear = nn.Linear(ch * block.expansion, 1)

    def _make_layer(self, block, out_channels, num_blocks, downsample, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.ch, out_channels, downsample, stride))
            self.ch = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = torch.sum(out, dim=(2, 3))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNetDiscriminator():
    return ResNet_Discriminator(Dis_Bottleneck, [3, 4, 6, 3])

#
# def test():
#     net = ResNetGenerator()
#     y = net(torch.randn(128, 128))
#     print(y)
#
#
# #
# #
# test()
# summary(ResNetGenerator(), (128, 128))
