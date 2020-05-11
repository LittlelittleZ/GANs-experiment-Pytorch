import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# from torchsummary import summary
import numpy as np


def _upsample(x):
    h, w = x.shape[2:]
    return F.interpolate(x, size=(h * 2, w * 2))


def upsample_conv(x, conv):
    return conv(_upsample(x))


class Generator(torch.nn.Module):
    def __init__(self, channels=3):
        super(Generator, self).__init__()
        # Filters [1024, 512, 256]
        # Input_dim = 128
        # Output_dim = C (number of channels)
        self.activation = nn.ReLU(True)
        self.bottom_width = 4
        self.ch = 512

        # Z latent vector 128
        self.l1 = nn.Linear(128, (self.bottom_width ** 2) * self.ch)
        nn.init.xavier_uniform_(self.l1.weight.data)
        # State (512x4x4)
        self.conv1 = nn.ConvTranspose2d(in_channels=512, out_channels=256,
                                        kernel_size=4, stride=2, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight.data, math.sqrt(2))
        self.b1 = nn.BatchNorm2d(num_features=256)
        # State (256x8x8)
        self.conv2 = nn.ConvTranspose2d(in_channels=256, out_channels=128,
                                        kernel_size=4, stride=2, padding=1)
        nn.init.xavier_uniform_(self.conv2.weight.data, math.sqrt(2))
        self.b2 = nn.BatchNorm2d(num_features=128)
        # State (128x16x16)
        self.conv3 = nn.ConvTranspose2d(in_channels=128, out_channels=64,
                                        kernel_size=4, stride=2, padding=1)
        nn.init.xavier_uniform_(self.conv3.weight.data, math.sqrt(2))
        self.b3 = nn.BatchNorm2d(num_features=64)
        # State (64x32x32)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=channels,
                               kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.conv4.weight.data)
        # State (3x32x32)
        self.tanh = nn.Tanh()

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

    def forward(self, x):
        x0 = self.l1(x)
        x = x0.view(x0.size(0), -1, self.bottom_width, self.bottom_width)
        out = self.b1(self.conv1(x))
        out = self.activation(out)
        out = self.b2(self.conv2(out))
        out = self.activation(out)
        out = self.b3(self.conv3(out))
        out = self.activation(out)
        out = self.tanh(self.conv4(out))
        return out


class Discriminator(torch.nn.Module):
    def __init__(self, channels=3):
        super(Discriminator, self).__init__()
        # Filters [256, 512, 1024]
        # Input_dim = channels (Cx64x64)
        # Output_dim = 1
        self.activation = nn.LeakyReLU(0.1, inplace=True)
        # Image (Cx32x32)
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(
            in_channels=channels, out_channels=64, kernel_size=3, stride=1, padding=1))
        nn.init.xavier_uniform_(self.conv1.weight.data, math.sqrt(2))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1))
        nn.init.xavier_uniform_(self.conv2.weight.data, math.sqrt(2))
        # State (256x16x16)
        self.conv3 = nn.utils.spectral_norm(nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1))
        nn.init.xavier_uniform_(self.conv3.weight.data, math.sqrt(2))
        self.conv4 = nn.utils.spectral_norm(nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1))
        nn.init.xavier_uniform_(self.conv4.weight.data, math.sqrt(2))
        # State (512x8x8)
        self.conv5 = nn.utils.spectral_norm(nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0))
        nn.init.xavier_uniform_(self.conv5.weight.data, math.sqrt(2))
        self.conv6 = nn.utils.spectral_norm(nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1))
        nn.init.xavier_uniform_(self.conv6.weight.data, math.sqrt(2))
        # State (1024x4x4)
        self.conv7 = nn.utils.spectral_norm(nn.Conv2d(
            in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=0))
        nn.init.xavier_uniform_(self.conv7.weight.data)
        self.l8 = nn.utils.spectral_norm(nn.Linear(512, 1, bias=False))
        nn.init.xavier_uniform_(self.l8.weight.data)
        # Output 1
        # self.initial()

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

    def forward(self, x):
        out = self.activation(self.conv1(x))
        out = self.activation(self.conv2(out))
        out = self.activation(self.conv3(out))
        out = self.activation(self.conv4(out))
        out = self.activation(self.conv5(out))
        out = self.activation(self.conv6(out))
        out = self.activation(self.conv7(out))
        out = self.l8(out.view(out.size(0), -1))
        return out

    # def feature_extraction(self, x):
    #     # Use discriminator for feature extraction then flatten to vector of 16384 features
    #     x = self.main_module(x)
    #     return x.view(-1, 1024 * 4 * 4)


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


def _downsample(x):
    return F.avg_pool2d(x, 2)


class disBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, ksize=3, pad=1,
                 activation=F.relu, downsample=False):
        super(disBlock, self).__init__()
        self.activation = activation
        self.downsample = downsample
        self.learnable_sc = (in_channels != out_channels) or downsample
        hidden_channels = in_channels if hidden_channels is None else hidden_channels
        self.c1 = nn.utils.spectral_norm(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=ksize, padding=pad))  # 谱归一化
        nn.init.xavier_uniform_(self.c1.weight.data, math.sqrt(2))
        self.c2 = nn.utils.spectral_norm(nn.Conv2d(hidden_channels, out_channels, kernel_size=ksize, padding=pad))
        nn.init.xavier_uniform_(self.c2.weight.data, math.sqrt(2))
        if self.learnable_sc:
            self.c_sc = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0))
            nn.init.xavier_uniform_(self.c_sc.weight.data)

    def residual(self, x):
        h = x
        h = self.activation(h)
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        if self.downsample:
            h = _downsample(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
            if self.downsample:
                return _downsample(x)
            else:
                return x
        else:
            return x

    def forward(self, input):
        return self.residual(input) + self.shortcut(input)


class OptimizedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=3, pad=1, activation=F.relu):
        super(OptimizedBlock, self).__init__()
        self.activation = activation
        self.c1 = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=ksize, padding=pad))
        nn.init.xavier_uniform_(self.c1.weight.data, math.sqrt(2))
        self.c2 = nn.utils.spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=ksize, padding=pad))
        nn.init.xavier_uniform_(self.c2.weight.data, math.sqrt(2))
        self.c_sc = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0))
        nn.init.xavier_uniform_(self.c_sc.weight.data)

    def residual(self, x):
        h = x
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        h = _downsample(h)
        return h

    def shortcut(self, x):
        return self.c_sc(_downsample(x))

    def forward(self, input):
        return self.residual(input) + self.shortcut(input)


class ResNetDiscriminator(nn.Module):
    def __init__(self, ch=128, activation=F.relu):
        super(ResNetDiscriminator, self).__init__()
        self.activation = activation
        self.block1 = OptimizedBlock(3, ch)
        self.block2 = disBlock(ch, ch, activation=activation, downsample=True)
        self.block3 = disBlock(ch, ch, activation=activation, downsample=False)
        self.block4 = disBlock(ch, ch, activation=activation, downsample=False)
        self.l5 = nn.utils.spectral_norm(nn.Linear(ch, 1, bias=False))
        nn.init.xavier_uniform_(self.l5.weight.data)
        self.initial()

    def initial(self):
        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                nn.init.constant_(m.bias.data, 0)
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

        self.apply(weights_init)

    def forward(self, input):
        h = input
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.activation(h)
        # Global average pooling
        h = torch.sum(h, dim=(2, 3))
        output = self.l5(h)
        return output


def weights_init(m):
    classname = m.__class__.__name__
    print(m)
    print(classname)
    if classname.find('Conv2d') != -1:
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

#
# def test():
#     net = Discriminator()
#     y = net(torch.randn(6, 3,32,32))
#     print(y.size())

# test()
# summary(ResNetDiscriminator(), (3, 32, 32))
