import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


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


def ResNet50_Discriminator():
    return ResNet_Discriminator(Dis_Bottleneck, [3, 4, 6, 3])


def _upsample(x):
    h, w = x.shape[2:]
    return F.interpolate(x, size=(h * 2, w * 2))


def upsample_conv(x, conv):
    return conv(_upsample(x))


class Gen_Bottleneck(nn.Module):
    # 前面1x1和3x3卷积的filter个数相等，最后1x1卷积是其expansion倍
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, upsample=False):
        super(Gen_Bottleneck, self).__init__()
        self.upsample = upsample
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
        out = F.relu(self.bn1(upsample_conv(x, self.conv1))) if self.upsample else \
            F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(x)))
        out = self.bn3(self.conv3(out))
        if self.upsample:
            out += upsample_conv(x, self.shortcut)
        else:
            out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_Generator(nn.Module):
    def __init__(self, block, num_blocks, ch=256, dim_z=128, bottom_width=4, ):
        super(ResNet_Generator, self).__init__()
        self.ch = ch
        self.dim_z = dim_z
        self.linear = nn.Linear(dim_z, (bottom_width ** 2) * ch)
        self.layer1 = self._make_layer(block, ch, num_blocks[0], upsample=True, stride=1)
        self.layer2 = self._make_layer(block, ch, num_blocks[1], upsample=False, stride=2)
        self.layer3 = self._make_layer(block, ch, num_blocks[2], upsample=False, stride=2)
        self.layer4 = self._make_layer(block, ch, num_blocks[3], upsample=False, stride=2)
        self.bn5 = nn.BatchNorm2d(ch)
        self.conv5 = nn.Conv2d(ch, 3, kernel_size=3,
                               stride=1, padding=1)

    def _make_layer(self, block, planes, num_blocks, upsample, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.ch, planes, upsample, stride))
            self.ch = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out0 = F.linear(x)
        out = out0.view(out0.size(0), -1, self.bottom_width, self.bottom_width)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.bn5(out)
        out = F.relu(out)
        out = torch.tanh(self.conv5(out))
        return out


def ResNet50_Generator():
    return ResNet_Generator(Gen_Bottleneck, [3, 4, 6, 3])


###########################################################################################
###########################################################################################


# 用于ResNet18和34的残差块，用的是2个3x3的卷积
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        # 经过处理后的x要与x的维度相同(尺寸和深度)
        # 如果不相同，需要添加卷积+BN来变换为同一维度
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# 用于ResNet50,101和152的残差块，用的是1x1+3x3+1x1的卷积
class Bottleneck(nn.Module):
    # 前面1x1和3x3卷积的filter个数相等，最后1x1卷积是其expansion倍
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
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
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, ch=256, dim_z=128, bottom_width=4, ):
        super(ResNet, self).__init__()
        self.ch = ch
        self.dim_z = dim_z
        self.linear = nn.Linear(dim_z, (bottom_width ** 2) * ch)
        self.layer1 = self._make_layer(block, ch, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, ch, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, ch, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, ch, num_blocks[3], stride=2)
        self.bn5 = nn.BatchNorm2d(ch)
        self.conv5 = nn.Conv2d(ch, 3, kernel_size=3,
                               stride=1, padding=1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.ch, planes, stride))
            self.ch = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out0 = F.linear(x)
        out = out0.view(out0.size(0), -1, self.bottom_width, self.bottom_width)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.bn5(out)
        out = F.relu(out)
        out = torch.tanh(self.conv5(out))
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet50_Discriminator()
    y = net(torch.randn(1, 3, 32, 32))
    print(y)


test()
# summary(ResNet50_Discriminator(), (3, 32, 32))
