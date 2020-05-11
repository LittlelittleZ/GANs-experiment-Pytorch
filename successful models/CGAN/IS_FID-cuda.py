import os
import sys
import time
import glob
import numpy as np
import torch
import logging
import argparse
import math
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.models.inception as incepnets
from torch.nn import init
import torch.backends.cudnn as cudnn
import utilities.Sampler as Sampler

from torch.autograd import Variable
from scipy.stats import entropy
from scipy import linalg
from inception_net import InceptionV3


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


def cal_IS(model, dim_z, bs=50, n_total=5000, splits=1, resize=True):
    model.eval()
    inception = incepnets.inception_v3(pretrained=True).cuda()
    inception.eval()

    def get_pred(x):
        if resize:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=True)
        x = inception(x)
        return F.softmax(x).data.cpu().numpy()

    preds = np.zeros((n_total, 1000))
    for i in range(n_total // bs):
        z = Variable(torch.randn(bs, dim_z)).cuda()
        gen = model(z.view(bs, dim_z)).detach()
        preds[i * bs:i * bs + bs] = get_pred(gen)

    split_scores = []
    for k in range(splits):
        part = preds[k * (n_total // splits):(k + 1) * (n_total // splits), :]
        py = part.mean(0)
        scores = []
        for j in range(part.shape[0]):
            pyx = part[i, :]
            # scores.append(np.sum(pyx * np.log(pyx / py), axis=0))
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def cal_FID(model, dim_z, dataset, bs=50, n_total=5000, splits=1, dim_feat=2048, resize=True):
    model.eval()
    # inception = incepnets.inception_v3(pretrained=True).cuda()
    # inception.eval()
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dim_feat]
    inception = InceptionV3([block_idx]).cuda()
    inception.eval()

    def get_pred(x):
        if resize:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=True)
        x = inception(x)[0].view(bs, dim_feat)
        return x.data.cpu().numpy()

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True)
    feat0 = np.zeros((n_total, dim_feat))
    feat = np.zeros((n_total, dim_feat))
    for i, data in enumerate(dataloader):
        if i * bs >= n_total:
            break
        img = data[0].cuda()
        feat0[i * bs:i * bs + bs] = get_pred(img)
        z = Variable(torch.randn(bs, dim_z)).cuda()
        gen = model(z.view(bs, dim_z)).detach()
        feat[i * bs:i * bs + bs] = get_pred(gen)

    split_scores = []
    for k in range(splits):
        part = feat[k * (n_total // splits):(k + 1) * (n_total // splits), :]
        part0 = feat0[k * (n_total // splits):(k + 1) * (n_total // splits), :]

        mu = np.mean(part, axis=0)
        sigma = np.cov(part, rowvar=False)
        mu0 = np.mean(part0, axis=0)
        sigma0 = np.cov(part0, rowvar=False)

        split_scores.append(calculate_frechet_distance(mu, sigma, mu0, sigma0))

    return np.mean(split_scores), np.std(split_scores)


if __name__ == '__main__':
    # device = torch.device("cuda:0" if opt.cuda else "cpu")
    ngpu = 0
    nz = 100
    nc = 3
    ngf = 64
    ndf = 64

    net = Generator()
    net.load_state_dict(torch.load('300_SN_RES_G.pth', map_location='cuda:0'))
    net.cuda()
    print('IS:{}'.format(cal_IS(net, dim_z=128, n_total=25000, splits=5)))

    dataset = dset.CIFAR10(root='./PyCharm_SN_Cifar10/cifar-10', download=True,
                           transform=transforms.Compose(
                               [transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                transforms.Lambda(lambda x: x + 1. / 128 * torch.rand(x.size())), ]))
    print('FID:{}'.format(cal_FID(net, dim_z=128, dataset=dataset, n_total=25000, splits=5)))
