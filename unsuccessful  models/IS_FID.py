import os
import sys
import time
import glob
import numpy as np
import torch
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.models.inception as incepnets
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from scipy.stats import entropy
from scipy import linalg
from inception_net import InceptionV3
from model import ResNetGenerator as Generator


# class Generator(nn.Module):
#     def __init__(self, ngpu):
#         super(Generator, self).__init__()
#         self.ngpu = ngpu
#         self.main = nn.Sequential(
#             # input is Z, going into a convolution
#             nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
#             #nn.BatchNorm2d(ngf * 8),
#             nn.ReLU(True),
#             # state size. (ngf*8) x 4 x 4
#             nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
#             #nn.BatchNorm2d(ngf * 4),
#             nn.ReLU(True),
#             # state size. (ngf*4) x 8 x 8
#             nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
#             #nn.BatchNorm2d(ngf * 2),
#             nn.ReLU(True),
#             # state size. (ngf*2) x 16 x 16
#             nn.ConvTranspose2d(ngf * 2,     nc, 4, 2, 1, bias=False),
#             #nn.BatchNorm2d(ngf),
#             #nn.ReLU(True),
#             ## state size. (ngf) x 32 x 32
#             #nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
#             nn.Tanh()
#             # state size. (nc) x 64 x 64
#         )
#
#     def forward(self, input):
#         if input.is_cuda and self.ngpu > 1:
#             output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
#         else:
#             output = self.main(input)
#         return output


def cal_IS(model, dim_z, bs=50, n_total=5000, splits=1, resize=True):
  model.eval()
  inception = incepnets.inception_v3(pretrained=True).cuda()
  inception.eval()
  up = nn.Upsample(size=(299, 299), mode='bilinear')
  def get_pred(x):
    if resize:
      x = up(x)
    x = inception(x)
    return F.softmax(x).data.cpu().numpy()

  preds = np.zeros((n_total, 1000))
  for i in range(n_total//bs):
    z = Variable(torch.randn(bs, dim_z)).cuda()
    gen = model(z.view(bs, dim_z, 1, 1)).detach()
    preds[i*bs:i*bs+bs] = get_pred(gen)

  split_scores = []
  for k in range(splits):
    part = preds[k*(n_total//splits):(k+1)*(n_total//splits), :]
    py = part.mean(0)
    scores = []
    for j in range(part.shape[0]):
      pyx = part[i, :]
      #scores.append(np.sum(pyx * np.log(pyx / py), axis=0))
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
  #inception = incepnets.inception_v3(pretrained=True).cuda()
  #inception.eval()
  block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dim_feat]
  inception = InceptionV3([block_idx]).cuda()
  inception.eval()

  up = nn.Upsample(size=(299, 299), mode='bilinear')
  def get_pred(x):
    if resize:
      x = up(x)
    x = inception(x)[0].view(bs, dim_feat)
    return x.data.cpu().numpy()

  dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True)
  feat0 = np.zeros((n_total, dim_feat))
  feat = np.zeros((n_total, dim_feat))
  for i, data in enumerate(dataloader):
    if i*bs >= n_total:
      break
    img = data[0].cuda()
    feat0[i*bs:i*bs+bs] = get_pred(img)
    z = Variable(torch.randn(bs, dim_z)).cuda()
    gen = model(z.view(bs, dim_z, 1, 1)).detach()
    feat[i*bs:i*bs+bs] = get_pred(gen)

  split_scores = []
  for k in range(splits):
    part = feat[k*(n_total//splits):(k+1)*(n_total//splits), :]
    part0 = feat0[k*(n_total//splits):(k+1)*(n_total//splits), :]
    
    mu = np.mean(part, axis=0)
    sigma = np.cov(part, rowvar=False)
    mu0 = np.mean(part0, axis=0)
    sigma0 = np.cov(part0, rowvar=False)

    split_scores.append( calculate_frechet_distance(mu, sigma, mu0, sigma0) )

  return np.mean(split_scores), np.std(split_scores)

if __name__ == '__main__':
    #device = torch.device("cuda:0" if opt.cuda else "cpu")
    ngpu = 1
    nz = 100
    nc = 3
    ngf = 64
    ndf = 64

    net = Generator(256)
    net.load_state_dict(torch.load('./checkpoint/350_G.pth'))
    net.cuda()
    print('IS:{}'.format(cal_IS(net, dim_z=128, n_total=25000, splits=5)))

    dataset = dset.CIFAR10(root='../cifar-10', download=True,
                       transform=transforms.Compose([
                           transforms.Resize(32),
                           transforms.ToTensor(),
                           #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                       ]))
    print('FID:{}'.format(cal_FID(net, dim_z=100, dataset=dataset, n_total=25000, splits=5)))
