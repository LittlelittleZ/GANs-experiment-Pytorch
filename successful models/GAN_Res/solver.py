import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.special as f
from model import ResNetGenerator, ResNetDiscriminator
from torch import autograd
import copy
import numpy as np

class GANModel():
    def __init__(self, config):
        self.ngpu = config['ngpu']
        self.batch_size = config['g_batch_size']
        self.d_batch_size = config['g_batch_size']
        self.lr = config['lr']
        self.device = 'cuda:0'
        self.z_dim = config['z_dim']
        self.max_step = config['max_step']        
        self.G = ResNetGenerator().to(self.device)
        self.D = ResNetDiscriminator().to(self.device)
        self.loss = nn.BCELoss().to(self.device)
        self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=self.lr, betas=(0, 0.9))
        self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=self.lr, betas=(0, 0.9))

    def set_input(self, imgs_eachgen, data):
        z = torch.randn(imgs_eachgen, self.z_dim)
        self.z = z.to(self.device)
        self.real = data.to(self.device)


    def set_requires_grad(self, nets, requires_grad=False):
        for param in nets.parameters():
            param.requires_grad_(requires_grad)

    def forward(self):
        self.fake = self.G(self.z)

    def backward_D(self):
        x_real = self.real
        output = self.D(x_real).view(-1)
        real_labels = torch.ones(output.size()).to(self.device)
        errD_real = self.loss(output, real_labels)
        errD_real.backward()
        D_x = output.mean().item()
        output = self.D(self.fake.detach())
        fake_labels = torch.zeros(output.size()).to(self.device)
        errD_fake = self.loss(output, fake_labels)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        lossD = errD_real + errD_fake
        return D_x, D_G_z1


    def backward_G(self):

        #x_fake = self.G(self.z)
        d_fake = self.D(self.fake)
        real_labels = torch.ones(d_fake.size()).to(self.device)
        gloss = self.loss(d_fake, real_labels)
        gloss.backward()
        D_G_z2 = d_fake.view(-1).mean().item()
        return D_G_z2

    def optimize_parametersD(self):
        self.forward()
        self.set_requires_grad(self.D, True)
        self.optimizer_D.zero_grad()
        D_x, D_G_z1 = self.backward_D()
        self.optimizer_D.step()
        return D_x, D_G_z1

    def optimize_parametersG(self):
        self.forward()
        self.set_requires_grad(self.D, False)
        self.optimizer_G.zero_grad()
        D_G_z2 = self.backward_G()
        self.optimizer_G.step()
        return D_G_z2
