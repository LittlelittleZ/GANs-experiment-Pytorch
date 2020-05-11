import os
import time
import datetime
import functools

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image

from utilities.Utilities import *
from tensorboardX import SummaryWriter
from utilities.Reporter import Reporter
import utilities.Sampler as Sampler

from torchviz import make_dot
from sngan_net import ResNetGenerator, ResNetDiscriminator


class Trainer(object):
    def __init__(self, data_loader, config):

        # Data loader
        self.data_loader = data_loader

        # exact model and loss
        self.cGAN = config.cGAN
        self.adv_loss = config.adv_loss

        # Model hyper-parameters
        self.imsize = config.imsize
        self.z_dim = 128
        self.g_conv_dim = 256
        self.d_conv_dim = 128
        self.num_gens = 10
        self.n_classes = config.n_class if config.cGAN else 0
        self.parallel = config.parallel
        self.seed = config.seed
        self.device = torch.device('cuda:0')
        self.GPUs = config.GPUs

        self.gen_distribution = config.gen_distribution
        self.gen_bottom_width = 4

        self.total_step = config.total_step
        self.batch_size = config.batch_size
        self.gbatch_size = config.gbatch_size
        self.num_workers = config.num_workers
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.lr_decay = config.lr_decay
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        self.use_pretrained_model = config.use_pretrained_model
        self.chechpoint_step = config.chechpoint_step
        self.use_pretrained_model = config.use_pretrained_model

        self.dataset = config.dataset
        self.image_path = config.image_path
        self.model_save_path = config.model_save_path
        self.sample_path = config.sample_path
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.DStep = 3
        self.GStep = 1
        self.build_model()


        self.metric_caculation_step = config.metric_caculation_step

        # Start with trained model

    def build_model(self):
        self.G = ResNetGenerator(self.g_conv_dim, self.z_dim, self.gen_bottom_width,
                                 num_gens=self.num_gens).to(self.device)
        self.D = ResNetDiscriminator(self.d_conv_dim).to(self.device)
        self.D.load_state_dict(torch.load('./240000_D.pth'))
        self.G.load_state_dict(torch.load('./240000_G.pth'))
        if self.parallel:
            self.G = nn.DataParallel(self.G, device_ids=self.GPUs)
            self.D = nn.DataParallel(self.D, device_ids=self.GPUs)
        # Loss and optimizer
        self.g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()), self.g_lr,
                                            [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D.parameters()), self.d_lr,
                                            [self.beta1, self.beta2])

    def train(self):

        # Data iterator
        data_iter = iter(self.data_loader)
        model_save_step = self.model_save_step

        # Fixed input for debugging
        sampleBatch = 10
        fixed_z = torch.randn(self.n_classes * sampleBatch, self.z_dim)
        fixed_z = fixed_z.to(self.device)
        fixed_c = Sampler.sampleFixedLabels(self.n_classes, sampleBatch, self.device)

        runingZ, runingLabel = Sampler.prepare_z_c(self.gbatch_size, self.z_dim, self.n_classes, device=self.device)

        # Start with trained model

        # Start time
        start_time = time.time()
        dstepCounter = 0
        gstepCounter = 0
        for step in range(0, self.total_step):
            # ================== Train D ================== #
            self.D.train()
            self.G.train()
            self.set_requires_grad(self.D, True)
            if dstepCounter < self.DStep:
                try:
                    realImages, realLabel = next(data_iter)
                except:
                    data_iter = iter(self.data_loader)
                    realImages, realLabel = next(data_iter)

                # Compute loss with real images
                realImages = realImages.to(self.device)
                realLabel = realLabel.to(self.device).long()
                d_out_real = self.D(realImages, realLabel)
                d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()

                # apply Gumbel Softmax
                runingZ.sample_()
                runingLabel.sample_()
                fake_images = self.G(runingZ, runingLabel)
                d_out_fake = self.D(fake_images, runingLabel)
                d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()
                # Backward + Optimize
                d_loss = d_loss_real + d_loss_fake
                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()
                dstepCounter += 1
            else:
                # ================== Train G and gumbel ================== #
                # Create random noise
                self.set_requires_grad(self.D, False)
                runingZ.sample_()
                runingLabel.sample_()
                fake_images = self.G(runingZ, runingLabel)

                # Compute loss with fake images
                g_out_fake = self.D(fake_images, runingLabel)
                g_loss_fake = - g_out_fake.mean()

                self.reset_grad()
                g_loss_fake.backward()
                self.g_optimizer.step()
                gstepCounter += 1

            if gstepCounter == self.GStep:
                dstepCounter = 0
                gstepCounter = 0

            # Print out log info
            if (step + 1) % self.log_step == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                print("Elapsed [{}], G_step [{}/{}], D_step[{}/{}], d_out_real: {:.4f}, "
                      " d_loss_fake: {:.4f}, g_loss_fake: {:.4f}".
                      format(elapsed, step + 1, self.total_step, (step + 1),
                             self.total_step, d_loss_real.item(),
                             d_loss_fake.item(), g_loss_fake.item()))

            if (step + 1) % self.sample_step == 0:
                fake_images = self.G(fixed_z, fixed_c)
                save_image(denorm(fake_images.data),
                           os.path.join(self.sample_path, '{}_fake.png'.format(step + 240001)), nrow=self.n_classes)

            if (step + 1) % model_save_step == 0:
                torch.save(self.G.state_dict(),
                           os.path.join(self.model_save_path, '{}_G.pth'.format(step + 240001)))

                torch.save(self.D.state_dict(),
                           os.path.join(self.model_save_path, '{}_D.pth'.format(step + 240001)))

    def reset_grad(self):
        self.d_optimizer.zero_grad()
        self.g_optimizer.zero_grad()

    def set_requires_grad(self, nets, requires_grad=False):
        for param in nets.parameters():
            param.requires_grad_(requires_grad)