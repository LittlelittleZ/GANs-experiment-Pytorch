import torch
import torchvision
import torchvision.transforms as transforms
T = transforms

from model import Generator, Discriminator
import os
import datetime
from torch import autograd

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def calc_gradient_penalty(netD, real_data, fake_data):
    # print "real_data: ", real_data.size(), fake_data.size()
    BATCH_SIZE = real_data.size(0)
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(BATCH_SIZE, real_data.nelement()//BATCH_SIZE).contiguous().view(BATCH_SIZE, 3, 32, 32)
    alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)


    interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
    return gradient_penalty

def compute_grad2(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg

if __name__ == "__main__":
    # random seed
    torch.manual_seed(0)
    transform = []
    transform.append(T.RandomHorizontalFlip())
    transform.append(T.CenterCrop(32))
    transform.append(T.Resize(32))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transforms.Lambda(lambda x: x + 1./128 * torch.rand(x.size()))
    transform = T.Compose(transform)


    dataset = torchvision.datasets.CIFAR10(root='./cifar10', transform=transform, download=True)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                    batch_size=64,
                                    shuffle=True,
                                    drop_last=True,
                                    num_workers=8)

    G = Generator().cuda()
    D = Discriminator().cuda()
    D.load_state_dict(torch.load('./checkpoint/25000_D.pth'))
    G.load_state_dict(torch.load('./checkpoint/25000_G.pth'))
    optimizer_G = torch.optim.RMSprop(G.parameters(), lr=1e-4, alpha=0.99, eps=1e-8)
    optimizer_D = torch.optim.RMSprop(D.parameters(), lr=1e-4, alpha=0.99, eps=1e-8)
    print("start...")
    dataiter = iter(data_loader)
    for idx in range(2000000):
        time_start = datetime.datetime.now()
        try:
            data, label = next(dataiter)
        except:
            dataiter = iter(data_loader)
            data, label = next(dataiter)

        data = data.cuda()
        
        if idx % 5 == 0:
            z = torch.randn(64, 128).cuda()
            fake = G(z)
            d_fake4g = D(fake)
            optimizer_G.zero_grad()
            loss_g = torch.mean(d_fake4g)
            loss_g.backward()
            optimizer_G.step()

        z = torch.randn(64, 128).cuda()
        fake = G(z)
        d_fake4d = D(fake.detach())
        data.requires_grad_()
        d_real4d = D(data)
        optimizer_D.zero_grad()
        #gp = calc_gradient_penalty(D, data, fake)
        gp = 10 * compute_grad2(d_real4d, data).mean()
        loss_d = torch.mean(d_real4d.mean() - d_fake4d.mean()) + gp
        loss_d.backward()
        optimizer_D.step()


        #save model
        if (idx+1)%25000 == 0:
            #
            
            torch.save(G.state_dict(), "./checkpoint/{}_G.pth".format(idx+25001))
            torch.save(D.state_dict(), "./checkpoint/{}_D.pth".format(idx+25001))
            #

        if (idx+1)%5000 == 0:
            torchvision.utils.save_image(fake, './samples/f{}.png'.format(idx+25001), nrow=8 , normalize=True)
        time_end = datetime.datetime.now()
        print('[%d/%d] D(x): %.4f D(G(z)): %.4f/ %.4f'% (idx+1, 2000000, d_real4d.mean().item(),\
             d_fake4d.mean().item(), d_fake4g.mean().item()))
        #print('alpha: ', model.alpha)
        print("remains {:.4f} minutes...".format((time_end - time_start).total_seconds() / 60. * (500000 - idx)))
