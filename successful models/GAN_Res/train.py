from solver import GANModel
from dataloader import getloader
import torch
import torchvision
import datetime
import yaml
import os
from torchvision import utils
import copy
import time

def log_image(z_fixed,G_test):
    fake = G_test(z_fixed)
    return fake


if __name__ == "__main__":
    def adjust_learning_rate(optimizer, epoch):
        for param_group in optimizer.param_groups:
            param_group['lr'] -= 2e-4/60000
        return param_group['lr']
    torch.manual_seed(0)
    config = yaml.load(open('./resnet.yml'))
    g_batch_size = config['g_batch_size']
    d_batch_size = config['d_batch_size']
    lr = 0
    dataset = getloader(batch_size=d_batch_size, root='../cifar-10')

    model = GANModel(config)

    # z_fixed = torch.randn(model.batch_size*model.num, model.z_dim).to(model.device)

    batch_num = len(dataset)
    model_dir = {"checkpoint":"./checkpoint", "samples":"./samples", "tb":"./tensorboard"}
    for dir_ in model_dir:
        if not os.path.exists(model_dir[dir_]):
            os.mkdir(model_dir[dir_])
            
    max_step = config['max_step']
    critic = config['critic']

    print("start...")
    for epoch in range(max_step):
        time_start = datetime.datetime.now()
        for idx, (data, label) in enumerate(dataset):
            if idx % critic == 0:
                model.set_input(g_batch_size, data)
                D_G_z2 = model.optimize_parametersG()
            model.set_input(d_batch_size, data)
            D_x, D_G_z1 = model.optimize_parametersD()           
            if idx % 5 == 4:
                _ =  adjust_learning_rate(model.optimizer_G, epoch)
                lr = adjust_learning_rate(model.optimizer_D, epoch)
                model.lr = lr
            # torchvision.utils.save_image(model.fake, model_dir['samples'] + '/f_temp.jpg', nrow=8, normalize=True)
        time_end = datetime.datetime.now()

        samples = model.fake
        samples = samples.mul(0.5).add(0.5)
        samples = samples.data.cpu()[:64]
        grid = utils.make_grid(samples)
        utils.save_image(grid, model_dir['samples'] + '/f{}.png'.format(epoch), nrow=g_batch_size , normalize=True)
        print('[%d/%d] D(x): %.4f D(G(z)): %.4f/ %.4f'% (epoch, 350, D_x, D_G_z1, D_G_z2))
        # print('alpha: ', model.alpha)
        print("{:.4f} minutes...".format((time_end - time_start).seconds / 60.))
        
        if epoch % 25 == 25 - 1:
            torch.save(model.G.state_dict(), model_dir['checkpoint'] + "/{}_G.pth".format(epoch+1))
            torch.save(model.D.state_dict(), model_dir['checkpoint'] +"/{}_D.pth".format(epoch + 1))
