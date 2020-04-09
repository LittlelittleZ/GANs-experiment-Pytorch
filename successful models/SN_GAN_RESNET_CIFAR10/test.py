from solver import GANModel
import torch
import torchvision
from PIL import Image
import yaml
import numpy as np
def make_batches(size, batch_size):
    '''Returns a list of batch indices (tuples of indices).
    '''
    return [(i, min(size, i + batch_size)) for i in range(0, size, batch_size)]

if __name__ == "__main__":
    model_dir = "./utils/"
    config = yaml.load(open('./resnet.yml'))
    model = GANModel(config)
    model.G.load_state_dict(torch.load(model_dir + "generator.pkl"))
    model.G.to('cuda:1')
    model.G.eval()  
    #model.D.load_state_dict(torch.load(model_dir + "10_D.pth"))
    z_fixed = torch.randn(model.batch_size, model.z_dim).to('cuda:1')
    fake = model.G(z_fixed)
    torchvision.utils.save_image(fake, 'f1.png', nrow=8 , normalize=True)
    noise = np.random.randn(50000, 128).astype(np.float32)
    batches = make_batches(50000, 25)
    h = []
    for batch_idx, (batch_start, batch_end) in enumerate(batches):
        noise_batch = noise[batch_start:batch_end]
        noise_batch = torch.from_numpy(noise_batch).to('cuda:1')
        out = model.G(noise_batch).detach().cpu().numpy()
        out = np.multiply(np.add(np.multiply(out, 0.5), 0.5), 255).astype('int32')
        #out = out[sli]
        h.append(out)
    h = np.vstack(h)
    h = np.transpose(h, (0, 2, 3, 1))
    im = Image.fromarray(np.uint8(h[0]))
    im.save("./{}.png".format('test'))
    np.save('lda300_1', h)
