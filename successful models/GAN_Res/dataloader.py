import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

def getloader(batch_size, root, dataset='cifar_ten'):
    def input_cifar10(batch_size, root='../cifar-10'):
        transform = transforms.Compose(
        [   transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Lambda(lambda x: x + 1./128 * torch.rand(x.size())),])

        trainset = torchvision.datasets.CIFAR10(root=root, transform=transform,download=True)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=4)
        
        return trainloader

    def input_cifar100(batch_size, root='../cifar-100'):
        transform = transforms.Compose(
        [   transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Lambda(lambda x: x + 1./128 * torch.rand(x.size())),])

        trainset = torchvision.datasets.CIFAR100(root=root, transform=transform,download=True)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=4)
        
        return trainloader

    def input_imagenet(batch_size,root='../imagenet'):
        transform = transforms.Compose(
        [   transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Lambda(lambda x: x + 1./128 * torch.rand(x.size())),])

        trainset = torchvision.datasets.ImageFolder(root=root,transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=4)
        return trainloader


    if 'cifar_ten' in dataset:
        return input_cifar10(batch_size, root)
    elif 'cifar_hundred' in dataset:
        return input_cifar100(batch_size, root)
    elif 'imagenet' in dataset:
        return input_imagenet(batch_size, root)
    else:
        print('error')
    