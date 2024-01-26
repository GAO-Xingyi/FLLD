import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class DatasetLoader:
    def __init__(self, dataset_name, root='./data', batch_size=64, shuffle=True):
        self.dataset_name = dataset_name
        self.root = root
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transform = self.get_transform()

    def get_transform(self):
        if self.dataset_name == 'MNIST':
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        elif self.dataset_name == 'CIFAR-10':
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            raise ValueError("Unsupported dataset name. Supported names: 'MNIST', 'CIFAR-10'")

    def load_dataset(self, train=True):
        if self.dataset_name == 'MNIST':
            dataset = datasets.MNIST(root=self.root, train=train, download=True, transform=self.transform)
        elif self.dataset_name == 'CIFAR-10':
            dataset = datasets.CIFAR10(root=self.root, train=train, download=True, transform=self.transform)
        else:
            raise ValueError("Unsupported dataset name. Supported names: 'MNIST', 'CIFAR-10'")
        return dataset

    def get_dataloader(self, train=True):
        dataset = self.load_dataset(train=train)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        return dataloader
