import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

class LocalDataset:
    def __init__(self, dataset_name, data_dir='./data', train_batch_size=64, test_batch_size=1000):
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.transform = self.get_transform()

        # Load and split the dataset
        self.train_dataset, self.test_dataset = self.load_dataset()
        self.train_loader, self.test_loader = self.create_data_loaders()

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
            raise ValueError("Unsupported dataset. Supported datasets: 'MNIST', 'CIFAR-10'.")

    def load_dataset(self):
        if self.dataset_name == 'MNIST':
            train_dataset = datasets.MNIST(root=self.data_dir, train=True, download=True, transform=self.transform)
            test_dataset = datasets.MNIST(root=self.data_dir, train=False, download=True, transform=self.transform)
        elif self.dataset_name == 'CIFAR-10':
            train_dataset = datasets.CIFAR10(root=self.data_dir, train=True, download=True, transform=self.transform)
            test_dataset = datasets.CIFAR10(root=self.data_dir, train=False, download=True, transform=self.transform)

        return train_dataset, test_dataset

    def create_data_loaders(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True)
        test_loader = DataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=False)

        return train_loader, test_loader

# Example Usage:
mnist_data = LocalDataset(dataset_name='MNIST')
cifar10_data = LocalDataset(dataset_name='CIFAR-10')
