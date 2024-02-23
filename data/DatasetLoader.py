import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import hashlib

from torch.utils.data import DataLoader, Subset
from torch.distributions.dirichlet import Dirichlet

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
    """
    def get_dataloader(self, train=True, client_id=None, num_clients=0):
        dataset = self.load_dataset(train=train)

        # 如果提供了client_id，可以使用它来为每个客户端创建相同的数据集
        if client_id is not None:
            # 将client_id转换为字符串后再进行哈希
            client_id_str = str(client_id)
            hash_value = int(hashlib.sha256(client_id_str.encode('utf-8')).hexdigest(), 16)
            hash_value = hash_value % (2 ** 32)  # 使用模运算限制哈希值范围
            torch.manual_seed(hash_value)
            indices = torch.randperm(len(dataset))
            dataset = torch.utils.data.Subset(dataset, indices[:len(dataset) // num_clients])

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        return dataloader
    """
    def get_dataloader(self, train=True, client_id=None, num_clients=0):
        dataset = self.load_dataset(train=train)

        if client_id is not None:
            client_id_str = str(client_id)
            hash_value = int(hashlib.sha256(client_id_str.encode('utf-8')).hexdigest(), 16)
            hash_value = hash_value % (2 ** 32)
            torch.manual_seed(hash_value)

            # 使用 Dirichlet 分布生成每个类别的比例，num_clients 表示客户端数量
            alpha = torch.ones(10)  # 10个类别
            dirichlet_dist = Dirichlet(alpha)
            class_proportions = dirichlet_dist.sample(torch.Size([num_clients]))

            # 将数据集分为 num_clients 份，每份按照对应的类别比例划分
            subset_indices = []
            for proportions in class_proportions:
                num_samples = int(len(dataset) // num_clients)
                indices = torch.multinomial(proportions, num_samples, replacement=True)
                subset_indices.extend(indices.tolist())

            dataset = Subset(dataset, subset_indices)

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        return dataloader