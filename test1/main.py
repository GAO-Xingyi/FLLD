import sys
sys.path.append("..")

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from client import Client
from FederatedCoordinator import FederatedCoordinator
from net.MNISTNet import MNISTNet  # 请替换成你的网络模块

# 定义一些超参数
num_clients = 5
num_epochs = 3

# 加载 MNIST 数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 创建全局模型
global_model = MNISTNet()

# 创建模拟客户端
clients = [Client(client_id=i, local_model=MNISTNet(), train_loader=train_loader, test_loader=train_loader) for i in range(num_clients)]

# 创建联邦学习协调器
coordinator = FederatedCoordinator(global_model, clients)

# 进行联邦学习
for epoch in range(num_epochs):
    coordinator.federated_learning()

    # 打印每个客户端的ID和对应模型参数
    for client in clients:
        print(f"Client ID: {client.client_id}, Model Parameters: {client.get_local_model().state_dict()}")

# 最终打印全局模型参数
print("Global Model Parameters:", global_model.state_dict())
