import sys

sys.path.append("..")

from torchvision import transforms
from client import Client
from FederatedCoordinator import FederatedCoordinator
from net.MNISTNet import MNISTNet
from data.DatasetLoader import DatasetLoader
import logging


# 定义一些超参数
num_clients = 2      # 5  2
num_epochs = 1    #3
sgld_samples = 20   #10 后验分布采样样本数 20  5
num_epochs_update = 1   #5
num_epochs_client = 25   #10 20  5  客户端训练多少次提交参数
num_epochs_pretrain = 2  #10
posion_client_id = 1    # 0
poisoned_fraction = 0.5

# # 数据集加载器
# mnist_loader = DatasetLoader(dataset_name='MNIST')

# 创建全局模型
global_model = MNISTNet()

# 创建模拟客户端，提供 dataset_type 参数
# clients = [Client(client_id=i, local_model=MNISTNet(), train_loader=mnist_loader.get_dataloader(train=True),
#                   test_loader=mnist_loader.get_dataloader(train=False), dataset_type='MNIST',
#                   num_epochs=num_epochs_client) for i in range(num_clients)]

# 创建联邦学习协调器
server = FederatedCoordinator(global_model, 'MNIST', num_clients, num_epochs_pretrain, num_epochs_client,
                              num_epochs_update, sgld_samples, posion_client_id=posion_client_id,
                              poisoned_fraction=poisoned_fraction)

#加载一份纯净数据实例化纯净样本机
server.setup_pure_client()

#创建模拟客户端
clients = server.build_client()
pure_client = server.pure_client

# 进行联邦学习
for epoch in range(num_epochs):
    # 外部循环epoch决定模型更新次数

    logging.basicConfig(filename='./kl.log', level=logging.INFO)
    server.federated_learning()


"""
    # 打印每个客户端的ID和对应模型参数
    for client in clients:
        local_model_params = client.get_local_model().state_dict()
        # print(f"Client ID: {client.client_id}, Model Parameters: {local_model_params}")

        # 打印每个参数的大小
        for param_name, param in local_model_params.items():
            print(f"  client Parameter: {param_name}, Size: {param.size()}")

    pure_model_params = pure_client.get_local_model().state_dict()
    for param_name, param in pure_model_params.items():
        print(f"  pure_client Parameter: {param_name}, Size: {param.size()}")
"""
# # 最终打印全局模型参数
# print("Global Model Parameters:", global_model.state_dict())
