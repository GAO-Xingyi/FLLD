import syft as sy
import torch
from net.Federated_EMNIST_net import Net
import federated_emnist


# 创建一个联邦学习服务器
server = sy.Server()

# 创建 5 个客户端
clients = [
    sy.VirtualWorker(server, id="client1"),
    sy.VirtualWorker(server, id="client2"),
    sy.VirtualWorker(server, id="client3"),
    sy.VirtualWorker(server, id="client4"),
    sy.VirtualWorker(server, id="client5"),
]

# 在每个客户端上实例化模型
model = Net()
model = model.share(clients)

# 加载数据集
federated_emnist = sy.FederatedDataset(clients, federated_emnist.load_data())

# 训练模型
for epoch in range(10):
    # 在每个客户端上训练模型
    for client in clients:
        client.send(model)
        client.train(model, federated_emnist)

# 保存每个模型训练完的参数
parameters = {}
for client in clients:
    parameters[client.id] = client.get(model)

# 打印参数
print(parameters)
