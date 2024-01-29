import torch
import torch.nn as nn
import copy

class SGLDProcess:
    def __init__(self, clients, pure_data_loader, num_samples, lr=0.01, noise_scale=0.1):
        self.clients = clients
        self.pure_data_loader = pure_data_loader
        self.num_samples = num_samples
        self.lr = lr
        self.noise_scale = noise_scale
        self.original_params = {}  # 存储原始参数的字典
        self.sgld_params = {}  # 存储 SGLD 参数的字典
        self.clients_sgld = None

    # def sgld_step(self):
    #     for client in self.clients_sgld:
    #         for param in client.get_local_model().parameters():
    #             print(f"Gradient for {param}: {param.grad}")
    #             noise = torch.normal(0, self.noise_scale * torch.sqrt(torch.tensor(self.lr)), size=param.size())
    #             param.data.add_(-self.lr * param.grad - noise)

    def sample(self):
        sampled_models = []
        self.clients_sgld = self.clients

        for _ in range(self.num_samples):
            # 执行一次梯度计算
            for client in self.clients_sgld:
                cloned_client = client.local_model
                cloned_client.train()
                cloned_client.zero_grad()

                for data, target in self.pure_data_loader:
                    output = cloned_client(data)
                    loss = nn.functional.nll_loss(output, target)
                    loss.backward()

                    # # 添加以下打印语句以检查梯度
                    # for param in cloned_client.parameters():
                    #     print(f"Gradient for {param}: {param.grad}")

                for param in cloned_client.parameters():
                    print(f"Gradient for {param}: {param.grad}")
                    noise = torch.normal(0, self.noise_scale * torch.sqrt(torch.tensor(self.lr)), size=param.size())
                    param.data.add_(-self.lr * param.grad - noise)
                # 在每个客户端上执行 SGLD 步骤
                # self.sgld_step()
                client.local_model = cloned_client

            # 将当前模型添加到样本中
            for client in self.clients_sgld:

                sampled_models.append(copy.deepcopy(client.get_local_model()))

        return sampled_models

    def combine_samples(self, samples):
        combined_params = {}
        for param_name, param in samples[0].state_dict().items():
            param_list = [sample.state_dict()[param_name].view(-1) for sample in samples]
            combined_params[param_name] = torch.stack(param_list, dim=0).mean(dim=0)

        return combined_params

    def startup(self):
        sgld_samples = self.sample()

        for client in self.clients:
            client_id = client.client_id
            self.original_params[client_id] = client.get_local_model().state_dict()
            self.sgld_params[client_id] = self.combine_samples(sgld_samples)

        return self

    def get_model_params(self, client_id, use_sgld=False):
        params = self.sgld_params if use_sgld else self.original_params
        return params[client_id]


"""
# 获取客户端2的采样后的参数
client_id_to_check = 1  # 请注意索引是从0开始的
sgld_params_client2 = sgld_sampler.sgld_params[client_id_to_check]

# 输出客户端2采样后的参数张量
for param_name, param_tensor in sgld_params_client2.items():
    print(f"Parameter Name: {param_name}")
    print(param_tensor)

"""


"""
# 创建SGLD采样器
sgld_sampler = SGLDProcess(global_model)

# 运行SGLD采样并收集参数
sgld_sampler.startup(train_loader, num_samples, clients)

# 打印每个客户端的ID和对应的模型参数
for client_id in range(num_clients):
    original_params = sgld_sampler.original_params[client_id]
    sgld_params = sgld_sampler.sgld_params[client_id]
    print(f"Client ID: {client_id}, Original Model Parameters: {original_params}")
    print(f"Client ID: {client_id}, SGLD Model Parameters: {sgld_params}")

# 最终打印全局模型参数
print("Global Model Parameters:", global_model.state_dict())
"""

