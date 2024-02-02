import sys

from DataPoisoner import DataPoisoner
from utils import sgld_params2dict

sys.path.append("..")

import copy
import torch
from SGLDProcess import SGLDProcess
from client import Client
from data.DatasetLoader import DatasetLoader
from AttentionMechanism import AttentionAggregator


class FederatedCoordinator:
    def __init__(self, global_model, dataset_name, num_clients, num_epochs_pretrain=10,
                 num_epochs_client=10, num_epochs_update=5, sgld_samples=5, posion_client_id=0, poisoned_fraction=0):
        self.global_model = global_model
        self.dataset_name = dataset_name
        self.num_clients = num_clients
        self.num_epochs_pretrain = num_epochs_pretrain
        self.num_epochs_client = num_epochs_client
        self.num_epochs_update = num_epochs_update
        self.sgld_samples = sgld_samples
        self.pure_train_loader = None
        self.pure_test_loader = None
        self.initial_params = None
        self.clients = None
        self.pure_client = None
        self.attention_scores = {}
        self.posion_client_id = posion_client_id
        self.poisoned_fraction = poisoned_fraction

    def average_aggregation(self, models):
        avg_state_dict = {}
        for key in models[0].keys():
            avg_state_dict[key] = torch.stack([model[key] for model in models]).mean(0)

        return avg_state_dict

    def setup_pure_client(self):
        # 实例化纯净样本机
        dataset_loader = DatasetLoader(self.dataset_name)
        train_loader = dataset_loader.get_dataloader(train=True, client_id="PureClient", num_clients=self.num_clients)
        test_loader = dataset_loader.get_dataloader(train=False, client_id="PureClient", num_clients=self.num_clients)

        self.pure_train_loader = train_loader
        self.pure_test_loader = test_loader

        # 创建新的 Client 实例用于训练
        pure_client_model = copy.deepcopy(self.global_model)
        pure_client = Client(client_id="PureClient", local_model=pure_client_model, train_loader=train_loader,
                             test_loader=test_loader, dataset_type=self.dataset_name,
                             num_epochs=self.num_epochs_client,
                             num_pretrain_epochs=self.num_epochs_pretrain)
        self.pure_client = pure_client
        self.global_model = self.pure_client.local_model

        # 预训练纯净样本机
        self.pure_client.pretrain()

        # 保存纯净样本机的参数
        self.initial_params = self.pure_client.get_local_model().state_dict()

        # print("cure machine", self.pure_client.local_model)

    def build_client(self):
        clients = []
        for i in range(self.num_clients):
            # 使用不同的数据加载器或不同的数据集
            client_loader = DatasetLoader(dataset_name=self.dataset_name)
            train_loader = client_loader.get_dataloader(train=True, client_id=i, num_clients=self.num_clients)
            test_loader = client_loader.get_dataloader(train=False, client_id=i, num_clients=self.num_clients)

            client_model = copy.deepcopy(self.global_model)
            client = Client(client_id=i, local_model=client_model, train_loader=train_loader,
                            test_loader=test_loader, dataset_type=self.dataset_name,
                            num_epochs=self.num_epochs_client)

            clients.append(client)
        self.clients = clients
        return self.clients

    def federated_learning(self):
        # # 在每个 epoch 之前，加载一份纯净数据实例化纯净样本机
        # self.setup_pure_client()

        for epoch in range(self.num_epochs_update):
            # 外部循环 epoch 决定模型更新次数（进行几次参数更新与参数合并）

            # Each client trains its local model
            poisoner = DataPoisoner(self.poisoned_fraction)
            for client in self.clients:
                if client.client_id is self.posion_client_id:
                    poisoned_train_data_loader = poisoner.poison_data(client.train_loader)
                    # poisoned_test_data_loader = poisoner.poison_data(client.test_loader)
                    client.train_loader = poisoned_train_data_loader
                    # client.test_loader = poisoned_test_data_loader
                    client.train()
                    print(f'{client.client_id} is poison')
                else:
                    client.train()
                    print(f'{client.client_id} is normal')



            self.pure_client.train()

            sgld_process = SGLDProcess(self.clients, self.pure_train_loader, self.sgld_samples)
            # Perform SGLD sampling and collect original and SGLD parameters
            sgld_process.startup()

            # self.pure_client = [self.pure_client]
            pure_sample = SGLDProcess([self.pure_client], self.pure_train_loader, self.sgld_samples)
            pure_sample.startup()

            global_sample = SGLDProcess(None, self.pure_train_loader, self.sgld_samples, global_model=self.global_model)
            global_sample.startup4model()

            attention_aggregator = AttentionAggregator()
            pure_client_sgld_params = pure_sample.sgld_params["PureClient"]
            for client in self.clients:
                client_sgld_params = sgld_process.sgld_params[client.client_id]
                # print(client_sgld_params)
                # print(type(client_sgld_params))
                ##这里两个客户端的参数是一样，明天检查一下sgldprocess的sample里面是不是有点问题
                self.attention_scores[client.client_id] = attention_aggregator.calculate_attention_scores(
                    client_sgld_params,
                    pure_client_sgld_params,
                    global_sample.sgld_params)

            print(self.attention_scores)

            ## attention 机制做好了，明天要评估在毒化攻击下Attention分数是否可以检测到


            # pure_params = pure_sample.sgld_params
            # print("pure params", torch.Tensor([pure_params[key] for key in pure_params]).size())
            """
            ##这里实验一下，发现可以跑通，今天可以休息了
            # 获取客户端2的采样后的参数
            client_id_to_check = 1  # 请注意索引是从0开始的
            sgld_params_client2 = sgld_process.sgld_params[client_id_to_check]
            sgld_params_client2 = sgld_params2dict(sgld_params_client2)
            # 输出客户端2采样后的参数张量
            for param_name, param_tensor in sgld_params_client2.items():
                print(f"Parameter Name: {param_name}")
                print(param_tensor)
            """
            print("attention_scores:", self.attention_scores)


"""
            # Collect local models from clients
            local_models = [client.get_local_model() for client in self.clients]

            # Aggregate local models
            aggregated_model_state = self.average_aggregation([model.state_dict() for model in local_models])

            # Update global model with aggregated model
            self.global_model.load_state_dict(aggregated_model_state)

            # Share updated global model with clients
            for client in self.clients:
                client.update_local_model(self.global_model)
"""
