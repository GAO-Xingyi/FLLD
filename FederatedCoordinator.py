import sys
sys.path.append("..")
import copy
import torch
from SGLDProcess import SGLDProcess
from client import Client
from data.DatasetLoader import DatasetLoader

class FederatedCoordinator:
    def __init__(self, global_model, dataset_name, num_clients, num_epochs_pretrain=10,
                 num_epochs_client=10, num_epochs_update=5, sgld_samples=5):
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

    def average_aggregation(self, models):
        avg_state_dict = {}
        for key in models[0].keys():
            avg_state_dict[key] = torch.stack([model[key] for model in models]).mean(0)

        return avg_state_dict

    def setup_pure_client(self):
        # 实例化纯净样本机
        dataset_loader = DatasetLoader(self.dataset_name)
        train_loader = dataset_loader.get_dataloader(train=True)
        test_loader = dataset_loader.get_dataloader(train=False)

        self.pure_train_loader = train_loader
        self.pure_test_loader = test_loader

        # 创建新的 Client 实例用于训练
        pure_client_model = copy.deepcopy(self.global_model)
        pure_client = Client(client_id="PureClient", local_model=pure_client_model, train_loader=train_loader,
                             test_loader=test_loader, dataset_type=self.dataset_name,
                             num_pretrain_epochs=self.num_epochs_pretrain)
        self.pure_client = pure_client

        # 预训练纯净样本机
        self.pure_client.pretrain()

        # 保存纯净样本机的参数
        self.initial_params = self.pure_client.get_local_model().state_dict()

    def build_client(self):
        clients = []
        for i in range(self.num_clients):
            # 使用不同的数据加载器或不同的数据集
            client_loader = DatasetLoader(dataset_name=self.dataset_name)
            train_loader = client_loader.get_dataloader(train=True)
            test_loader = client_loader.get_dataloader(train=False)

            client_model = copy.deepcopy(self.pure_client)
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
            for client in self.clients:
                client.train()

            sgld_process = SGLDProcess(self.clients, self.pure_train_loader, self.sgld_samples)
            # Perform SGLD sampling and collect original and SGLD parameters
            sgld_process.startup()

            # 获取客户端2的采样后的参数
            client_id_to_check = 1  # 请注意索引是从0开始的
            sgld_params_client2 = sgld_process.sgld_params[client_id_to_check]

            # 输出客户端2采样后的参数张量
            for param_name, param_tensor in sgld_params_client2.items():
                print(f"Parameter Name: {param_name}")
                print(param_tensor)

            # Collect local models from clients
            local_models = [client.get_local_model() for client in self.clients]

            # Aggregate local models
            aggregated_model_state = self.average_aggregation([model.state_dict() for model in local_models])

            # Update global model with aggregated model
            self.global_model.load_state_dict(aggregated_model_state)

            # Share updated global model with clients
            for client in self.clients:
                client.update_local_model(self.global_model)
