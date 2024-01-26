import torch
import torch.optim as optim
import copy

class Client:
    def __init__(self, client_id, local_model, train_loader, test_loader, dataset_type, learning_rate=0.01, num_epochs=1,
                 num_pretrain_epochs=5, initial_model_state=None):
        self.client_id = client_id
        if initial_model_state:
            local_model.load_state_dict(initial_model_state)
        self.local_model = local_model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.optimizer = optim.SGD(self.local_model.parameters(), lr=self.learning_rate)
        self.dataset_type = dataset_type
        self.num_pretrain_epochs = num_pretrain_epochs

    def pretrain(self):
        """
        用于指定epoch数进行本地模型的预训练。
        """
        self.local_model.train()
        for epoch in range(self.num_pretrain_epochs):
            print(f"客户端 {self.client_id} - 预训练 Epoch {epoch + 1}/{self.num_pretrain_epochs}")
            for batch_idx, (data, target) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                output = self.local_model(data)
                loss = torch.nn.functional.cross_entropy(output, target)
                loss.backward()
                self.optimizer.step()
                if batch_idx % 100 == 0:
                    print(f"  批次 {batch_idx}/{len(self.train_loader)} - 损失: {loss.item()}")

    def train(self):
        if self.dataset_type == 'MNIST':
            self.train_mnist()
        elif self.dataset_type == 'CIFAR-10':
            self.train_cifar10()
        else:
            raise ValueError("不支持的数据集类型。支持的类型: 'MNIST', 'CIFAR-10'")

    def train_mnist(self):
        self.local_model.train()
        for epoch in range(self.num_epochs):
            print(f"客户端 {self.client_id} - Epoch {epoch + 1}/{self.num_epochs}")
            for batch_idx, (data, target) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                output = self.local_model(data)
                loss = torch.nn.functional.cross_entropy(output, target)
                loss.backward()
                self.optimizer.step()
                if batch_idx % 100 == 0:
                    print(f"  批次 {batch_idx}/{len(self.train_loader)} - 损失: {loss.item()}")

    def train_cifar10(self):
        self.local_model.train()
        for epoch in range(self.num_epochs):
            print(f"客户端 {self.client_id} - Epoch {epoch + 1}/{self.num_epochs}")
            for batch_idx, (data, target) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                output = self.local_model(data)
                loss = torch.nn.functional.cross_entropy(output, target)
                loss.backward()
                self.optimizer.step()
                if batch_idx % 100 == 0:
                    print(f"  批次 {batch_idx}/{len(self.train_loader)} - 损失: {loss.item()}")

    def test(self):
        self.local_model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                output = self.local_model(data)
                test_loss += torch.nn.functional.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)
        accuracy = correct / len(self.test_loader.dataset)
        return test_loss, accuracy

    def get_local_model(self):
        return copy.deepcopy(self.local_model)

    def update_local_model(self, global_model):
        self.local_model.load_state_dict(global_model.state_dict())
