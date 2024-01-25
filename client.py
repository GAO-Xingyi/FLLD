import torch
import torch.optim as optim
import copy

class Client:
    def __init__(self, client_id, local_model, train_loader, test_loader, learning_rate=0.01):
        self.client_id = client_id
        self.local_model = local_model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.learning_rate = learning_rate
        self.optimizer = optim.SGD(self.local_model.parameters(), lr=self.learning_rate)

    #执行本地模型的训练，使用客户端自己的训练数据
    def train(self, num_epochs=1):
        self.local_model.train()
        for epoch in range(num_epochs):
            for data, target in self.train_loader:
                self.optimizer.zero_grad()
                output = self.local_model(data)
                loss = torch.nn.functional.nll_loss(output, target)
                loss.backward()
                self.optimizer.step()

    #在本地模型上执行测试，返回测试损失和准确度。
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

    #返回本地模型的深层拷贝，用于在每轮训练结束后发送给协调器。
    def get_local_model(self):
        # Return a deep copy of the local model
        return copy.deepcopy(self.local_model)

    #从全局模型接收更新，更新本地模型的权重。
    def update_local_model(self, global_model):
        # Update the local model with the global model
        self.local_model.load_state_dict(global_model.state_dict())
