"""
id：客户端的标识
train_data：客户端的训练数据
test_data：客户端的测试数据
model：客户端使用的模型
optimizer：客户端使用的优化器
lr：客户端使用的学习率
batch_size：客户端使用的批大小
num_epochs：客户端的训练轮数

train()方法用于训练客户端模型。该方法将训练数据分批送入模型进行训练。

evaluate()方法用于评估客户端模型的性能。该方法将测试数据送入模型进行预测，并计算预测的准确率。

get_parameters()方法用于获取客户端模型的参数。

set_parameters()方法用于设置客户端模型的参数。
"""

import torch

class Client(object):

    def __init__(self, id, train_data, test_data, model, optimizer, lr, batch_size, num_epochs):
        self.id = id
        self.train_data = train_data
        self.test_data = test_data
        self.model = model
        self.optimizer = optimizer
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def train(self):
        for epoch in range(self.num_epochs):
            for batch_idx, (data, target) in enumerate(self.train_data):
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = torch.nn.functional.cross_entropy(output, target)
                loss.backward()
                self.optimizer.step()

    def evaluate(self):
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.test_data:
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        return correct / total

    def get_parameters(self):
        return self.model.state_dict()

    def set_parameters(self, parameters):
        self.model.load_state_dict(parameters)


"""
# 定义客户端
client1 = Client(1, train_data1, test_data1, model, optimizer, lr, batch_size, num_epochs)
client2 = Client(2, train_data2, test_data2, model, optimizer, lr, batch_size, num_epochs)

# 训练客户端
client1.train()
client2.train()

# 评估客户端
print("client1 accuracy:", client1.evaluate())
print("client2 accuracy:", client2.evaluate())

"""