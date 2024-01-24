"""
数据攻击之：随机毒化攻击

data：原始数据集
target：原始数据集的标签
epsilon：攻击强度(数据量的百分比)
"""


import random

class RandomPoisoningAttack:

    def __init__(self, data, target, epsilon):
        self.data = data
        self.target = target
        self.epsilon = epsilon

    def attack(self):
        # 计算 epsilon 对应的数据量
        poisoned_num = int(len(self.data) * self.epsilon)

        # 随机选择 poisoned_num 个数据样本
        poisoned_indices = random.sample(range(len(self.data)), poisoned_num)

        # 将这些数据样本的标签改为错误的标签
        for index in poisoned_indices:
            self.target[index] = 1 - self.target[index]

        return poisoned_indices


"""
# 定义数据集
data = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
target = [0, 0, 0, 0, 0]

# 创建攻击对象
attack = RandomPoisoningAttack(data, target, 0.4)

# 执行攻击
poisoned_indices = attack.attack()

# 打印攻击后的标签
print(target)

"""