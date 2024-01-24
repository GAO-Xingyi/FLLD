import numpy as np

class BackdoorAttack:

    def __init__(self, data, target, trigger, trigger_label):
        self.data = data
        self.target = target
        self.trigger = trigger
        self.trigger_label = trigger_label

    def attack(self):
        # 找到触发器所在的样本
        trigger_indices = np.where(self.data == self.trigger)[0]

        # 将这些样本的标签改为攻击者指定的标签
        for index in trigger_indices:
            self.target[index] = self.trigger_label

        return trigger_indices


"""
# 定义数据集
data = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]
target = [0, 0, 0, 0, 0]

# 创建攻击对象
attack = BackdoorAttack(data, target, [1, 1], 1)

# 执行攻击
trigger_indices = attack.attack()

# 打印攻击后的标签
print(target)

"""