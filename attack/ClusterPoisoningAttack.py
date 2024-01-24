"""
聚类毒化攻击

data：原始数据集
target：原始数据集的标签
epsilon：攻击强度
"""

import numpy as np
from sklearn.cluster import KMeans

class ClusterPoisoningAttack:

    def __init__(self, data, target, epsilon):
        self.data = data
        self.target = target
        self.epsilon = epsilon

    def attack(self):
        # 聚类数据
        km = KMeans(n_clusters=2)
        labels = km.fit_predict(self.data)

        # 选择攻击目标
        poisoned_indices = []
        for label in range(2):
            # 选择 label 类别中 epsilon 个数据样本
            poisoned_indices.extend(random.sample(np.where(labels == label)[0], self.epsilon))

        # 将这些数据样本的标签改为错误的标签
        for index in poisoned_indices:
            self.target[index] = 1 - self.target[index]

        return poisoned_indices

"""
# 定义数据集
data = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
target = [0, 0, 0, 0, 0]

# 创建攻击对象
attack = ClusterPoisoningAttack(data, target, 0.4)

# 执行攻击
poisoned_indices = attack.attack()

# 打印攻击后的标签
print(target)

"""