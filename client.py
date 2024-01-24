import numpy as np


class Client:
    def __init__(self, client_id, local_data):
        self.client_id = client_id
        self.local_data = local_data
        self.local_model = None
        self.local_gradients = None

    def download_model(self, global_model):
        # 下载全局模型到本地
        self.local_model = global_model.copy()

    def upload_gradients(self):
        # 模拟本地训练并计算梯度
        # 在实际场景中，你应该使用你的机器学习框架在本地数据上进行训练并计算梯度。
        self.local_gradients = self.compute_gradients()

    def poison_data(self, attack_percentage):
        # 模拟数据毒化，修改一定比例的本地数据
        num_poisoned_samples = int(len(self.local_data) * attack_percentage)
        indices_to_poison = np.random.choice(len(self.local_data), num_poisoned_samples, replace=False)

        for idx in indices_to_poison:
            # 修改数据（例如，更改标签或特征）以引入毒素
            # 在实际场景中，你应该精心设计毒素，以影响全局模型。
            self.local_data[idx] = self.modify_data(self.local_data[idx])

    def compute_gradients(self):
        # 梯度计算的占位符
        # 在实际场景中，你应该使用你的机器学习框架在本地数据上进行训练并计算梯度。
        # 这里，我们简单地返回随机梯度作为占位符。
        return np.random.rand()

    def modify_data(self, data_point):
        # 数据修改的占位符（例如，用于数据毒化）
        # 在实际场景中，你应该精心设计修改。
        # 这里，我们简单地返回相同的数据点作为占位符。
        return data_point
