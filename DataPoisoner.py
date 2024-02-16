import torch
import torch.nn.functional as F


class DataPoisoner:
    def __init__(self, poisoned_fraction=0.2):
        self.poisoned_fraction = poisoned_fraction

    def poison_data(self, client_data_loader):
        # 获取原始数据集和索引
        original_dataset = client_data_loader.dataset.dataset
        indices = client_data_loader.dataset.indices

        # 计算要毒化的样本数量
        num_poisoned_samples = int(self.poisoned_fraction * len(indices))

        # 从样本中随机选择要毒化的索引
        poisoned_indices = torch.randperm(len(indices))[:num_poisoned_samples]

        # 存储被毒化的数据
        poisoned_data = []

        for idx in indices:
            # 获取原始数据集的元素
            data, target = original_dataset[idx]

            # 判断是否对该批次进行毒化
            if idx in poisoned_indices:
                # 对数据进行毒化操作
                data, target = self.poison_function(data)

            # 存储毒化后的数据
            poisoned_data.append((data, target))

        # 创建新的 DataLoader 对象
        poisoned_data_loader = torch.utils.data.DataLoader(
            poisoned_data, batch_size=client_data_loader.batch_size, shuffle=client_data_loader.sampler is not None
        )

        return poisoned_data_loader

    def poison_function(self, data):

        # 引入离散的异常值
        # anomaly_mask = (torch.rand_like(data) < self.anomaly_intensity).float()
        # 0， 256  -1024   0
        # anomaly_values = torch.randint(-1024, 0, size=data.size(), dtype=torch.float32)
        # data = data * (1 - anomaly_mask) + anomaly_values * anomaly_mask
        # data = anomaly_values

        # 9 5
        target = torch.randint(0, 1, size=(1,), dtype=torch.long).item()

        return data, target

    """
    def poison_function(self, data):
        # 引入明显的异常值
        anomaly_intensity = 10  # 调整异常值的强度
        anomaly_values = torch.ones_like(data) * anomaly_intensity

        # 使用明显的异常值覆盖原始数据
        data = anomaly_values

        # 毒化标签为一个极端值
        target = 1  # 调整为你认为合适的极端值

        return data, target
    """

    # def poison_function(self, data):
    #
    #     # 添加高斯噪声
    #     noise = torch.randn_like(data) * self.noise_std
    #     data = data + noise
    #
    #     return data

    # def poison_function(self, data):
    #     # 在这里，你可以根据需求定义对数据的毒化操作
    #
    #     # 动态获取原始数据的大小
    #     original_size = data.size()[1:]
    #
    #     # 使用插值确保毒化后的数据形状与原始数据相同
    #     data = F.interpolate(data.unsqueeze(0), size=original_size, mode='bilinear', align_corners=False)
    #     data = data.squeeze(0)
    #
    #     return data
