import torch
import torch.nn.functional as F

from utils import sgld_params2dict


class AttentionAggregator:
    def __init__(self):
        pass

    # def transformFormat(self,client_sgld_params):
    #     global params
    #     for param in client_sgld_params:
    #         params = param.state_dict()
    #     param_tensors = dict((key, value) for key, value in params.items())
    #     return param_tensors
    def calculate_attention_scores(self, client_sgld_params, pure_client_sgld_params):
        global params
        attention_scores = {}
        print(client_sgld_params)
        print(client_sgld_params[0].state_dict())
        print(type(client_sgld_params[0].state_dict()))
        # def transformFormat(clientparams):
        #     for param in clientparams:
        #         params = param.state_dict()
        #     param_tensors = dict((key, value) for key, value in params.items())
        #     return param_tensors

        client_params_tensors = sgld_params2dict(client_sgld_params)
        pure_params_tensors = sgld_params2dict(pure_client_sgld_params)

        print(client_params_tensors)
        print(type(client_params_tensors))
        for param_name in client_params_tensors.keys():
            client_param = client_params_tensors[param_name]
            pure_client_param = pure_params_tensors[param_name]

            # 计算注意力分数，使用点积注意力并进行缩放
            scale_factor = client_param.numel()  # 缩放因子为参数的数量
            # 计算注意力分数，使用点积注意力
            attention_score = torch.sum(client_param * pure_client_param) / scale_factor

            # 使用 sigmoid 激活
            attention_score = torch.sigmoid(attention_score)

            attention_scores[param_name] = attention_score

        return attention_scores



"""
attention_aggregator = AttentionAggregator()

# 获取两个样本的参数
client_sgld_params = sgld_process.sgld_params[0]  # 假设第一个客户端的参数
pure_client_sgld_params = pure_sample.sgld_params[0]  # 纯净样本机的参数

# 计算注意力分数
attention_scores = attention_aggregator.calculate_attention_scores(client_sgld_params, pure_client_sgld_params)

# 打印注意力分数
for param_name, score in attention_scores.items():
    print(f"Parameter Name: {param_name}, Attention Score: {score}")
"""