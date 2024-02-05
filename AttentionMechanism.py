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
        # print(client_sgld_params)
        # print(client_sgld_params[0].state_dict())
        # print(type(client_sgld_params[0].state_dict()))
        # def transformFormat(clientparams):
        #     for param in clientparams:
        #         params = param.state_dict()
        #     param_tensors = dict((key, value) for key, value in params.items())
        #     return param_tensors

        client_params_tensors = sgld_params2dict(client_sgld_params)
        pure_params_tensors = sgld_params2dict(pure_client_sgld_params)
        # global_params_tensors = sgld_params2dict(global_sgld_params)

        # print(client_params_tensors)
        # print(type(client_params_tensors))
        for param_name in client_params_tensors.keys():
            client_param = client_params_tensors[param_name]
            pure_client_param = pure_params_tensors[param_name]
            # global_params = global_params_tensors[param_name]

            """
            # 点积
            # 计算注意力分数，使用点积注意力并进行缩放
            # scale_factor = client_param.numel()  # 缩放因子为参数的数量
            scale_factor = pure_client_param.numel()
            print(f'{param_name} scale factor : {scale_factor}')
            # 计算注意力分数，使用点积注意力
            dot_product = torch.sum(client_param * pure_client_param)
            print(f'{param_name} dot product : {dot_product}')
            attention_score = dot_product / scale_factor
            # attention_score = dot_product
            print(f'dot product (without sigmoid) : {param_name} attention score : {attention_score}')
            # 使用 sigmoid 激活
            attention_score = torch.sigmoid(attention_score)
            print(f'dot product score (sigmoid) : {param_name} attention score : {attention_score}')
            """

            """
            # 目前余弦是比较好的方法
            #余弦
            # 计算余弦相似度
            cosine_similarity = F.cosine_similarity(client_param.flatten(), pure_client_param.flatten(), dim=0)
            print(f'{param_name} cosine similarity: {cosine_similarity}')

            # 使用 sigmoid 激活
            attention_score = torch.sigmoid(cosine_similarity)
            print(f'cosine similarity attention score : {param_name} attention score : {attention_score}')
            """

            """       
            #马氏距离
            # 计算马氏距离的平方
            mahalanobis_distance_sq = torch.sum((client_param - pure_client_param).pow(2))
            print(mahalanobis_distance_sq)
            mahalanobis_distance_sq_global = torch.sum((global_params - pure_client_param).pow(2))
            # 计算欧氏距离的平方
            # euclidean_distance_sq = torch.sum((client_param - pure_client_param).pow(2))
            # 使用 sigmoid 激活
            # attention_score = torch.sigmoid(mahalanobis_distance_sq)
            # attention_score = mahalanobis_distance_sq / euclidean_distance_sq
            # attention_score = mahalanobis_distance_sq / mahalanobis_distance_sq_global
            print(f'mahalanobis_distance_sq_global - mahalanobis_distance_sq : {mahalanobis_distance_sq_global - mahalanobis_distance_sq}')
            attention_score = mahalanobis_distance_sq_global - mahalanobis_distance_sq /\
                              min(mahalanobis_distance_sq_global, mahalanobis_distance_sq)
            print(f'{param_name} attention score : before sigmoid {attention_score}')
            attention_score = torch.sigmoid(attention_score)
            print(f'mahalanobis distance attention score : {param_name} attention score : after sigmoid {attention_score}')
            """


            # 计算 KL 散度
            kl_divergence = F.kl_div(F.log_softmax(client_param, dim=0), F.softmax(pure_client_param, dim=0), reduction='batchmean')
            print(f'{param_name} KL Divergence: {kl_divergence}')

            # 使用 sigmoid 激活
            attention_score = torch.sigmoid(-kl_divergence)
            print(f'KL Divergence attention score : {param_name} attention score : {attention_score}')



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