import torch
import torch.nn.functional as F
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from utils import sgld_params2dict


class One_ClassSVM:
    def __init__(self):
        pass

    @staticmethod
    def runsvm(client_sgld_params, pure_client_sgld_params):
        attention_scores = {}

        client_params_tensors = sgld_params2dict(client_sgld_params)
        pure_params_tensors = sgld_params2dict(pure_client_sgld_params)

        for param_name in client_params_tensors.keys():
            client_param = client_params_tensors[param_name].flatten().numpy()
            pure_client_param = pure_params_tensors[param_name].flatten().numpy()

            # 使用 One-Class SVM 进行异常检测
            clf = make_pipeline(StandardScaler(), OneClassSVM(nu=0.1))  # 设置nu参数，nu是训练数据中的异常点的比例
            clf.fit(pure_client_param.reshape(-1, 1))

            # 预测异常分数
            anomaly_score = clf.decision_function(client_param.reshape(-1, 1))

            print(f'{param_name} origin anomaly score: {anomaly_score}')

            # 使用 sigmoid 激活
            attention_score = torch.sigmoid(torch.tensor(anomaly_score))

            print(f'{param_name} anomaly score: {attention_score.tolist()}')

            attention_scores[param_name] = attention_score

        return attention_scores


"""
# 示例用法
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
