import torch

class FederatedCoordinator:
    def __init__(self, global_model, clients):
        self.global_model = global_model
        self.clients = clients

    #这里是默认的算法，到时候要改一下
    def average_aggregation(self, models):
        # Simple average aggregation
        avg_state_dict = {}
        for key in models[0].keys():
            avg_state_dict[key] = torch.stack([model[key] for model in models]).mean(0)

        return avg_state_dict

    def federated_learning(self, num_epochs=1):
        for epoch in range(num_epochs):
            # Each client trains its local model
            for client in self.clients:
                client.train()

            # Collect local models from clients
            local_models = [client.get_local_model() for client in self.clients]

            # Aggregate local models
            aggregated_model_state = self.average_aggregation([model.state_dict() for model in local_models])

            # Update global model with aggregated model
            self.global_model.load_state_dict(aggregated_model_state)

            # Share updated global model with clients
            for client in self.clients:
                client.update_local_model(self.global_model)
