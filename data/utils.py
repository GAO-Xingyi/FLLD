import torch
from torchvision import datasets, transforms
import syft as sy

def load_federated_mnist(clients, batch_size=1, shuffle=True):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist_dataset = datasets.MNIST(root='.', train=True, download=True, transform=transform)

    data_per_client = len(mnist_dataset) // len(clients)
    client_datasets = torch.utils.data.random_split(mnist_dataset, [data_per_client]*len(clients))

    federated_datasets = [sy.BaseDataset(client_data).send(client) for client_data, client in zip(client_datasets, clients)]

    federated_loader = sy.FederatedDataLoader(
        federated_datasets,
        batch_size=batch_size,
        shuffle=shuffle
    )

    return federated_loader