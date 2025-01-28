"""fdaopt: A Flower / HuggingFace app."""

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from transformers import AutoModelForSequenceClassification

from fdaopt.task import get_weights, load_data, set_weights, train


# Flower client
class FlowerClient(NumPyClient):
    def __init__(self, model, trainloader, local_epochs):
        self.model = model
        self.trainloader = trainloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def fit(self, parameters, config):
        set_weights(self.model, parameters)
        train(self.model, self.trainloader, epochs=self.local_epochs, device=self.device)
        return get_weights(self.model), len(self.trainloader), {}
    

def client_fn(context: Context):

    # Get this client's dataset partition
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    model_name = context.run_config["model-name"]
    
    trainloader = load_data(partition_id, num_partitions, model_name)

    # Load model
    num_labels = context.run_config["num-labels"]
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    local_epochs = context.run_config["local-epochs"]

    # Return Client instance
    return FlowerClient(model, trainloader, local_epochs).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
