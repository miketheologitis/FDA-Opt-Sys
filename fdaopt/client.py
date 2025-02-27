"""fdaopt: A Flower / HuggingFace app."""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Change to available GPU indices

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from transformers import AutoModelForSequenceClassification

from fdaopt.training import get_weights, set_weights, train
from fdaopt.data import load_data

from torch.optim import SGD


# device
# model_name
# num_labels
# self.local_epochs
# self.optimizer


# Flower client
class FlowerClient(NumPyClient):
    def __init__(self, model, optimizer, device, trainloader, local_epochs):
        self.model = model
        self.trainloader = trainloader
        self.local_epochs = local_epochs
        self.device = device
        self.model.to(self.device)
        self.optimizer = optimizer

    def fit(self, parameters, config):
        set_weights(self.model, parameters)
        train(self.model, self.optimizer, self.trainloader, self.device, self.local_epochs)
        return get_weights(self.model), len(self.trainloader), {}
    

def client_fn(context: Context):
    
    # 1. Prepare the creation of the client (class)

    # Get this client's dataset partition
    partition_id = context.node_config["partition-id"]  # From 0 to num-partitions
    num_partitions = context.node_config["num-partitions"]  # num of clients (num-supernodes)
    
    model_checkpoint = "prajjwal1/bert-tiny"  # TODO

    num_labels = 2  # TODO
    
    ds_path = "glue"
    ds_name = "mrpc"
    
    trainloader = load_data(partition_id, num_partitions, model_checkpoint, ds_path, ds_name)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint, num_labels=num_labels
    )
    
    client_lr = 1e-3
    
    optimizer = SGD(model.parameters(), client_lr)
    
    #device = 'cuda'
    device = "cuda:0"

    local_epochs = 1  # TODO
    
    
    # 2. Create the client
    
    client = FlowerClient(
        model=model, 
        optimizer=optimizer,
        device=device,
        trainloader=trainloader, 
        local_epochs=local_epochs
    ).to_client()

    # Return Client instance
    return client


# Flower ClientApp
client_app = ClientApp(
    client_fn,
)


# backend_config