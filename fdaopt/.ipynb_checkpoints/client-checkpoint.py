"""fdaopt: A Flower / HuggingFace Federated Learning Client."""

import os
import torch
from flwr.client import NumPyClient, start_client
from flwr.common import Context
from transformers import AutoModelForSequenceClassification
from torch.optim import SGD

# Import custom modules
from fdaopt.training import get_weights, set_weights, train
from fdaopt.data import load_data
from fdaopt.parameters import load_parameters

import argparse

import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------- #
#                              Flower Client Class                             #
# ---------------------------------------------------------------------------- #

class FlowerClient(NumPyClient):
    """Custom Flower Client for Federated Learning using Hugging Face models."""
    
    def __init__(self, model, optimizer, device, trainloader, local_epochs):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.trainloader = trainloader
        self.local_epochs = local_epochs
        self.model.to(self.device)

    def fit(self, parameters, config):
        """Perform local training and return updated model weights."""
        set_weights(self.model, parameters)
        train(self.model, self.optimizer, self.trainloader, self.device, self.local_epochs)
        return get_weights(self.model), len(self.trainloader), {}

# ---------------------------------------------------------------------------- #
#                              Client Function                                 #
# ---------------------------------------------------------------------------- #

def client_func(
    context: Context, 
    json_name: str, 
    total_clients: int, 
    client_id: int,
    device: str,
    model_checkpoint: str,
    num_labels: int,
    ds_path: str,
    ds_name: str,
    client_lr: float,
    local_epochs: int
):
    """
    Function to initialize and return a Flower client.

    Parameters:
    - context (Context): Flower context.
    - json_name (str): JSON configuration file name.
    - total_clients (int): Total number of FL clients.
    - client_id (int): Unique identifier for this client.
    - device (str): Computation device ('cpu' or 'cuda').
    - model_checkpoint (str): Hugging Face model checkpoint.
    - num_labels (int): Number of classification labels.
    - ds_path (str): Path to dataset.
    - ds_name (str): Dataset name.
    - client_lr (float): Learning rate for optimizer.
    - local_epochs (int): Number of local training epochs.
    """
    
    print(f"Client {client_id} inside `client_fn`.")

    # Load dataset partition for this client
    trainloader = load_data(client_id, total_clients, model_checkpoint, ds_path, ds_name)

    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint, num_labels=num_labels
    )

    # Initialize optimizer
    optimizer = SGD(model.parameters(), client_lr)

    # Create and return Flower client instance
    return FlowerClient(
        model=model, 
        optimizer=optimizer,
        device=device,
        trainloader=trainloader, 
        local_epochs=local_epochs
    ).to_client()

# ---------------------------------------------------------------------------- #
#                             Main Client Process                              #
# ---------------------------------------------------------------------------- #

if __name__ == '__main__':
    
    # ------------------------ Step 1: Get CLI Parameters ------------------------ #
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--client_id', type=int, required=True, help="The client ID.")
    parser.add_argument('--cuda', type=str, default="0", help="CUDA_VISIBLE_DEVICES. e.g, '0,1'")
    args = parser.parse_args()
    client_id = args.client_id
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda  # Set GPU visibility (modify if needed)
    
    # Set computation device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ------------------------ Step 2: Get Json from Kafka ------------------------ #
    
    json_name = '0.json'  # JSON file containing configuration for this client
    params = load_parameters(json_name)  # Load parameters from JSON
    
    # ----------------------- Step 3: Load Configuration JSON ---------------------- #
    
    # Extract server details
    server_address = f"{params['server']['network']['ip']}:{params['server']['network']['port']}"
    
    # Extract model and dataset configurations
    model_checkpoint = params['model']['checkpoint']
    num_labels = params['model']['num_labels']
    ds_path = params['dataset']['path']
    ds_name = params['dataset']['name']
    
    # Extract client-specific training parameters
    client_lr = params['clients']['lr']
    local_epochs = params['training']['local_epochs']
    total_clients = params['training']['num_clients']
    
    # ----------------------- Step 4: Define Client Function ---------------------- #

    client_fn = lambda context: client_func(
        context=context,
        json_name=json_name, 
        total_clients=total_clients, 
        client_id=client_id,
        device=device,
        model_checkpoint=model_checkpoint,
        num_labels=num_labels,
        ds_path=ds_path,
        ds_name=ds_name,
        client_lr=client_lr,
        local_epochs=local_epochs
    )
    
    # ------------------------- Step 5: Start Flower Client ---------------------- #

    start_client(
        server_address=server_address,
        client_fn=client_fn,
    )
