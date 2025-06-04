"""fdaopt: A Flower / HuggingFace Federated Learning Client."""

import os
import torch
import argparse
import json

from flwr.client import NumPyClient, start_client
from flwr.common import Context
from transformers import AutoModelForSequenceClassification
from torch.optim import SGD

# Import custom modules
from fdaopt.training import get_weights, set_weights, train, train_fda
from fdaopt.data import load_data
from fdaopt.networking import create_push_socket, create_pull_socket
from fdaopt.sketch import AmsSketch

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')

# ---------------------------------------------------------------------------- #
#                              Flower Client Class                             #
# ---------------------------------------------------------------------------- #

class FlowerClient(NumPyClient):
    """Custom Flower Client for Federated Learning using Hugging Face models."""
    
    def __init__(self, model, optimizer, device, trainloader, local_epochs, ip,
                 port, server_ip_pull_socket, server_port_pull_socket, fda, client_id):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.trainloader = trainloader
        self.local_epochs = local_epochs
        self.model.to(self.device)
        self.ip = ip
        self.port = port
        self.server_ip_pull_socket = server_ip_pull_socket
        self.server_port_pull_socket = server_port_pull_socket
        self.fda = fda
        self.client_id = client_id
        
    def fit(self, parameters, config):
        """Perform local training and return updated model weights."""
        
        set_weights(self.model, parameters)
        
        if self.fda:
            threshold = float(config['threshold'])
            
            # Create sockets for side-channel communication for local-states / variance monitoring
            logging.info(f"[Client - FDA-Opt] Round's Threshold: {threshold:.4f}!")

            # Create push and pull sockets for variance approximation communication
            push_to_server_socket = create_push_socket(self.server_ip_pull_socket, self.server_port_pull_socket)
            pull_variance_approx_socket = create_pull_socket(self.ip, self.port)

            sketch = AmsSketch()

            # Start local FDA-Opt training round
            epochs_completed = train_fda(
                self.model, 
                self.optimizer, 
                self.trainloader, 
                self.device, 
                self.local_epochs, 
                self.client_id,
                threshold,
                push_to_server_socket, 
                pull_variance_approx_socket, 
                sketch
            )
            
            return get_weights(self.model), len(self.trainloader), {'epochs_completed': epochs_completed}

        # Start local FedOpt training round
        train(
            self.model, 
            self.optimizer, 
            self.trainloader, 
            self.device, 
            self.local_epochs
        )
        
        return get_weights(self.model), len(self.trainloader), {}

# ---------------------------------------------------------------------------- #
#                              Client Function                                 #
# ---------------------------------------------------------------------------- #

def client_func(
    context: Context,
    total_clients: int, 
    client_id: int,
    device: str,
    model_checkpoint: str,
    num_labels: int,
    ds_path: str,
    ds_name: str,
    dirichlet_alpha: float,
    data_path: str,
    client_lr: float,
    local_epochs: int,
    ip: str,
    port: int,
    server_ip_pull_socket: str,
    server_port_pull_socket: int,
    fda: bool
):
    """
    Function to initialize and return a Flower client.

    Parameters:
    - context (Context): Flower context.
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
    
    #print(f"Client {client_id} inside `client_fn`.")

    # Load dataset partition for this client
    trainloader = load_data(client_id, total_clients, model_checkpoint, ds_path, ds_name, dirichlet_alpha, data_path)

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
        local_epochs=local_epochs,
        ip=ip,
        port=port,
        server_ip_pull_socket=server_ip_pull_socket,
        server_port_pull_socket=server_port_pull_socket,
        fda=fda,
        client_id=client_id
    ).to_client()

# ---------------------------------------------------------------------------- #
#                             Main Client Process                              #
# ---------------------------------------------------------------------------- #

if __name__ == '__main__':
    
    # ------------------------ Step 1: Get CLI Parameters ------------------------ #
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--client_id', type=int, required=True, help="The client ID.")
    parser.add_argument('--cuda', type=str, default="0", help="CUDA_VISIBLE_DEVICES.")
    parser.add_argument('--local_json', type=str, required=True, help="The client reads json locally.")
    args = parser.parse_args()
    client_id = args.client_id
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda  # Set GPU visibility (modify if needed)
    
    # Set computation device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ------------------------ Step 2: Get Json from Kafka ------------------------ #
    
    # Load parameters from JSON
    with open(args.local_json) as f:
        params = json.load(f)
    
    # ----------------------- Step 3: Load Configuration JSON ---------------------- #
    
    # Extract server details
    server_address = f"{params['server']['network']['ip']}:{params['server']['network']['port']}"
    
    # Extract model and dataset configurations
    model_checkpoint = params['model']['checkpoint']
    num_labels = params['model']['num_labels']
    ds_path = params['dataset']['path']
    ds_name = params['dataset']['name']
    dirichlet_alpha = params['dataset']['dirichlet_alpha']
    
    data_path = ""
    for client_info in  params['clients']['network']:
        if client_info["id"] == client_id:
            data_path = client_info["data_path"]
            break
    
    # Extract client-specific training parameters
    client_lr = params['clients']['lr']
    local_epochs = params['training']['local_epochs']
    total_clients = params['training']['num_clients']
    
    # Networking stuff...
    ip = params['clients']['network'][client_id]['ip']
    port = params['clients']['network'][client_id]['port']
    server_ip_pull_socket = params['server']['network']['ip_pull_socket']
    server_port_pull_socket = params['server']['network']['port_pull_socket']
    
    fda = params['server']['strategy']['fda']
    
    # ----------------------- Step 4: Define Client Function ---------------------- #

    client_fn = lambda context: client_func(
        context=context,
        total_clients=total_clients, 
        client_id=client_id,
        device=device,
        model_checkpoint=model_checkpoint,
        num_labels=num_labels,
        ds_path=ds_path,
        ds_name=ds_name,
        data_path=data_path,
        dirichlet_alpha=dirichlet_alpha,
        client_lr=client_lr,
        local_epochs=local_epochs,
        ip=ip,
        port=port,
        server_ip_pull_socket=server_ip_pull_socket,
        server_port_pull_socket=server_port_pull_socket,
        fda=fda
    )
    
    # ------------------------- Step 5: Start Flower Client ---------------------- #

    start_client(
        server_address=server_address,
        client_fn=client_fn,
    )
