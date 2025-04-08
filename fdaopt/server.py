"""fdaopt: A Flower / HuggingFace Federated Learning Server."""

from fdaopt.fda_strategies import FdaAdam

import torch
import os
import argparse
import threading
import json
import flwr.server.strategy as flwr_strats
import fdaopt.fda_strategies as fda_strats
from flwr.common import ndarrays_to_parameters
from flwr.server import start_server, ServerConfig
from transformers import AutoModelForSequenceClassification

# Import custom modules
from fdaopt.training import get_weights, get_evaluate_fn
from fdaopt.networking import start_variance_monitoring_loop, create_pull_socket

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')

# ---------------------------------------------------------------------------- #
#                           Strategy Selection Function                        #
# ---------------------------------------------------------------------------- #

def get_flwr_strategy_class(strategy_name: str):
    return getattr(flwr_strats, strategy_name) 

def get_fda_strategy_class(strategy_name: str):
    strategy_name = strategy_name.replace('Fed', 'Fda')  # safety
    return getattr(fda_strats, strategy_name) 

# ---------------------------------------------------------------------------- #
#                             Main Server Process                             #
# ---------------------------------------------------------------------------- #

if __name__ == '__main__':
    
    # ------------------------ Step 1: Get CLI Parameters ------------------------ #
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=str, default="0", help="CUDA_VISIBLE_DEVICES.")
    parser.add_argument('--local_json', type=str, required=True, help="The client reads json locally.")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda  # Set GPU visibility (modify if needed)
    
    # Set computation device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ------------------------ Step 2: Get Json ------------------------ #
    
    # Load parameters from JSON
    with open(args.local_json) as f:
        params = json.load(f)
    
    # ------------------------ Step 3: Load Configuration ----------------------- #

    # Extract training settings
    total_rounds = params['training']['total_rounds']
    num_clients = params['training']['num_clients']
    clients_per_round = params['training']['clients_per_round']

    # Compute client fraction for FL strategy
    fraction_fit = clients_per_round / num_clients

    # Extract model and dataset configurations
    model_checkpoint = params['model']['checkpoint']
    num_labels = params['model']['num_labels']
    ds_path = params['dataset']['path']
    ds_name = params['dataset']['name']

    # Extract strategy and server details
    strategy_name = params['server']['strategy']['name']
    server_address = f"{params['server']['network']['ip']}:{params['server']['network']['port']}"
    fda = params['server']['strategy']['fda']
    clients_network = params['clients']['network']  # list
    

    # ---------------------- Step 4: Initialize Model ---------------------- #

    # Load pre-trained model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint, num_labels=num_labels
    )

    # Convert model weights to Flower parameters
    weights = get_weights(model)
    initial_parameters = ndarrays_to_parameters(weights)

    # ---------------------- Step 5: Define Federated Strategy ---------------------- #

    # if not fda:
    
    # Get strategy class dynamically
    if fda:
        Strat = get_fda_strategy_class(strategy_name)
    else:
        Strat = get_flwr_strategy_class(strategy_name)

    # Define FL strategy
    strategy = Strat(
        fraction_fit=fraction_fit,
        fraction_evaluate=0.0,  # Modify if evaluation is needed
        initial_parameters=initial_parameters,
        evaluate_fn=get_evaluate_fn(model, device, model_checkpoint, ds_path, ds_name),
    )
    
    # ---------------------- Step 6: Create Pull socket that Accepts States ---------------------- #
    if fda:
        ip_pull_socket = params['server']['network']['ip_pull_socket']
        port_pull_socket = params['server']['network']['port_pull_socket']
        pull_socket = create_pull_socket(ip_pull_socket, port_pull_socket)
        start_variance_monitoring_loop(pull_socket, clients_per_round, clients_network)  # Threads and stuff...

    # ---------------------- Step 7: Start Flower Server ---------------------- #

    config = ServerConfig(num_rounds=total_rounds)

    start_server(
        server_address=server_address,
        config=config,
        strategy=strategy
    )
