"""fdaopt: A Flower / HuggingFace Federated Learning Server."""

import torch
import flwr.server.strategy as flwr_strat
from flwr.common import ndarrays_to_parameters
from flwr.server import start_server, ServerConfig
from transformers import AutoModelForSequenceClassification

# Import custom modules
from fdaopt.training import get_weights, get_evaluate_fn
from fdaopt.parameters import load_parameters

import os

import argparse

import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------- #
#                           Strategy Selection Function                        #
# ---------------------------------------------------------------------------- #

def get_strategy_class(strategy_name: str):
    """
    Dynamically imports and returns the strategy class from `flwr.server.strategy`.

    Parameters:
    - strategy_name (str): The name of the strategy class as a string.

    Returns:
    - type: The strategy class from `flwr.server.strategy`
    """
    return getattr(flwr_strat, strategy_name)

# ---------------------------------------------------------------------------- #
#                             Main Server Process                             #
# ---------------------------------------------------------------------------- #

if __name__ == '__main__':
    
    # ------------------------ Step 1: Get CLI Parameters ------------------------ #
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=str, default="0", help="CUDA_VISIBLE_DEVICES. e.g, '0,1'")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda  # Set GPU visibility (modify if needed)
    
    # Set computation device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ------------------------ Step 2: Get Json from Kafka ------------------------ #
    
    json_name = '0.json'  # JSON file containing configuration for the server
    params = load_parameters(json_name)  # Load parameters from JSON
    
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

    # ---------------------- Step 4: Initialize Model ---------------------- #

    # Load pre-trained model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint, num_labels=num_labels
    )

    # Convert model weights to Flower parameters
    weights = get_weights(model)
    initial_parameters = ndarrays_to_parameters(weights)

    # ---------------------- Step 5: Define Federated Strategy ---------------------- #

    # Get strategy class dynamically
    FedStrat = get_strategy_class(strategy_name)

    # Define FL strategy
    strategy = FedStrat(
        fraction_fit=fraction_fit,
        fraction_evaluate=0.0,  # Modify if evaluation is needed
        initial_parameters=initial_parameters,
        evaluate_fn=get_evaluate_fn(model, device, model_checkpoint, ds_path, ds_name),
    )

    # ---------------------- Step 6: Start Flower Server ---------------------- #

    config = ServerConfig(num_rounds=total_rounds)

    start_server(
        server_address=server_address,
        config=config,
        strategy=strategy
    )
