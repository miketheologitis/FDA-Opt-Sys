"""fdaopt: A Flower / HuggingFace Federated Learning Server."""

import torch
import os
import argparse
import threading
import json
from types import SimpleNamespace
import flwr.server.strategy as flwr_strats
import fdaopt.fda_strategies as fda_strats
from flwr.common import ndarrays_to_parameters
from flwr.server import start_server, ServerConfig
from transformers import AutoModelForSequenceClassification

# Import custom modules
from fdaopt.training import get_weights, get_evaluate_fn
from fdaopt.networking import start_variance_monitoring_loop, create_pull_socket

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CHECKPOINTS_DIR = SCRIPT_DIR.parent / "logs" / "checkpoints"

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')

# ---------------------------------------------------------------------------- #
#                           Strategy Selection Function                        #
# ---------------------------------------------------------------------------- #

def get_flwr_strategy_class(strat_name: str):
    """ Get the Flower strategy class based on the strategy name.
    Args:
        strat_name: the name of the strategy to be used.

    Returns:
        The corresponding Flower strategy class.
    """
    return getattr(flwr_strats, strat_name)

def get_fda_strategy_class(strat_name: str):
    """ Get the FDA strategy class based on the strategy name.
    Args:
        strat_name: the name of the strategy to be used.
    Returns:
        The corresponding FDA strategy class.
    """
    strat_name = strat_name.replace('Fed', 'Fda')  # safety
    return getattr(fda_strats, strat_name)

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
    local_epochs = params['training']['local_epochs']

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
    
    # Get strategy class dynamically
    if fda:
        Strat = get_fda_strategy_class(strategy_name)

        # Initialize the threshold for FDA-Opt as a SimpleNamespace container
        threshold = SimpleNamespace(value=float('-inf'))

        # Define the on_fit_config_fn to pass the threshold to clients
        # This function is called before each `fit` round and allows us to pass custom parameters
        # such as the round's threshold value to the clients.
        on_fit_config_fn = lambda _: {'threshold': str(threshold.value)}
        
        # `key_metrics` is List[Tuple[int, Metrics]]
        # This function is called after each `fit` round to aggregate the metrics from clients.
        # The clients with FDA-Opt, currently, return the `epochs_completed` -- the number of epochs they completed.
        # Thus, we can use the first client's metrics as the aggregated metrics (same for all clients) so that
        # the server knows how many epochs were completed in the round locally.
        fit_metrics_aggregation_fn = lambda key_metrics: key_metrics[0][1] if key_metrics else {}
        
        # Define FL strategy
        strategy = Strat(
            local_epochs=local_epochs,
            threshold=threshold,
            fraction_fit=fraction_fit,
            fraction_evaluate=0.0,  # Modify if evaluation is needed
            initial_parameters=initial_parameters,
            evaluate_fn=get_evaluate_fn(model, device, model_checkpoint, ds_path, ds_name),
            on_fit_config_fn=on_fit_config_fn,  # to pass threshold to clients before each fit round
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn  # to aggregate clients' metrics after each fit round
        )
        
        # ---------------------- Step 6: Create Pull socket that Accepts States ---------------------- #
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
        
        
        
    else:
        Strat = get_flwr_strategy_class(strategy_name)

        # Define FL strategy
        strategy = Strat(
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
        
    job_id = os.environ.get("JOB_ID", "unknown")
    model_checkpoint_str = model_checkpoint.replace('/', '-')
    model_checkpoint_save = f"{model_checkpoint_str}-{job_id}.pth" 
    save_path = CHECKPOINTS_DIR / model_checkpoint_save
    torch.save(model.state_dict(), save_path)
    logging.info(f"[Server-{job_id}] Finished training!")
    logging.info(f"[Server-{job_id}] Saved model!")
    logging.info(f"[Server-{job_id}] Exiting...!")
    