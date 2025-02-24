"""fdaopt: A Flower / HuggingFace app."""

import warnings
from collections import OrderedDict

import torch
import transformers
from datasets.utils.logging import disable_progress_bar
from evaluate import load as load_metric
import evaluate
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset
from fdaopt.data import get_test_ds

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
disable_progress_bar()
transformers.logging.set_verbosity_error()


def train(model, optimizer, trainloader, device, epochs):
    
    # Set the model to training mode
    model.train()
    
    for _ in range(epochs):
        
        for batch in trainloader:
            
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Perform a forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            # Backward pass: compute gradients
            loss.backward()
            
            # Update model parameters
            optimizer.step()
            
            # Zero the gradients before the next backward pass
            optimizer.zero_grad()


def get_evaluate_fn(model, device, model_checkpoint, ds_path, ds_name):
    """Return an evaluation function for server-side evaluation."""
    
    # Test dataset
    test_ds = get_test_ds(model_checkpoint, ds_path, ds_name)
    
    model.to(device)

    def compute_metrics(server_round, parameters, config):
        
        set_weights(model, parameters)

        # Load the evaluation metric
        metric = evaluate.load(path=ds_path, config_name=ds_name)

        testing_loss = 0.0
        num_batches = len(test_ds)

        # Set the model to evaluation mode
        model.eval()

        for batch in test_ds:
            batch = {k: v.to(device) for k, v in batch.items()}

            # Perform a forward pass
            outputs = model(**batch)
            loss = outputs.loss

            # Get logits and predictions
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            # Add batch predictions and references to the metric
            metric.add_batch(predictions=predictions, references=batch["labels"])

            testing_loss += loss.item()

        # Calculate the average test loss
        average_test_loss = testing_loss / num_batches

        # Compute the final evaluation metrics
        metrics = metric.compute()

        # Add the average test loss to the evaluation metrics
        metrics['testing_loss'] = average_test_loss

        # Compute and return the final evaluation metrics
        return average_test_loss, metrics

    return compute_metrics
            
            
def get_weights(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_weights(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
    
