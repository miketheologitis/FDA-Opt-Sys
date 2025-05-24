"""fdaopt: A Flower / HuggingFace app."""

from collections import OrderedDict

import torch
import evaluate
from fdaopt.data import get_test_ds
from fdaopt.networking import send_number_matrix, receive_number

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')


@torch.no_grad
def compute_drift(old_params, new_params, device):
    """Compute the drifts between old and new parameters."""

    drifts = []
    for old_param, new_param in zip(old_params, new_params):
        old_param = old_param.to(device)
        new_param = new_param.to(device)

        drifts.append(
            (new_param - old_param).to(device)
        )

    return drifts

@torch.no_grad
def vectorize(parameters, device):
    """Vectorize a list of parameters."""

    con = []
    for param in parameters:
        param = param.to(device)
        con.append(param.reshape(-1))

    return torch.cat(con)

@torch.no_grad
def get_weights(model):
    """Get the model weights as a list of numpy arrays."""

    return [val.cpu().numpy() for _, val in model.state_dict().items()]

@torch.no_grad
def set_weights(model, parameters):
    """Set the model weights from a list of numpy arrays."""

    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def train(model, optimizer, trainloader, device, epochs):
    
    # Set the model to training mode
    model.train()
    
    for i in range(epochs):
        
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
        
        logging.info(f"[Client Training - FedOpt] Epoch {i+1}/{epochs} complete!")
        
        
def train_fda(model, optimizer, trainloader, device, epochs, client_id, threshold, push_to_server_socket, pull_variance_approx_socket, ams_sketch):
    
    # Set the model to training mode
    model.train()
    
    # Extract trainable parameters from the model, which reside on the device that the model resides in
    train_params = [param for param in model.parameters()]
    # Create a copy of the trainable parameters in the same device as model, detached from the computation graph
    round_start_train_params = [param.detach().clone() for param in train_params]
    
    for i in range(epochs):
        
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
            
        
        drift_vec = vectorize(
            compute_drift(round_start_train_params, train_params, device),
            device
        )
        # Compute the squared l2 norm of the client-drift
        norm_sq_drift = float(torch.dot(drift_vec, drift_vec).cpu().numpy())
        # compute sketch
        sketch = ams_sketch.sketch_for_vector(drift_vec).numpy()
            
        send_number_matrix(push_to_server_socket, client_id, norm_sq_drift, sketch)
        logging.info(f"[Client Training - FDA-Opt] Successfully sent local state to server!")

        variance_approx = receive_number(pull_variance_approx_socket)
        logging.info(f"[Client Training - FDA-Opt] Successfully received variance approximation from server: {variance_approx:.4f}!")
        
        logging.info(f"[Client Training - FDA-Opt] Epoch {i+1}/{epochs} complete!")
    
        if variance_approx > threshold:
            logging.info(f"[Client Training - FDA-Opt] Threshold is violated. Ending round!")
            return i+1
        
        logging.info(f"[Client Training - FDA-Opt] Variance approximation is still bellow the threshold!")
        
    return epochs
            


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
            
    
