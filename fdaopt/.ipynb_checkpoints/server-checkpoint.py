"""fdaopt: A Flower / HuggingFace app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg, FedAdam
from transformers import AutoModelForSequenceClassification

from fdaopt.training import get_weights, get_evaluate_fn


def server_fn(context: Context):
    num_rounds = 100  # TODO
    fraction_fit = 0.5  # TODO
    model_checkpoint = "prajjwal1/bert-tiny"
    num_labels = 2  # TODO
    
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint, num_labels=num_labels
    )
    
    ds_path = "glue"
    ds_name = "mrpc"
    
    device = "cuda:0"

    weights = get_weights(model)
    initial_parameters = ndarrays_to_parameters(weights)

    # Define strategy
    strategy = FedAdam(
        fraction_fit=fraction_fit,
        fraction_evaluate=0.0,
        initial_parameters=initial_parameters,
        evaluate_fn=get_evaluate_fn(model, device, model_checkpoint, ds_path, ds_name),
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
server_app = ServerApp(server_fn=server_fn)
