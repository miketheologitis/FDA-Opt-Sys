### JSON Schema Explanation:

- `model`: Includes:
    - Required:
      - `checkpoint`: HuggingFace model checkpoint: Specifies the pre-trained model to use (e.g., "roberta-base").
      - `num_labels`: Defines the number of output labels for the task.

- `dataset`: Includes:
   - Required:
     - `path`: HuggingFace path or name of the dataset (e.g., "glue"). See [here](https://huggingface.co/docs/datasets/v3.2.0/en/package_reference/loading_methods#datasets.load_dataset).
     - `batch_size`: Batch size for data processing.
     - `dirichlet_alpha`: Dirichlet data distribution parameter for non-IID partitioning.
   - Optional:
     - `name`: HuggingFace name of the dataset (e.g., "mrpc"). See [here](https://huggingface.co/docs/datasets/v3.2.0/en/package_reference/loading_methods#datasets.load_dataset).

- `training`: Includes:
   - Required:
     - `num_clients`: Total number of clients in the federation.
     - `clients_per_round`: Number of clients participating in each training round.
     - `local_epochs`: Number of epochs each client trains locally.
     - `total_rounds`: Total number of federated training rounds.

- `server`: Includes:
   - Required:
     - `network`: Includes:
       - `ip`: Server IP address.
       - `port`: Server port number.
     - `strategy`: Specifies the federated learning strategy used in Flower.
       - `name`: Name of the strategy (e.g., `"FedAdam"`). See [here](https://flower.ai/docs/framework/ref-api/flwr.server.strategy.html).
       - `fda`: Whether or not to use FDA extention. Either `True` or `False` 
       - Optional (All strategy-specific applicable hyperparameters from [here](https://flower.ai/docs/framework/ref-api/flwr.server.strategy.html)):
         - e.g., `eta`: Server-side learning rate hyperparameter.

- `clients`: Includes:
   - Required:
     - `network`: List of client details:
       - `id`: Unique identifier for each client.
       - `ip`: IP address for the client.
       - `port`: Port number for communication.
       - Optional (null/empty otherwise):
         - `path`: Dataset path for each client.
     - `lr`: Learning rate for the client-side optimizer (SGD). The optimizer is fixed as Stochastic Gradient Descent (SGD) for all clients.

---

### Different Options

<details><summary>Federated Learning Strategies (Flower)</summary>

- [FedAvg](https://flower.ai/docs/framework/ref-api/flwr.server.strategy.FedAvg.html#flwr.server.strategy.FedAvg)
- [FedAdam](https://flower.ai/docs/framework/ref-api/flwr.server.strategy.FedAdam.html#flwr.server.strategy.FedAdam)
- [FedYogi](https://flower.ai/docs/framework/ref-api/flwr.server.strategy.FedAdam.html#flwr.server.strategy.FedYogi)
- [FedAdagrad](https://flower.ai/docs/framework/ref-api/flwr.server.strategy.FedAdam.html#flwr.server.strategy.FedAdagrad)
- [FedAvgM](https://flower.ai/docs/framework/ref-api/flwr.server.strategy.FedAdam.html#flwr.server.strategy.FedAvgM)

</details>
