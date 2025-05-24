### JSON Schema Explanation:

- `model`:
  - `checkpoint`: HuggingFace model checkpoint: Specifies the pre-trained model to use (e.g., "roberta-base").
  - `num_labels`: Defines the number of output labels for the task.

- `dataset`:
  - `path`: HuggingFace path or name of the dataset (e.g., "glue"). See [here](https://huggingface.co/docs/datasets/v3.2.0/en/package_reference/loading_methods#datasets.load_dataset).
  - `batch_size`: Batch size for data processing.
  - `dirichlet_alpha`: Dirichlet data distribution parameter for non-IID partitioning.
  - `name`: HuggingFace name of the dataset. See [here](https://huggingface.co/docs/datasets/v3.2.0/en/package_reference/loading_methods#datasets.load_dataset). (**Optional**)

- `training`:
  - `num_clients`: Total number of clients in the federation.
  - `clients_per_round`: Number of clients participating in each training round.
  - `local_epochs`: Number of epochs each client trains locally.
  - `total_rounds`: Total number of federated training rounds.

- `server`:
    - `network`:
      - `ip`: Server IP address.
      - `port`: Server port number.
      - `ip_pull_socket`: Server IP address for PULL socket.
      - `port_pull_socket`: Server IP address for PULL socket.
    - `strategy`: Specifies the federated learning strategy used in Flower.
      - `name`: Name of the strategy (e.g., `"FedAdam"`). See [here](https://flower.ai/docs/framework/ref-api/flwr.server.strategy.html).
      - `fda`: Whether or not to use FDA extention. Either `True` or `False` 
      - **Optional** (All strategy-specific applicable hyperparameters from [here](https://flower.ai/docs/framework/ref-api/flwr.server.strategy.html)):
         - e.g., `eta`: Server-side learning rate hyperparameter.

- `clients`:
  - `network`: List of client details:
    - `id`: Unique identifier for each client.
    - `ip`: IP address for client PULL socket. **Optional** (empty otherwise)
    - `port`: Port number for client PULL socket. **Optional** (empty otherwise)
    - `data_path`: Dataset path for each client. **Optional** (empty otherwise)
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


```bash
kafka-console-producer.sh --bootstrap-server localhost:9092 --topic FedL < test_1.json
```