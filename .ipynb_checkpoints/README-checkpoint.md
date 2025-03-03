# fdaopt: A Flower / HuggingFace app

## Install dependencies and project

```bash
pip install flwr torch transformers datasets confluent-kafka
```

## Run with the Simulation Engine

### Start Server

In the `fdaopt` directory:

```bash
python -m fdaopt.server --local_json /home/mtheologitis/FDA-Opt-Sys/hyperparameters/0.json
```

### Start Client(s)

In the `fdaopt` directory:

```bash
python -m fdaopt.client --client_id 0 --local_json /home/mtheologitis/FDA-Opt-Sys/hyperparameters/0.json
```
```bash
python -m fdaopt.client --client_id 1 --local_json /home/mtheologitis/FDA-Opt-Sys/hyperparameters/0.json
```
