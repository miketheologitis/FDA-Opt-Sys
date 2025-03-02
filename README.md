# fdaopt: A Flower / HuggingFace app

## Install dependencies and project

```bash
pip install flwr torch transformers datasets
```

## Run with the Simulation Engine

### Start Server

In the `fdaopt` directory:

```bash
python -m fdaopt.server
```

### Start Client(s)

In the `fdaopt` directory:

```bash
python -m fdaopt.client --client_id 0
```
```bash
python -m fdaopt.client --client_id 1
```
