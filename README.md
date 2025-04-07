# fdaopt: A Flower / HuggingFace app

## Install dependencies and project

```bash
pip install flwr torch transformers datasets confluent-kafka
```

## Run

### Start Mediator

Continuously listen and wait for parameters in Kafka and launch FL jobs:

```bash
python mediator.py --cleanup
```

Run 1-time local parameters and launch 1 job:
```bashγσ
python mediator.py --cleanup --local /home/mtheologitis/FDA-Opt-Sys/hyperparameters/test_8084.json
```




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
