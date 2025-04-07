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

For example, go to `/hyperparameters` and run:
```bash
kafka-console-producer.sh --bootstrap-server localhost:9092 --topic  FedL < test_8085.json
```

Run 1-time local parameters and launch 1 job:
```bash
python mediator.py --cleanup --local /home/mtheologitis/FDA-Opt-Sys/hyperparameters/test_8084.json
```