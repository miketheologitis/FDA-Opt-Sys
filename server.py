import sys

# Open a file in write mode
log_file = open("logs/server.out", "w")

# Redirect stdout and stderr to the file
sys.stdout = log_file
sys.stderr = log_file

from confluent_kafka import Consumer, KafkaError
import subprocess
import json

def kafka_get_test_hyper_parameters(topic='FedL', bootstrap_servers='localhost:9092', group_id='fda1'):

    # Consumer example
    c = Consumer({
        'bootstrap.servers': bootstrap_servers,
        'group.id': group_id,
        'auto.offset.reset': 'latest'
    })

    c.subscribe([topic])

    while True:
        msg = c.poll(1.0)
        if not msg:
            continue
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                continue
            else:
                print(msg.error())
                break
        val = msg.value().decode("utf-8")

        # Parse the JSON string back into a Python data structure
        parameters = json.loads(val)

        c.close()

        return parameters

import time 

time.sleep(5)

print("Waiting for Federated Learning parameters from Kafka...")
sys.stdout.flush()

time.sleep(5)
#params = kafka_get_test_hyper_parameters()

#print(params)

print("Received Federated Learning parameters from Kafka!")
sys.stdout.flush()
time.sleep(1)
print("Starting Federated Learning Workflow...")
sys.stdout.flush()
time.sleep(2)


from flwr.simulation import run_simulation

from fdaopt.client_app import client_fn
from flwr.client import ClientApp

# Construct the ClientApp passing the client generation function
client_app = ClientApp(client_fn=client_fn)

from flwr.server import ServerApp
from fdaopt.server_app import server_fn

# Create your ServerApp passing the server generation function
server_app = ServerApp(server_fn=server_fn)

run_simulation(
    server_app=server_app,
    client_app=client_app,
    num_supernodes=20,
    backend_config={"client_resources": {"num_cpus": 4, "num_gpus": 0.1}}
)

#https://packaging.python.org/en/latest/guides/writing-pyproject-toml/