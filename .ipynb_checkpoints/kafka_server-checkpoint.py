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
