from confluent_kafka import Consumer, KafkaError
import json
import uuid
import sys
import subprocess
import logging
from pathlib import Path
import time
import socket
import argparse
import os

# Path to the current script's directory
LOCAL_BASE_DIR = Path(__file__).resolve().parent
LOCAL_HYPERPARAMS_DIR = LOCAL_BASE_DIR / "hyperparameters"
LOCAL_LOGS_DIR = LOCAL_BASE_DIR / "logs"
LOCAL_MEDIATOR_LOG_FILE = LOCAL_LOGS_DIR / "mediator.log"

# Common format
LOG_FORMAT = '%(asctime)s - %(jobid)s - %(message)s'
DATE_FORMAT = '%H:%M:%S'

# Create handlers
file_handler = logging.FileHandler(LOCAL_MEDIATOR_LOG_FILE)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

# Configure the root logger
logging.basicConfig(level=logging.DEBUG, handlers=[file_handler, console_handler])


launched_processes = []


def is_port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0


def launch_job(params):
    global launched_processes
    
    # -------- Step 1. Create Local .json
    
    job_id = str(uuid.uuid4())[:8]
    json_name = "tmp-" + job_id + '.json'
    json_path = LOCAL_HYPERPARAMS_DIR / json_name
    
    # Ensure the directory exists
    json_path.parent.mkdir(parents=True, exist_ok=True) 
    
    # Write dictionary to JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(params, f, ensure_ascii=False)
        
    # convert to string
    json_path_str = str(json_path)
    
    d = {"jobid": job_id}
    
    logging.info(f"Job is launching...", extra=d)
    
    # -------- Step 2. Launch Flower FL server
    server_ip = params['server']['network']['ip']
    server_port = params['server']['network']['port']
    
    if is_port_in_use(server_ip, server_port):
        logging.error(f"The Flower server socket address {server_ip}:{server_port} is in use. Aborting Job...", extra=d)
        return
    
    log_file = LOCAL_LOGS_DIR / f"server-{job_id}.log"
    with open(log_file, "w") as log:
        
        env = dict(os.environ, TERM="dumb")
        
        proc = subprocess.Popen(
            [sys.executable, "-m", "fdaopt.server", '--local_json', json_path_str],
            cwd=LOCAL_BASE_DIR,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env
        )
        
        launched_processes.append(proc)
    
    logging.info(f"Flower FL Server has successfully launched!", extra=d)
    
    # -------- Step 3. Launch Clients
    
    # Wait so that the server has been launched
    while not is_port_in_use(server_ip, server_port):
        logging.info(f"Waiting for Flower FL Server to boot up...", extra=d)
        time.sleep(5)
        
    logging.info(f"Flower FL Server is up and running!", extra=d)
    
    num_clients = params['training']['num_clients']
    
    for i in range(num_clients):
        
        log_file = LOCAL_LOGS_DIR / f"client-{i}-{job_id}.log"
        
        with open(log_file, "w") as log:
            
            env = dict(os.environ, TERM="dumb")
            
            proc = subprocess.Popen(
                [sys.executable, "-m", "fdaopt.client", '--client_id', str(i), '--local_json', json_path_str, '--cuda', '0'],
                cwd=LOCAL_BASE_DIR,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env
            )
            
            launched_processes.append(proc)
        
        logging.info(f"Client {i} has successfully launched!", extra=d)
        
    logging.info(f"All components have launched successfully!", extra=d)
    

def listen_to_kafka(topic='FedL', bootstrap_servers='localhost:9092'):
    
    d = {"jobid": 'Root'}

    # Consumer example
    c = Consumer({
        'bootstrap.servers': bootstrap_servers,
        'group.id': str(uuid.uuid4()),
        'auto.offset.reset': 'latest'
    })

    c.subscribe([topic])
    
    logging.info(f"Just subcribed to Kafka topic {topic}!", extra=d)
    
    logging.info(f"Waiting for new jobs from Kafka...!", extra=d)

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
        
        logging.info(f"Just read new job parameters!", extra=d)
        
        # Parse the JSON string back into a Python data structure
        parameters = json.loads(val)

        launch_job(parameters)
        
        logging.info(f"Waiting for new jobs from Kafka...!", extra=d)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cleanup', action='store_true', help="If given we cleanup all JSON/Output files.")
    parser.add_argument('--topic', type=str, default="FedL", help="The Kafka topic to subscribe to.")
    parser.add_argument('--local', type=str, default='', help="A local .json file to run a single experiment.")
    args = parser.parse_args()
    
    d = {"jobid": 'Root'}
    
    try:
        if args.local:
            
            logging.info(f"Launching one-time local job!", extra=d)
            
            with open(args.local) as f:
                params = json.load(f)
            launch_job(params)
            
            while True:
                time.sleep(5)
            
        else:
            listen_to_kafka(topic=args.topic)
            
    except KeyboardInterrupt:
        
        for file in LOCAL_HYPERPARAMS_DIR.glob('tmp-*.json'):
                file.unlink()
        
        if args.cleanup:
            logging.info(f"Cleaning up server/client logs as requested...", extra=d)
            
            for ext in ("server-*.log", "client-*.log"):
                for file in LOCAL_LOGS_DIR.glob(ext):
                    file.unlink()
                    
            logging.info(f"Cleaned up server/client logs!", extra=d)
            
        logging.warning("Terminating all subprocesses...", extra=d)
        
        for proc in launched_processes:
            if proc.poll() is None:  # still running
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
        logging.info("All subprocesses terminated. Exiting...", extra=d)
    except Exception as e:
        logging.exception(f"Unhandled exception occurred: {e}", extra=d)
    finally:
        logging.info("Mediator shutdown complete.", extra=d)