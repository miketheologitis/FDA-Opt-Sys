import zmq
import numpy as np
import threading
import torch
from fdaopt.sketch import AmsSketch

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')

def create_push_socket(ip, port):
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.connect(f"tcp://{ip}:{port}")
    
    return socket

def create_pull_socket(ip, port):
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.bind(f"tcp://{ip}:{port}")
    
    return socket

def send_number_matrix(push_socket, sender_id, number, matrix):

    msg = {"id": sender_id, "number": number, "matrix": matrix.tolist()}

    push_socket.send_json(msg)
    
def send_number_anonymous(push_socket, number):

    msg = {"number": number}

    push_socket.send_json(msg)
    
def receive_number(pull_socket):
    msg = pull_socket.recv_json()
    
    return float(msg["number"])
    
    
def aggregate_numbers_matrices(pull_socket, n):
    local_matrices = []
    local_numbers = []
    sender_ids = []

    for _ in range(n):
        msg = pull_socket.recv_json()
        matrix = np.array(msg["matrix"])
        number = float(msg["number"])
        sender_id = int(msg["id"])
        local_matrices.append(matrix)
        local_numbers.append(number)
        sender_ids.append(sender_id)
        #logging.info(f"[Monitor Variance] Client {sender_id} Drift {number}!")

    mean_matrix = np.mean(local_matrices, axis=0)
    mean_number = np.mean(local_numbers)

    return sender_ids, mean_number, mean_matrix


def start_variance_monitoring_loop(pull_socket, n, clients_network):

    def loop():
        while True:
            sender_ids, mean_drift_sq, mean_sketch = aggregate_numbers_matrices(pull_socket, n)
            logging.info(f"[Monitor Variance] Aggregated local states successfully from Client IDs {sender_ids}!")
            
            # Compute the approximation of ||avg(u_t)||^2 using the sketch strategy
            epsilon = 0.002
            est = (1 / (1+epsilon)) * AmsSketch.estimate_euc_norm_squared(
                torch.from_numpy(mean_sketch)
            )
            
            # Compute total variance approximation
            variance_approx = mean_drift_sq - est
            
            # For each sender-client, send him back the variance approximation
            for sender_id in sender_ids:
                # find the client with sender_id
                for client in clients_network:
                    if client['id'] == sender_id:
                        # Create push socket
                        client_ip, client_port = client['ip'], client['port']
                        push_socket = create_push_socket(client_ip, client_port)
                        # Send him the variance approximation
                        send_number_anonymous(push_socket, variance_approx)
                        logging.info(f"[Monitor Variance] Successfully sent back to Client {sender_id} at {client_ip}:{client_port} the variance approximation {variance_approx:.4f}!")
            

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    