import zmq
import numpy as np
import threading
import torch
from fdaopt.sketch import AmsSketch

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')

def create_push_socket(ip, port):
    """Create a ZeroMQ PUSH socket to send messages to the specified IP and port.
    Args:
        ip (str): The IP address to connect to.
        port (int): The port number to connect to.
    Returns:
        zmq.Socket: A ZeroMQ PUSH socket connected to the specified IP and port.
    """

    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.connect(f"tcp://{ip}:{port}")
    
    return socket

def create_pull_socket(ip, port):
    """Create a ZeroMQ PULL socket to receive messages from the specified IP and port.
    Args:
        ip (str): The IP address to bind to.
        port (int): The port number to bind to.
    Returns:
        zmq.Socket: A ZeroMQ PULL socket bound to the specified IP and port.
    """

    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.bind(f"tcp://{ip}:{port}")
    
    return socket

def send_number_matrix(push_socket, sender_id, number, matrix):
    """Send a message containing a number and a matrix to the specified push socket.
    Args:
        push_socket (zmq.Socket): The ZeroMQ PUSH socket to send the message to.
        sender_id (int): The ID of the sender.
        number (float): The number to send.
        matrix (np.ndarray): The matrix to send.
    """

    msg = {"id": sender_id, "number": number, "matrix": matrix.tolist()}

    push_socket.send_json(msg)
    
def send_number_anonymous(push_socket, number):
    """Send a message containing a number to the specified push socket without sender ID.
    Args:
        push_socket (zmq.Socket): The ZeroMQ PUSH socket to send the message to.
        number (float): The number to send.
    """

    msg = {"number": number}

    push_socket.send_json(msg)
    
def receive_number(pull_socket):
    """Receive a message containing a number from the specified pull socket.
    Args:
        pull_socket (zmq.Socket): The ZeroMQ PULL socket to receive the message from.
    Returns:
        float: The number received from the message.
    """
    msg = pull_socket.recv_json()
    
    return float(msg["number"])
    
    
def aggregate_numbers_matrices(pull_socket, n):
    """Aggregate the number and matrix received from clients.
    Args:
        pull_socket (zmq.Socket): The ZeroMQ PULL socket to receive messages from.
        n (int): The number of clients to aggregate from.
    Returns:
        tuple: A tuple containing:
            - List of sender IDs (int)
            - Mean number (float)
            - Mean matrix (np.ndarray)
    """

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
    """Start a loop to monitor variance by aggregating local states from clients.
    Args:
        pull_socket (zmq.Socket): The ZeroMQ PULL socket to receive messages from clients.
        n (int): The number of clients to aggregate from.
        clients_network (list): List of all client network information, each containing 'id', 'ip', and 'port'.
    """

    def loop():
        while True:
            # Receive the aggregated local states from clients
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
            
    # Create a daemon thread to run the loop
    t = threading.Thread(target=loop, daemon=True)
    t.start()
    