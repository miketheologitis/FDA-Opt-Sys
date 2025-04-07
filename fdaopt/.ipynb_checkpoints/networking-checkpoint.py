import zmq
import numpy as np
import threading

import zmq
import numpy as np

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

def send_number_matrix(push_socket, number, matrix):

    msg = {"number": number, "matrix": matrix.tolist()}

    push_socket.send_json(msg)
    
def receive_number(pull_socket):
    msg = pull_socket.recv_json()
    
    return float(msg["number"])
    
    
def aggregate_numbers_matrices(pull_socket, n):
    local_matrices = []
    local_numbers = []

    for _ in range(n):
        msg = pull_socket.recv_json()
        matrix = np.array(msg["matrix"])
        number = float(msg["number"])
        local_matrices.append(matrix)
        local_numbers.append(number)

    mean_matrix = np.mean(local_matrices, axis=0)
    mean_number = np.mean(local_numbers)

    return mean_number, mean_matrix


def start_variance_monitoring_loop(pull_socket, n):
    def loop():
        while True:
            mean_number, mean_matrix = aggregate_numbers_matrices(pull_socket, n)
            
            # ---- HERE CHANGE  Callback
            print("Aggregation complete")
            print("Mean number:", mean_number)
            print("Mean matrix:\n", mean_matrix)
            # TODO: Save in file and see all ok. Thread thats why problematic
            # ---- HERE CHANGE  Callback

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    