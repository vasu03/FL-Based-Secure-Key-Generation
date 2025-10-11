"""
- File:     src/server/aggregator.py
- Desc:     Simulated IoT TCP-based server that acts as aggregator (FedAvg style) 
            for managing the global model. 
- Author:   Vasu Makadia
- License:  Apache License 2.0
"""


# Import required modules
import socket
import threading
import pickle
import struct
from typing import Any, Dict, Tuple

# Import custom modules
from src.model.model_utils import create_key_model, set_weights_from_numpy, average_weights, get_weights_as_numpy

# Define Global variables
HOST = "127.0.0.1"
BASE_PORT = 8000

class AggregatorServer:
    """
    Brief:
        Minimal TCP server to orchestrate federated rounds:
            - Accept client connections
            - Send current global weights
            - Receive local weights
            - Compute FedAvg and update global model
    Parameters:
        num_clients (int):          number of expected clients
        input_dim/output_dim (int): model dims for local model creation (used for typing)
    """
    # method to initialize the parameter
    def __init__(self, num_clients: int = 3, input_dim: int = 16, output_dim: int = 16):
        # Expected number of clients to participate in a round
        self.num_clients = num_clients
        # Create the "global" Keras model
        self.global_model = create_key_model(input_dim=input_dim, output_dim=output_dim)
        # Keep track of client sockets: {client_id: (socket, addr)}
        self.client_sockets: Dict[int, Tuple[socket.socket, Tuple[str,int]]] = {}
        # A simple lock to protect shared state
        self.lock = threading.Lock()
        # Use a listening socket to accept incoming clients
        self.listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Bind to HOST and BASE_PORT
        self.listen_socket.bind((HOST, BASE_PORT))
        self.listen_socket.listen(self.num_clients)
        print(f"[Aggregator] Listening on {HOST}:{BASE_PORT} for {num_clients} clients\n")


    def accept_clients(self) -> None:
        """
        Brief:
            Accept expected number of clients and store their sockets.
        Parameters:
            None
        Returns:
            None
        """
        # Accept clients until we have num_clients
        while len(self.client_sockets) < self.num_clients:
            client_sock, addr = self.listen_socket.accept()
            # Expect client to send an integer id on connect
            id_bytes = client_sock.recv(4)
            if len(id_bytes) < 4:
                print("[Aggregator] malformed client id")
                client_sock.close()
                continue
            # Unpack client id (network byte order)
            client_id = struct.unpack("!I", id_bytes)[0]
            with self.lock:
                self.client_sockets[client_id] = (client_sock, addr)
            print(f"[Aggregator] Client {client_id} connected from {addr}")


    def send_pickle(self, sock: socket.socket, obj) -> None:
        """
        Brief:
            Helper to send a pickled object with length-prefix framing.
        Parameters:
            sock (socket.socket):   connected socket
            obj: (Any)              picklable object
        Returns:
            None
        """
        data = pickle.dumps(obj)
        # Send length prefix (4 bytes network-endian) then payload
        sock.sendall(struct.pack("!I", len(data)))
        sock.sendall(data)


    def recv_pickle(self, sock: socket.socket, timeout: float = 10.0) -> Any | None:
        """
        Brief:
            Receive a length-prefixed pickled object from socket.
        Parameters:
            sock (socket.socket):   connected socket
            timeout (float):        receive timeout in seconds
        Returns:
            object:                 unpickled object or None on error/timeout
        """
        sock.settimeout(timeout)
        try:
            # Read length first
            raw_len = sock.recv(4)
            if not raw_len or len(raw_len) < 4:
                return None
            msg_len = struct.unpack("!I", raw_len)[0]
            # Read full payload
            data = b''
            while len(data) < msg_len:
                chunk = sock.recv(msg_len - len(data))
                if not chunk:
                    return None
                data += chunk
            obj = pickle.loads(data)
            return obj
        except Exception as e:
            print(f"[Aggregator] recv_pickle error: {e}")
            return None


    def run_round(self, round_idx: int, local_epochs: int = 1) -> None:
        """
        Brief:
            Perform a single federated round: send global weights, receive local updates,
            average them and update the global model.
        Parameters:
            round_idx (int):    index of the round (for logging)
            local_epochs (int): not used by server; informative
        Returns:
            None
        """
        print(f"\n[Aggregator] Starting round {round_idx}")
        # Get current global weights as numpy list
        global_weights = get_weights_as_numpy(self.global_model)

        # Send global weights to all clients
        for cid, (sock, addr) in list(self.client_sockets.items()):
            try:
                self.send_pickle(sock, {"cmd": "global_weights", "weights": global_weights})
            except Exception as e:
                print(f"[Aggregator] failed to send to client {cid}: {e}")

        # Collect weights from clients
        collected = []
        for cid, (sock, addr) in list(self.client_sockets.items()):
            # Receive a pickled message containing updated weights
            msg = self.recv_pickle(sock, timeout=20.0)
            if msg and "weights" in msg:
                collected.append(msg["weights"])
                print(f"[Aggregator] Received weights from client {cid}")
            else:
                print(f"[Aggregator] No weights from client {cid} (skipping)")

        # If we received at least one client, average
        if len(collected) > 0:
            new_global = average_weights(collected)
            # Update global model's weights
            set_weights_from_numpy(self.global_model, new_global)
            print(f"[Aggregator] Updated global model after round {round_idx}")
        else:
            print("[Aggregator] No client updates collected in this round.")


    def close(self):
        """
        Brief:
            Close all sockets and cleanup.
        Parameters:
            None
        Returns:
            None
        """
        for cid, (sock, addr) in list(self.client_sockets.items()):
            try:
                sock.close()
            except:
                pass
        self.listen_socket.close()


    def shutdown_clients(self):
        """
        Brief:
            Inform all clients to exit after final round.
        Parameters:
            None
        Returns:
            None
        """
        for cid, (sock, addr) in list(self.client_sockets.items()):
            try:
                self.send_pickle(sock, {"cmd": "shutdown"})
                sock.close()
            except:
                pass
