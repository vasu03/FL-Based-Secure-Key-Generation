"""
- File:     src/client/client_node.py
- Desc:     Simulated IoT client that connects to aggregator,
            performs local training and derives a cryptographic key.
- Author:   Vasu Makadia
- License:  Apache License 2.0
"""


# Import required modules
import pickle
import socket
import struct
import os
import warnings
import time
from typing import Any
import numpy as np

# Import custom modules
from src.model.model_utils import create_key_model, set_weights_from_numpy, get_weights_as_numpy
from src.data.synthetic_data_generator import generate_local_dataset
from src.crypto.key_utils import derive_key_from_vector

# To supress tensorflow warnings related to CUDA drivers
# "0" - all, "1" - info, "2" - warning, "3" = error
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISISBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")


# Define global variables
HOST = "127.0.0.1"
BASE_PORT = 8000

# Define a class for Client Node
class ClientNode:
    """
    A class representing a client node in a simulated IoT environment
    """
    # a method to initialize the parameters
    def __init__ (self, client_id: int, input_dim: int = 16, output_dim: int = 16):
        self.client_id = client_id
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = create_key_model(input_dim=input_dim, output_dim=output_dim)
        self.X, self.y = generate_local_dataset(client_id, num_samples=200, input_dim=input_dim)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.running = True


    def connect (self, host:str = HOST, port: int = BASE_PORT, retries: int = 5, delay:int = 2) -> None:
        """
        Brief:
            To connect with the server aggregator within the simulated IoT Network
        Parameters:
            host (str):     Server IP address
            port (int):     Port no. to connect on with the server
            retries (int):  No. of times to retry for connection with server
            delay (int):    Delay between the retries for connection
        Returns:
            None
        """
        for attempt in range(retries):
            try:
                self.sock.connect((host, port))
                self.sock.sendall(struct.pack("!I", self.client_id))
                print(f"[Client {self.client_id} Connected to {host}:{port}]")
                return
            except ConnectionRefusedError:
                print(f"[Client {self.client_id}] Connected attempt {attempt+1} failed. Retrying in {delay}s...")
                time.sleep(delay)
        raise ConnectionError(f"[Client {self.client_id}] Could not connect to server after {retries} attempts")


    def recv_pickle (self, timeout: float = 20.0):
        """
        Brief:
            To receive the messages from the Server aggregator
        Parameters:
            timeout (float):    Message receiving timeout in seconds
        Returns:
            Any | None
        """
        self.sock.settimeout(timeout)
        try:
            # get the raw packet length
            raw_len = self.sock.recv(4)
            if not raw_len or len(raw_len) < 4:
                print(f"[Client {self.client_id}] Warning: Incomplete length header received.")
                return None
            # Get the length of message
            msg_len = struct.unpack("!I", raw_len)[0]
            data = b""
            while len(data) < msg_len:
                chunk = self.sock.recv(msg_len - len(data))
                if not chunk:
                    print(f"[Client {self.client_id}] Warning: Connection closed while receiving data")
                    return None
                data += chunk
            obj = pickle.loads(data)
            print(f"[Client {self.client_id}] Received Message: {obj.get('cmd', '<unknown>')} ({len(data)} bytes)")
            return obj
        except Exception as e:
            print(f"[Client {self.client_id}] Error in recv_pickle: {e}")
            return None


    def send_pickle (self, obj: Any) -> None:
        """
        Brief:
            To send messages to the Server aggregator
        Parameters:
            obj (Any):  A data object that is to be sent
        Returns:
            None
        """
        try:
            data = pickle.dumps(obj)
            self.sock.sendall(struct.pack("!I", len(data)))
            self.sock.sendall(data)
            print(f"[Client {self.client_id}] Sent Message: {obj.get('cmd', '<unknown>')} ({len(data)} bytes)")
        except Exception as e:
            print(f"[Client {self.client_id}] Error in send_pickle: {e}")
            raise


    def run (self, local_epochs: int = 3, batch_size: int = 32) -> None:
        """
        Brief:
            To execute the local model training for key generation. Keep the connection alive
            and wait for server commands.
        Parameters:
            local_epochs (int):     No of iterations to perform during local model training
            batch_size (int):       Size of batch containing samples to be used for training
        Returns:
            None
        """
        while self.running:
            # receive the message from server with 60 seconds timeout
            msg = self.recv_pickle(timeout=60.0)
            if not msg:
                print(f"[Client {self.client_id}] No message received, server may have closed. Exiting.")
                break

            cmd = msg.get("cmd", "")
            if cmd == "global_weights":
                # set the global weights
                set_weights_from_numpy(self.model, msg["weights"])
                # Train the model locally
                self.model.fit(self.X, self.y, epochs=local_epochs, batch_size=batch_size, verbose=0)
                # Send the updated model weights
                updated_weights = get_weights_as_numpy(self.model)
                diff = np.sum([np.sum(np.abs(u - g)) for u, g in zip(updated_weights, msg["weights"])])
                print(f"[Client {self.client_id}] Weight change after training: {diff: .6f}")
                try:
                    self.send_pickle({"cmd": "local_weights", "weights": updated_weights})
                    print(f"[Client {self.client_id}] Sent Updated weights")
                except Exception as e:
                    print(f"[Cleint {self.client_id}] Failed to send weights: {e}")
                    break
            elif cmd == "shutdown":
                print(f"[Client {self.client_id}] Received Shutdown. Exiting...")
                self.running = False
                break
            else:
                print(f"[Client {self.client_id}] Unknown Command: {cmd}")


    def derive_key(self, challenge_vector: np.ndarray, salt: bytes = b"", info: bytes = b"") -> bytes:
        """
        Brief:
        Parameters:
            challenge_vector (np.ndarray):  An array of challenge to use for key derivation
            salt (bytes):                   Salt to be added during HKDF
            info (bytes):                   Info to be added during HKDF
        Returns:
            bytes:                          Derived Key in bytes format
        """
        try:
            # set the input for key derivation function
            inp = np.expand_dims(challenge_vector.astype(np.float32), axis=0)
            # get the output from the model predictions
            out = self.model.predict(inp, verbose=0)[0]
            # derive the key usng Key Derivation function
            key = derive_key_from_vector(out, salt=salt, info=info, key_len=32)
            # Print the hex formatted key partially for debugging
            print(f"[Client {self.client_id}] Derived key: {key.hex()[:16]}...")
            return key
        except Exception as e:
            print(f"[Client {self.client_id}] Error deriving key: {e}")
            return b""


    def close(self) -> None:
        """
        Brief:
            To close the client connection with Server in network
        Parameters:
            None
        Retrurns:
            None
        """
        try:
            self.sock.close()
        except Exception as e:
            print(f"[Client {self.client_id}] Error closing connection: {e}")
