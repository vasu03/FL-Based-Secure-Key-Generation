"""
- File:     src/experiments/run_experiment.py
- Desc:     A simple orchestrator to run the aggregator & multiple clients in threads for experiments
- Author:   Vasu Makadia
- License:  Apache License 2.0
"""


# Import required modules
import threading
import time
import numpy as np

# Import custom modules
from src.server.aggregator import AggregatorServer
from src.client.client_node import ClientNode


def client_thread_fn(client_id: int) -> None:
    """
    Brief:
        Helper to run a client instance: connect, wait for rounds, train and then derive a key.
    Parameters:
        client_id (int):    A numeric serial/id of the client
    Returns:
        None
    """
    client = ClientNode(client_id=client_id)
    client.connect()
    # For this simple demo we expect server to request training once
    client.run()
    # After some delay expect a challenge for key derivation (in a simple test we'll derive locally)
    # For the demo we just close (actual key derivation would happen after final model sync)
    client.close()


def run_demo(num_clients=3, rounds=3):
    """
    Brief:
        Launch aggregator and client threads to perform multiple federated rounds.
    Parameters:
        num_clients (int):  number of simulated client nodes
        rounds (int):       number of federated averaging rounds
    Returns:
        None
    """
    # Start aggregator
    agg = AggregatorServer(num_clients=num_clients)
    # Accept clients in separate thread
    accept_thread = threading.Thread(target=agg.accept_clients, daemon=True)
    accept_thread.start()

    # Launch client threads that will connect
    client_threads = []
    for cid in range(num_clients):
        t = threading.Thread(target=client_thread_fn, args=(cid,), daemon=True)
        t.start()
        client_threads.append(t)
        time.sleep(0.2)  # small stagger to avoid connection races

    # Wait for clients to connect
    time.sleep(1.0)

    # Run federated rounds
    for r in range(rounds):
        agg.run_round(round_idx=r)
        time.sleep(0.5)

    # After training rounds, demonstrate challenge-based key derivation
    # For demo: pick a fixed challenge vector all clients will use
    challenge = np.ones((agg.global_model.input_shape[1],), dtype=np.float32) * 0.12345
    # For demo: set the same global model locally by connecting to clients
    # (In production we broadcast global weights and clients set them before deriving)
    global_weights = agg.global_model.get_weights()

    # For simplicity, spin up temporary client objects to derive keys locally using the final global model
    derived_keys = []
    for cid in range(num_clients):
        c = ClientNode(client_id=cid)
        c.model.set_weights(global_weights)
        k = c.derive_key(challenge_vector=challenge)
        derived_keys.append(k)
        c.close()

    # Print hex of keys and agreement stats
    hex_keys = [k.hex() for k in derived_keys]
    print("\nDerived keys (hex):")
    for i, hk in enumerate(hex_keys):
        print(f" client {i}: {hk[:32]}...")  # print prefix only
    # Agreement check
    unique = set(hex_keys)
    print(f"\nUnique keys count: {len(unique)} out of {len(hex_keys)}\n")
    agg.shutdown_clients()
    agg.close()


if __name__ == "__main__":
    # Run demo with default settings
    run_demo(num_clients=3, rounds=3)
