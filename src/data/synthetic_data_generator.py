"""
- File:     src/data/synthetic_data_generator.py
- Desc:     To generate synthetic correlated data for simulated IoT clients
- Author:   Vasu Makadia
- License:  Apache License 2.0
"""


# Import required modules
import numpy as np


def generate_local_dataset (
    client_id:int,
    num_samples: int = 200,
    input_dim: int = 16,
    base_correlation: float = 0.9,
    noise_std: float = 0.05
):
    """
    Brief:
        Generate a synthetic correlated dataset localy for simulated IoT client.
        Each client's data is generated from a common base vector plus small client-specific
        perturbations so that datasets are correlated but not identical.
    Parameters:
        client_id (int):            unique integer id for the client (affects the offset)
        num_samples(int):           no. of local training samples to be generated
        inpute_dim (int):           dimensionality of each input sample
        base_correlation (float):   scales how similar client's data is to global base
        noise_std (float):          standard deviation of additive Gaussian Noise
    Returns:
        (X, y):
            X (np.ndarray):         shape (num_samplesm input_dim) training inputs
            y (np.ndarray):         shape (num_samples, output_dim) training targets
    """

    # Create a global base pattern the aggregator could distribute
    rng = np.random.RandomState(seed=12345)       # deterministic base across runs
    global_base = rng.normal(loc=0.0, scale=1.0, size=(input_dim,))

    # Create a small deterministic client specific offset to break perfect equality in data
    client_offset = (client_id % 10) * 0.01

    # Prepare the initial vectors
    X = np.zeros((num_samples, input_dim), dtype=np.float32)
    y = np.zeros((num_samples, input_dim), dtype=np.float32)

    for i in range(num_samples):
        # start from global base
        sample = base_correlation * global_base
        # add client-specific offset and noise
        sample += client_offset
        sample += rng.normal(scale=noise_std, size=(input_dim,))
        # for simulations set y=sample, but in real scenarios it could be a derived feature
        X[i] = sample
        y[i] = sample

    return X, y
