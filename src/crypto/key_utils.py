"""
- File:     src/crypto/key_utils.py
- Desc:     Cryptographic key derivation utilities. To convert model outputs into 
            deterministic keys using HKDF + SHA256
- Author:   Vasu Makadia
- License:  Apache License 2.0
"""


# Import required modules
import hashlib
import numpy as np
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend


def quantize_vector (vec: np.ndarray, decimals: int = 6) -> np.ndarray:
    """
    Brief:
        Quantize float vector to fixed decimal places to reduce numerical mismatch
        before hashing into a key
    Parameters:
        vec (np.ndarray):   continuos output vector from the model
        decimals (int):     number of decimal places to round to
    Returns:
        np.ndarray:         quantized vector of strings encoded bytes-ready
    """
    # Round to fixed decimals; convert to bytes-friendly representation
    rounded = np.round(vec, decimals=decimals)
    return rounded


def derive_key_from_vector (
        vec: np.ndarray,
        salt: bytes = b"",
        info: bytes = b"",
        key_len: int = 32
) -> bytes:
    """
    Brief:
        Deterministically derive a symmetric key from a numeric vector using SHA256 + HKDF
    Parameters:
        vec (np.ndarray):   input numeric vector
        salt (bytes):       optional salt for HKDF
        info (bytes):       optional context info for HKDF
        key_len (int):      desired output key length in bytes (e.g., 16 for AES128, 32 for AES256)
    Return:
    """
    # Convert vector into bytes in a stable manner
    # Use SHA256 over the bytes representation to get a uniform premise
    # First quantize to reduce tiny floating differences
    q = quantize_vector (vec, decimals=6)
    # Flatten and convert to bytes Deterministically
    vec_bytes = q.tobytes()
    # Hash with SHA256 to produce a fixed length digest for HKDF input
    digest = hashlib.sha256(vec_bytes).digest()
    # Use HKDF to expand to desired key length
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=key_len,
        salt=salt,
        info=info,
        backend=default_backend()
    )
    # derive final key bytes
    key = hkdf.derive(digest)

    return key
