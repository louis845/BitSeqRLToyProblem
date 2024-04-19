import numpy as np

def bit_sequence_to_int(x: np.ndarray, n_exp: np.ndarray):
    """Assumes x is of shape (B, n), and n_exp is of shape (n,)"""
    return x @ n_exp

def int_to_bit_sequence(x: int, n_exp: np.ndarray):
    """Assumes x is of shape (B,), and n_exp is of shape (n,)"""
    return ((np.expand_dims(x, axis=1) & np.expand_dims(n_exp, axis=0)) > 0).astype(np.int32)

def sample_uniform(n: int, k: int) -> np.ndarray:
    """Sample k integers from [0, n) uniformly without replacement."""
    if k > n:
        remaining = k % n
        return np.concatenate([np.tile(np.arange(n), k // n), np.random.choice(n, remaining, replace=False)])
    return np.random.choice(n, k, replace=False)