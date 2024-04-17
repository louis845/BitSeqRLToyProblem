import numpy as np

def bit_sequence_to_int(x: np.ndarray, n_exp: np.ndarray):
    """Assumes x is of shape (B, n), and n_exp is of shape (n,)"""
    return x @ n_exp

def int_to_bit_sequence(x: int, n_exp: np.ndarray):
    """Assumes x is of shape (B,), and n_exp is of shape (n,)"""
    return ((np.expand_dims(x, axis=1) & np.expand_dims(n_exp, axis=0)) > 0).astype(np.int32)