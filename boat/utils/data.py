import numpy as np
from typing import Tuple

def split_sequence(sequence: np.ndarray, n_steps_in: int, n_steps_out: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split a univariate sequence into samples.

    Args:
        sequence (np.ndarray): The input sequence.
        n_steps_in (int): Number of input steps.
        n_steps_out (int): Number of output steps.

    Returns:
        Tuple[np.ndarray, np.ndarray]: X (inputs) and y (outputs).
    """
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def split_sequences_multivariate(sequences: np.ndarray, n_steps_in: int, n_steps_out: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split a multivariate sequence into samples.

    Args:
        sequences (np.ndarray): The input multivariate sequence.
        n_steps_in (int): Number of input steps.
        n_steps_out (int): Number of output steps.

    Returns:
        Tuple[np.ndarray, np.ndarray]: X (inputs) and y (outputs).
    """
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out - 1
        # check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix - 1:out_end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)
