import numpy as np

amino_acids = "ACDEFGHIKLMNPQRSTVWY"

def encode(seq, max_len=None):
    if max_len is None:
        max_len = len(sequence)
        
    if len(seq) > max_len:
        raise Exception('Sequence is longer than max_len!')
        
    encoding = np.zeros((max_len, 21))
    
    for i, aa in enumerate(seq):
        encoding[i, amino_acids.index(aa)] = 1
    if len(seq) < max_len:
        encoding[len(seq):, 20] = 1
    return encoding


def batch_encode(seq_list):
    max_len = max(len(seq) for seq in seq_list)
    encoding_list = [encode(seq, max_len) for seq in seq_list]
    return np.stack(encoding_list)

def batch_flatten(encodings):
    if not isinstance(encodings, np.ndarray):
        raise Exception(f"Expected numpy array, got {type(encodings)}!")
    batch_size, seq_len, n_residues = encodings.shape
    return encodings.reshape(batch_size, seq_len * n_residues)

def batch_encode_and_flatten(seq_list):
    encodings = batch_encode(seq_list)
    return batch_flatten(encodings)