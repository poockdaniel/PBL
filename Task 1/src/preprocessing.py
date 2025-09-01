import numpy as np
import pandas as pd
from protlearn import preprocessing

amino_acids = "ACDEFGHIKLMNPQRSTVWY"

# One Hot encodes a given sequence padding up to a specified max_len
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

# One Hot encodes the given sequence list
def batch_encode(seq_list):
    max_len = max(len(seq) for seq in seq_list)
    encoding_list = [encode(seq, max_len) for seq in seq_list]
    return np.stack(encoding_list)

# Flattens the given batch One Hot encodings to a 2-dimensional vector with size batch size x (residue count * max. seq length)
def batch_flatten(encodings):
    if not isinstance(encodings, np.ndarray):
        raise Exception(f"Expected numpy array, got {type(encodings)}!")
    batch_size, seq_len, n_residues = encodings.shape
    return encodings.reshape(batch_size, seq_len * n_residues)

# One Hot encodes and flattens the given sequence list
def batch_encode_and_flatten(seq_list):
    encodings = batch_encode(seq_list)
    return batch_flatten(encodings)

# Add column 'Enzyme' (0 = No, 1 = Yes) for easier binary classification
def add_enz_flag(df):
    return df.assign(Enzyme=df['EC number'].apply(lambda x : 1 if isinstance(x, str) else 0))

# Removes duplicate sequences so that the representative sequence counts as an enzyme if at least one of the entries with that sequence was an enzyme
def remove_duplicate_seqs(df):
    return df.drop_duplicates('Sequence', keep=False)

# Truncates DataFrame to specified columns
def truncate(df, columns):
    if set(columns).issubset(set(df.columns)):
        return df.loc[:, columns]
    raise Exception("Dataframe doesn't contain given column names!")

# Removes sequences with length under the i-th and over the j-th percentile
def filter_lengths(df, i, j):
    len_list = [len(s) for s in df['Sequence']]
    P1 = np.percentile(len_list, i)
    P2 = np.percentile(len_list, j)
    return df[(df['Sequence'].str.len() >= P1) & (df['Sequence'].str.len() <= P2)]

# Return a DataFrame object that contains no unnatural aminoacid sequences, Implicitly removes duplicate sequences as well
def remove_unnatural_seqs(df):
    seq_list = [seq for seq in PreProcessingTask1.remove_duplicate_seqs(df)['Sequence']]
    natural = preprocessing.remove_unnatural(seq_list)
    nat_df = pd.DataFrame(natural, columns=['Sequence'])
    return df.merge(nat_df, on='Sequence')

# Given .tsv file from MMseqs2 clustering, returns the DataFrame that results when only taking the representative sequences from the given DataFrame
def cluster_df(df, cluster_tsv_path):
    clusters = pd.read_csv(cluster_tsv_path, sep='\t', header=None, names=['Representative', 'Member'])
    df_rep = PreProcessingTask1.truncate(clusters, ['Representative']).drop_duplicates().reset_index(drop=True)
    if 'Entry' in df.columns:
        temp = df.merge(df_rep, left_on='Entry', right_on='Representative')
        return PreProcessingTask1.truncate(temp, ['Entry', 'Entry Name', 'Sequence', 'Enzyme'])
    raise Exception("DataFrame doesn't contain column 'Entry'.")

# Removes the first num_e entries from the non-enzyme entries to achieve equal class counts
def undersample(df):
    if 'Enzyme' in df.columns:
        non_enzymes = df[df['Enzyme'] == 0]
        enzymes = df[df['Enzyme'] == 1]
        min_size = min(len(non_enzymes), len(enzymes))
        rand_non_enzymes = non_enzymes.sample(min_size)
        rand_enzymes = enzymes.sample(min_size)
        return pd.concat([rand_non_enzymes, rand_enzymes]).reset_index(drop=True)
    raise Exception("DataFrame doesn't contain column 'Enzyme'.")

# Removes n random entries from the specified DataFrame object
def remove_random(df, n):
    return df.drop(df.sample(n).index)


    
