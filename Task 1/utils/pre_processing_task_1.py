import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt 
import seaborn as sns
from protlearn import preprocessing

class PreProcessingTask1:   
    # Add column 'Enzyme' (0 = No, 1 = Yes) for easier binary classification
    @staticmethod
    def add_enz_flag(df):
        return df.assign(Enzyme=df['EC number'].apply(lambda x : 1 if isinstance(x, str) else 0))

    # Removes duplicate sequences so that the representative sequence counts as an enzyme if at least one of the entries with that sequence was an enzyme
    @staticmethod
    def remove_duplicate_seqs(df):
        return df.drop_duplicates('Sequence', keep=False)

    # Truncates DataFrame to specified columns
    @staticmethod
    def truncate(df, columns):
        if set(columns).issubset(set(df.columns)):
            return df.loc[:, columns]
        raise Exception("Dataframe doesn't contain given column names!")

    # Removes sequences with length under the i-th and over the j-th percentile
    @staticmethod
    def filter_lengths(df, i, j):
        len_list = [len(s) for s in df['Sequence']]
        P1 = np.percentile(len_list, i)
        P2 = np.percentile(len_list, j)
        return df[(df['Sequence'].str.len() >= P1) & (df['Sequence'].str.len() <= P2)]

    # Return a DataFrame object that contains no unnatural aminoacid sequences, Implicitly removes duplicate sequences as well
    @staticmethod
    def remove_unnatural_seqs(df):
        seq_list = [seq for seq in PreProcessingTask1.remove_duplicate_seqs(df)['Sequence']]
        natural = preprocessing.remove_unnatural(seq_list)
        nat_df = pd.DataFrame(natural, columns=['Sequence'])
        return df.merge(nat_df, on='Sequence')
    
    # Given .tsv file from MMseqs2 clustering, returns the DataFrame that results when only taking the representative sequences from the given DataFrame
    @staticmethod
    def cluster_df(df, cluster_tsv_path):
        clusters = pd.read_csv(cluster_tsv_path, sep='\t', header=None, names=['Representative', 'Member'])
        df_rep = PreProcessingTask1.truncate(clusters, ['Representative']).drop_duplicates().reset_index(drop=True)
        if 'Entry' in df.columns:
            temp = df.merge(df_rep, left_on='Entry', right_on='Representative')
            return PreProcessingTask1.truncate(temp, ['Entry', 'Entry Name', 'Sequence', 'Enzyme'])
        raise Exception("DataFrame doesn't contain column 'Entry'.")

    # Removes the first num_e entries from the non-enzyme entries to achieve equal class counts
    @staticmethod
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
    @staticmethod
    def remove_random(df, n):
        return df.drop(df.sample(n).index)
    
    # Shows distribution of the lengths of sequences in dataset as histogram
    @staticmethod
    def length_distribution(df, name=''):
        len_list = [len(s) for s in df['Sequence']]
        sns.histplot(len_list, bins=50, kde=True)
        plt.title("Distribution of Sequence Lengths in Dataset " + name)
        plt.xlabel("Length")
        plt.ylabel("Count")
        plt.show()
        
    # Graphs the distribution of non-enzyme and enzyme entries in the given DataFrame as a bar chart
    @staticmethod
    def class_distribution(df, name=''):     
        non_enzymes = [n for n in df['Enzyme'] if n == 0]
        enzymes = [n for n in df['Enzyme'] if n == 1]
        keys = ['Enzyme Entries', 'Non-enzyme entries']
        values = [len(enzymes), len(non_enzymes)]
        bars = plt.bar(keys, values)
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height, str(height), ha='center', va='bottom')
        plt.ylabel('Protein Count')
        plt.title('Distribution of Enzyme and Non-Enzyme Protein Entries in Dataset ' + name)
        plt.show()

    # Writes given DataFrame to a .fasta file at the given path 
    @staticmethod
    def df_to_fasta(df, path):
        if set(['Entry', 'Entry Name', 'Sequence', 'Enzyme']).issubset(set(df.columns)):
            with open(path, 'w') as outfile:
                columns = df.columns
                ent_idx = list(columns).index('Entry') + 1
                name_idx = list(columns).index('Entry Name') + 1
                seq_idx = list(columns).index('Sequence') + 1
                ez_idx = list(columns).index('Enzyme') + 1

                for row in df.itertuples():
                    ez = (row[ez_idx] == 1)
                    outfile.write('>sp|' + row[ent_idx] + '|' + row[name_idx] + ' ' + 'Enzyme=' + str(ez) + '\n')
                    outfile.write(row[seq_idx] + '\n')
            return
        raise Exception("DataFrame doesn't contain the necessary columns.")

    # Writes given DataFrame to a .tsv file at the given path 
    @staticmethod
    def df_to_tsv(df, path):
        df.to_csv(path, sep='\t', index=False)