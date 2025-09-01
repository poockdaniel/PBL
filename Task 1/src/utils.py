import matplotlib.pyplot as plt 
import seaborn as sns

# Shows distribution of the lengths of sequences in dataset as histogram
def length_distribution(df, name=''):
    len_list = [len(s) for s in df['Sequence']]
    sns.histplot(len_list, bins=50, kde=True)
    plt.title("Distribution of Sequence Lengths in Dataset " + name)
    plt.xlabel("Length")
    plt.ylabel("Count")
    plt.show()

# Graphs the distribution of non-enzyme and enzyme entries in the given DataFrame as a bar chart
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
def df_to_tsv(df, path):
    df.to_csv(path, sep='\t', index=False)