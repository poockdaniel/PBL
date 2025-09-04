import matplotlib.pyplot as plt 
import seaborn as sns
import yaml
import torch.optim as optim
import os

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

def save_config(path, model, dataset=None, optimizer=None, loss_fn=None, epochs=None):
    config = {
        'model': {
            'model_type': type(model).__name__,
            'dims': model.dims,
            'dropouts': model.dropouts,
            'activation': {
                'name': type(model.activation).__name__,
                'negative_slope': model.activation.negative_slope
            },
            'normalize': model.normalize, 
            'dtype': str(model.dtype)
        },
        'training': {
            'optimizer': {
                'name': type(optimizer).__name__,
                'lr': optimizer.param_groups[0]['lr'],
                'weight_decay': optimizer.param_groups[0]['weight_decay']
            },
            'loss_fn': type(loss_fn).__name__,
            'epochs': epochs
        },
        'dataset': {
            'name': dataset.name,
            'size': len(dataset),
            'feature_dim': str(tuple((dataset[0][0].shape)))
        }
    }
    file_path = os.path.join(path, f'{os.path.basename(path)}_config.yaml')
    with open(file_path, 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    return config

def create_run_folder():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_path = os.path.join(base_dir, 'logs')
    os.makedirs(log_path, exist_ok=True)
    path = os.path.join(log_path, f'run_{len(os.listdir(log_path))}')
    os.makedirs(path, exist_ok=True)
    return path

def visualize_cv_training(path, tlm, vlm, tam, vam, num_epochs):
    file_path_loss = os.path.join(path, f'{os.path.basename(path)}_loss.png')
    file_path_acc = os.path.join(path, f'{os.path.basename(path)}_acc.png')
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, tlm, label='Training loss')
    plt.plot(epochs, vlm, label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Average training vs validation loss over splits')
    plt.legend()
    plt.grid(True)
    plt.savefig(file_path_loss)
    plt.show()

    plt.plot(epochs, tam, label='Training accuracy')
    plt.plot(epochs, vam, label='Validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Average training vs validation Accuracy over splits')
    plt.legend()
    plt.grid(True)
    plt.savefig(file_path_acc)
    plt.show()

def log_training(path, msg):
    msg_path = os.path.join(path, f'{os.path.basename(path)}_cv_training.log')
    with open(msg_path, 'w') as f:
        f.write(msg)

def log_metrics(path, metrics):
    file_path = os.path.join(path, f'{os.path.basename(path)}_metrics.yaml')
    with open(file_path, 'w') as f:
        yaml.dump(metrics, f, sort_keys=False)
