from src.crossval import cross_validate
import src.preprocessing as pp
from src.models import Task1Model
import torch.nn as nn
import torch.optim as optim
import torch
from pathlib import Path

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embedding_path = Path.cwd() / Path('Task 1/embeddings/embeddings_exp_proteins_clustered_70_filter_lens_25-75_undersampled_s35526.npy')
    labels_path = Path.cwd() / Path('Task 1/data/preprocessed/exp_proteins_clustered_70_filter_lens_25-75_undersampled.tsv')

    dataset = pp.create_dataset(embedding_path, labels_path, 'exp_proteins_clustered_70_filter_lens_25-75_undersampled')
    model = Task1Model(dims=[1024, 683, 683, 1], dropouts=[0.5, 0.3], activation=nn.LeakyReLU(), normalize=False).to(device=device)
    criterion = nn.BCELoss()
    cross_validate(model_in=model, dataset=dataset, optimizer_fn=optim.SGD, weight_decay=0.0012, loss_fn=criterion, n_splits=5, batch_size=32, lr=0.017, epochs=2)
    