import torch
from torch.utils.data import DataLoader, Subset
from torch import nn, optim
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    total_acc = 0
    for feats, labels in loader:
        optimizer.zero_grad()
        outputs = model(feats)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        with torch.no_grad():
            outputs_rounded = np.round(outputs)
            total_acc += accuracy_score(labels, outputs_rounded)
    avg_loss = np.round(total_loss / len(loader), 4)
    avg_acc = np.round(total_acc / len(loader), 4)
    return avg_loss, avg_acc

def evaluate(model, loader, loss_fn):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        conf = np.zeros([2, 2])
        for feats, labels in loader:
            preds = model(feats)
            loss = loss_fn(preds, labels)
            total_loss += loss.item()
            conf += confusion_matrix(np.round(preds), labels, labels=[0, 1])
        avg_loss = total_loss / len(loader)
        tn, fp, fn, tp = conf.ravel().tolist()
        acc = np.round((tn + tp) / len(loader.dataset), 4)
        return avg_loss, acc, tp, fn, fp, tn

def cross_validate(model_fn, dataset, optimizer_fn, loss_fn, n_splits, batch_size, learning_rate, epochs):
    np.set_printoptions(suppress=True)
    train_loss_tracking = []
    val_loss_tracking = []
    train_acc_tracking = []
    val_acc_tracking = []
    results = []
    skfold = StratifiedKFold(n_splits=n_splits, shuffle=True)
    
    for fold, (train_idx, val_idx) in enumerate(skfold.split(dataset.X, dataset.y)):
        print(f'Fold {fold + 1}')
        train_set = Subset(dataset, train_idx)
        val_set = Subset(dataset, val_idx)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

        train_loss_list = []
        val_loss_list = []
        train_acc_list = []
        val_acc_list = []

        model = model_fn()
        optimizer = optimizer_fn(model.parameters(), lr=learning_rate)
        
        for e in range(epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_fn)
            val_loss, val_acc, tp, fn, fp, tn = evaluate(model, val_loader, loss_fn)
            print(f'Train loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}, Validation acc: {val_acc:.4f}')
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            train_acc_list.append(train_acc)
            val_acc_list.append(val_acc)
        train_loss_tracking.append(train_loss_list)
        val_loss_tracking.append(val_loss_list)
        train_acc_tracking.append(train_acc_list)
        val_acc_tracking.append(val_acc_list)

        results.append([val_acc, tp, fn, fp, tn])

    train_loss_means = np.round(np.stack(train_loss_tracking).mean(axis=0), 4)
    val_loss_means = np.round(np.stack(val_loss_tracking).mean(axis=0), 4)
    train_acc_means = np.round(np.stack(train_acc_tracking).mean(axis=0), 4)
    val_acc_means = np.round(np.stack(val_acc_tracking).mean(axis=0), 4)
    visualize_training(train_loss_means, val_loss_means, train_acc_means, val_acc_means, epochs)
    print(f'5 {results}')
    return np.array(results, dtype=np.float64)

def visualize_training(tlm, vlm, tam, vam, num_epochs):
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, tlm, label='Training Loss')
    plt.plot(epochs, vlm, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Average Training vs Validation Loss over Splits')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.plot(epochs, tam, label='Training Accuracy')
    plt.plot(epochs, vam, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Average Training vs Validation Accuracy over Splits')
    plt.legend()
    plt.grid(True)
    plt.show()