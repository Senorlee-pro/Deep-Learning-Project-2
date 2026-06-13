import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        correct += out.argmax(dim=1).eq(y).sum().item()
        total += y.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)

        running_loss += loss.item() * x.size(0)
        correct += out.argmax(dim=1).eq(y).sum().item()
        total += y.size(0)

    return running_loss / total, correct / total


def train_model(model, train_loader, val_loader, optimizer, criterion,
                epochs, device, scheduler=None, verbose=True):
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [],   'val_acc': [],
        'epoch_time': [],
    }
    best_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        elapsed = time.time() - t0

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['epoch_time'].append(elapsed)

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if verbose:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"train loss: {train_loss:.4f} | train acc: {train_acc:.4f} | "
                  f"val loss: {val_loss:.4f} | val acc: {val_acc:.4f} | "
                  f"time: {elapsed:.1f}s")

    model.load_state_dict(best_state)
    return history, best_acc


class CrossEntropyWithL1(nn.Module):
    def __init__(self, model, l1_lambda=1e-5):
        super().__init__()
        self.model = model
        self.l1_lambda = l1_lambda
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        ce_loss = self.ce(pred, target)
        l1 = sum(p.abs().sum() for p in self.model.parameters())
        return ce_loss + self.l1_lambda * l1


class CrossEntropyWithL2(nn.Module):
    def __init__(self, model, l2_lambda=1e-4):
        super().__init__()
        self.model = model
        self.l2_lambda = l2_lambda
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        ce_loss = self.ce(pred, target)
        l2 = sum(p.pow(2).sum() for p in self.model.parameters())
        return ce_loss + self.l2_lambda * l2


def get_optimizer(name, model, lr, **kwargs):
    name = name.lower()
    if name == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr,
                                weight_decay=kwargs.get('weight_decay', 0))
    elif name == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr,
                               momentum=kwargs.get('momentum', 0.9),
                               weight_decay=kwargs.get('weight_decay', 5e-4))
    elif name == 'rmsprop':
        return torch.optim.RMSprop(model.parameters(), lr=lr,
                                   weight_decay=kwargs.get('weight_decay', 0))
    elif name == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr,
                                 weight_decay=kwargs.get('weight_decay', 1e-2))
    else:
        raise ValueError(f"Unknown optimizer: {name}")
