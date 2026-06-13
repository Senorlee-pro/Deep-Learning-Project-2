import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import random
from tqdm import tqdm as tqdm

from models.vgg import VGG_A, VGG_A_BatchNorm
from data.loaders import get_cifar_loader
device_id = [0, 1, 2, 3]
num_workers = 4
batch_size = 128

module_path = os.path.dirname(os.getcwd()) if os.getcwd().endswith('VGG_BatchNorm') else os.getcwd()
home_path = module_path
figures_path = os.path.join(home_path, 'reports', 'figures')
models_path = os.path.join(home_path, 'reports', 'models')
os.makedirs(figures_path, exist_ok=True)
os.makedirs(models_path, exist_ok=True)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


@torch.no_grad()
def get_accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        correct += out.argmax(dim=1).eq(y).sum().item()
        total += y.size(0)
    return correct / total


def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu':
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train(model, optimizer, criterion, train_loader, val_loader,
          scheduler=None, epochs_n=100, best_model_path=None,
          record_every=1):
    model.to(device)
    learning_curve = [np.nan] * epochs_n
    train_accuracy_curve = [np.nan] * epochs_n
    val_accuracy_curve = [np.nan] * epochs_n
    max_val_accuracy = 0
    max_val_accuracy_epoch = 0

    batches_n = len(train_loader)
    losses_list = []
    grads = []

    for epoch in range(epochs_n):
        if scheduler is not None:
            scheduler.step()
        model.train()

        epoch_loss_sum = 0.0
        loss_list = []
        grad_list = []

        for data in train_loader:
            x, y = data
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)

            loss_list.append(loss.item())
            epoch_loss_sum += loss.item()

            loss.backward()

            grad_norm = model.classifier[-1].weight.grad.norm().item()
            grad_list.append(grad_norm)

            optimizer.step()

        losses_list.append(loss_list)
        grads.append(grad_list)

        learning_curve[epoch] = epoch_loss_sum / batches_n

        train_acc = get_accuracy(model, train_loader, device)
        train_accuracy_curve[epoch] = train_acc

        val_acc = get_accuracy(model, val_loader, device)
        val_accuracy_curve[epoch] = val_acc

        if val_acc > max_val_accuracy:
            max_val_accuracy = val_acc
            max_val_accuracy_epoch = epoch
            if best_model_path is not None:
                torch.save(model.state_dict(), best_model_path)

        print(f"Epoch {epoch+1:3d}/{epochs_n} | "
              f"loss: {learning_curve[epoch]:.4f} | "
              f"train acc: {train_acc:.4f} | val acc: {val_acc:.4f}")

    return (losses_list, grads, learning_curve,
            train_accuracy_curve, val_accuracy_curve, max_val_accuracy)


def compute_loss_landscape(model_cls, lr_list, train_loader, val_loader,
                           epochs_n=30, seed=2020, label='model'):
    all_loss_curves = []

    for lr in lr_list:
        print(f"\n--- {label} | lr={lr} ---")
        set_random_seeds(seed_value=seed, device=str(device))

        model = model_cls()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        _, _, learning_curve, _, _, best_acc = train(
            model, optimizer, criterion,
            train_loader, val_loader,
            epochs_n=epochs_n,
            best_model_path=os.path.join(models_path,
                                         f'{label}_lr{lr}_best.pt'),
        )
        all_loss_curves.append(learning_curve)
        print(f"  Best val acc: {best_acc:.4f}")

    min_len = min(len(c) for c in all_loss_curves)
    all_loss_curves = [c[:min_len] for c in all_loss_curves]
    all_loss_curves = np.array(all_loss_curves)

    max_curve = np.max(all_loss_curves, axis=0)
    min_curve = np.min(all_loss_curves, axis=0)

    return max_curve, min_curve, all_loss_curves


def plot_loss_landscape_comparison(max_no_bn, min_no_bn, max_bn, min_bn,
                                   lr_list, save_path=None):
    epochs = range(1, len(max_no_bn) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, max_no_bn, label='max loss', color='red', alpha=0.8)
    axes[0].plot(epochs, min_no_bn, label='min loss', color='blue', alpha=0.8)
    axes[0].fill_between(epochs, min_no_bn, max_no_bn, alpha=0.2, color='purple')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Training Loss')
    axes[0].set_title(f'VGG-A WITHOUT BN\nLRs: {lr_list}')
    axes[0].legend()

    axes[1].plot(epochs, max_bn, label='max loss', color='red', alpha=0.8)
    axes[1].plot(epochs, min_bn, label='min loss', color='blue', alpha=0.8)
    axes[1].fill_between(epochs, min_bn, max_bn, alpha=0.2, color='purple')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Training Loss')
    axes[1].set_title(f'VGG-A WITH BN\nLRs: {lr_list}')
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved loss landscape comparison to {save_path}")
    plt.close()


def plot_combined_loss_range(max_no_bn, min_no_bn, max_bn, min_bn,
                             save_path=None):
    epochs = range(1, len(max_no_bn) + 1)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.fill_between(epochs, min_no_bn, max_no_bn,
                    alpha=0.3, color='red', label='Without BN')
    ax.plot(epochs, max_no_bn, color='red', alpha=0.7)
    ax.plot(epochs, min_no_bn, color='red', alpha=0.7)

    ax.fill_between(epochs, min_bn, max_bn,
                    alpha=0.3, color='blue', label='With BN')
    ax.plot(epochs, max_bn, color='blue', alpha=0.7)
    ax.plot(epochs, min_bn, color='blue', alpha=0.7)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Loss Landscape: VGG-A With vs Without BatchNorm')
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved combined loss range to {save_path}")
    plt.close()


def plot_accuracy_comparison(acc_no_bn, acc_bn, save_path=None):
    epochs = range(1, len(acc_no_bn) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, acc_no_bn, label='Without BN', linewidth=2)
    ax.plot(epochs, acc_bn, label='With BN', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('VGG-A Accuracy: With vs Without BatchNorm')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved accuracy comparison to {save_path}")
    plt.close()


def main():
    train_loader = get_cifar_loader(root='../../data/', train=True,
                                     batch_size=batch_size, num_workers=0)
    val_loader = get_cifar_loader(root='../../data/', train=False,
                                   batch_size=batch_size, num_workers=0)

    for X, y in train_loader:
        print(f"Batch shape: {X.shape}, labels: {y[:5]}")
        break

    lr_list = [1e-3, 2e-3, 5e-4, 1e-4]

    print("\n" + "=" * 60)
    print("Training VGG-A WITHOUT BatchNorm at multiple LRs ...")
    print("=" * 60)
    max_no_bn, min_no_bn, all_no_bn = compute_loss_landscape(
        VGG_A, lr_list, train_loader, val_loader,
        epochs_n=30, label='vgg_a_no_bn')

    print("\n" + "=" * 60)
    print("Training VGG-A WITH BatchNorm at multiple LRs ...")
    print("=" * 60)
    max_bn, min_bn, all_bn = compute_loss_landscape(
        VGG_A_BatchNorm, lr_list, train_loader, val_loader,
        epochs_n=30, label='vgg_a_bn')

    plot_loss_landscape_comparison(
        max_no_bn, min_no_bn, max_bn, min_bn, lr_list,
        save_path=os.path.join(figures_path, 'loss_landscape_comparison.png'))

    plot_combined_loss_range(
        max_no_bn, min_no_bn, max_bn, min_bn,
        save_path=os.path.join(figures_path, 'loss_landscape_combined.png'))

    print("\n" + "=" * 60)
    print("Training final models (no BN vs BN) with lr=1e-3 ...")
    print("=" * 60)

    best_lr = 1e-3
    set_random_seeds(seed_value=2020, device=str(device))

    model_no_bn = VGG_A()
    optimizer_no_bn = torch.optim.Adam(model_no_bn.parameters(), lr=best_lr)
    criterion = nn.CrossEntropyLoss()
    _, _, _, _, val_acc_no_bn, best_no_bn = train(
        model_no_bn, optimizer_no_bn, criterion,
        train_loader, val_loader, epochs_n=40,
        best_model_path=os.path.join(models_path, 'vgg_a_no_bn_best.pt'))

    set_random_seeds(seed_value=2020, device=str(device))
    model_bn = VGG_A_BatchNorm()
    optimizer_bn = torch.optim.Adam(model_bn.parameters(), lr=best_lr)
    _, _, _, _, val_acc_bn, best_bn = train(
        model_bn, optimizer_bn, criterion,
        train_loader, val_loader, epochs_n=40,
        best_model_path=os.path.join(models_path, 'vgg_a_bn_best.pt'))

    plot_accuracy_comparison(
        val_acc_no_bn, val_acc_bn,
        save_path=os.path.join(figures_path, 'accuracy_comparison.png'))

    print(f"\nFinal results:")
    print(f"  VGG-A without BN — best val acc: {best_no_bn:.4f}")
    print(f"  VGG-A with BN    — best val acc: {best_bn:.4f}")

    np.savez(os.path.join(models_path, 'loss_landscape_data.npz'),
             max_no_bn=max_no_bn, min_no_bn=min_no_bn,
             max_bn=max_bn, min_bn=min_bn,
             all_no_bn=all_no_bn, all_bn=all_bn,
             val_acc_no_bn=val_acc_no_bn, val_acc_bn=val_acc_bn,
             lr_list=lr_list)

    print("\nDone. All results saved to reports/figures/ and reports/models/")


if __name__ == '__main__':
    main()
