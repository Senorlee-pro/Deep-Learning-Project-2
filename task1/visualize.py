import os
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torchvision
import torchvision.transforms as transforms

from models import CifarCNN
from train import evaluate


plt.rcParams.update({
    'figure.figsize': (10, 6),
    'figure.dpi': 100,
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
})


def plot_training_curves(histories, labels, save_path='training_curves.png'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for hist, label in zip(histories, labels):
        axes[0].plot(hist['train_loss'], alpha=0.6, label=f'{label} (train)')
        axes[0].plot(hist['val_loss'], alpha=0.6, linestyle='--',
                     label=f'{label} (val)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend(fontsize=8)

    for hist, label in zip(histories, labels):
        axes[1].plot(hist['train_acc'], alpha=0.6, label=f'{label} (train)')
        axes[1].plot(hist['val_acc'], alpha=0.6, linestyle='--',
                     label=f'{label} (val)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Curves')
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved training curves to {save_path}")


def visualize_filters(model, layer_name='stages.0.0.conv1', save_path='filters.png'):
    parts = layer_name.split('.')
    module = model
    for p in parts:
        module = getattr(module, p)

    weights = module.weight.data.cpu()
    out_ch, in_ch, kh, kw = weights.shape

    n_filters = min(out_ch, 64)
    grid_size = int(np.ceil(np.sqrt(n_filters)))

    w_min, w_max = weights.min(), weights.max()
    weights_norm = (weights - w_min) / (w_max - w_min + 1e-8)

    fig, axes = plt.subplots(grid_size, grid_size,
                              figsize=(12, 12))
    for i in range(n_filters):
        row, col = i // grid_size, i % grid_size
        ax = axes[row, col] if grid_size > 1 else axes

        filt = weights_norm[i]
        if in_ch == 3:
            img = filt.permute(1, 2, 0).numpy()
        else:
            img = filt[0].numpy()

        ax.imshow(img, cmap='gray' if in_ch != 3 else None)
        ax.axis('off')

    for i in range(n_filters, grid_size * grid_size):
        row, col = i // grid_size, i % grid_size
        axes[row, col].axis('off')

    plt.suptitle(f'Conv Filters — {layer_name} ({out_ch} filters)', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved filter visualization to {save_path}")


def plot_loss_landscape(model, loader, criterion, device,
                        save_path='loss_landscape.png', n_steps=51, alpha=1.0):
    model.eval()

    original = {k: v.clone() for k, v in model.state_dict().items()
                if v.dtype == torch.float32}

    direction = {}
    for k, v in original.items():
        direction[k] = torch.randn_like(v)

    for k in original:
        direction[k] = direction[k] * (original[k].norm() /
                                       (direction[k].norm() + 1e-8))

    alphas = np.linspace(-alpha, alpha, n_steps)
    losses = []

    for a in alphas:
        for k in original:
            model.state_dict()[k].copy_(original[k] + a * direction[k])

        loss_sum, n = 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                out = model(x)
                loss = criterion(out, y)
            loss_sum += loss.item() * x.size(0)
            n += y.size(0)
            if n > 500:
                break
        losses.append(loss_sum / n)

    for k in original:
        model.state_dict()[k].copy_(original[k])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(alphas, losses, linewidth=2)
    ax.fill_between(alphas, min(losses), losses, alpha=0.2)
    ax.set_xlabel('Perturbation α')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Landscape (1D slice)')
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved loss landscape to {save_path}")


@torch.no_grad()
def plot_feature_embedding(model, loader, device, save_path='tsne.png',
                           method='tsne', max_samples=2000):
    model.eval()

    features = []
    labels = []

    hook = model.global_pool.register_forward_hook(
        lambda m, inp, out: features.append(out.cpu()))

    for x, y in loader:
        x = x.to(device)
        labels.append(y)
        model(x)
        if sum(f.size(0) for f in features) >= max_samples:
            break

    hook.remove()

    feats = torch.cat(features, dim=0)[:max_samples]
    feats = feats.view(feats.size(0), -1).numpy()
    lbls = torch.cat(labels, dim=0)[:max_samples].numpy()

    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    else:
        reducer = PCA(n_components=2)
    emb = reducer.fit_transform(feats)

    fig, ax = plt.subplots(figsize=(10, 8))
    classes = ['airplane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for c in range(10):
        mask = lbls == c
        ax.scatter(emb[mask, 0], emb[mask, 1], c=[colors[c]],
                   label=classes[c], alpha=0.6, s=10)
    ax.legend(markerscale=3, fontsize=9)
    ax.set_title(f'Feature Embedding ({method.upper()})')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved feature embedding to {save_path}")


def plot_comparison(summary_path='experiment_results/summary.json',
                    save_path='comparison.png'):
    with open(summary_path, 'r') as f:
        data = json.load(f)

    data = sorted(data, key=lambda x: x['best_acc'], reverse=True)
    names = [d['name'] for d in data]
    accs = [d['best_acc'] for d in data]
    params = [d['n_params'] for d in data]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(range(len(names)), accs, color=plt.cm.viridis(
        np.linspace(0.2, 0.9, len(names))))

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([f"{n} ({p:,} params)" for n, p in zip(names, params)],
                       fontsize=9)
    ax.set_xlabel('Best Test Accuracy')
    ax.set_title('CIFAR-10 Experiment Comparison')
    ax.invert_yaxis()

    for i, (acc, bar) in enumerate(zip(accs, bars)):
        ax.text(acc + 0.002, bar.get_y() + bar.get_height()/2,
                f'{acc:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved comparison chart to {save_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint .pt')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['curves', 'filters', 'landscape', 'tsne',
                                 'comparison', 'all'])
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    device = torch.device(args.device)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    test_set = torchvision.datasets.CIFAR10(
        root='../data', train=False, download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=128, shuffle=False, num_workers=2)

    criterion = torch.nn.CrossEntropyLoss()

    if args.checkpoint and args.mode in ('filters', 'landscape', 'tsne', 'all'):
        ckpt = torch.load(args.checkpoint, map_location=device)
        config = ckpt['config']
        model = CifarCNN(
            filters=config['filters'],
            num_blocks=config['num_blocks'],
            dense_units=config['dense_units'],
            activation=config['activation'],
            dropout=config['dropout'],
            use_bn=config.get('use_bn', True),
            use_residual=config.get('use_residual', True),
            num_classes=10,
        ).to(device)
        model.load_state_dict(ckpt['state_dict'])

        if args.mode in ('filters', 'all'):
            visualize_filters(model, save_path='filters.png')
        if args.mode in ('landscape', 'all'):
            plot_loss_landscape(model, test_loader, criterion, device,
                                save_path='loss_landscape_task1.png', n_steps=31)
        if args.mode in ('tsne', 'all'):
            plot_feature_embedding(model, test_loader, device,
                                   save_path='tsne.png')

    if args.mode in ('comparison', 'all'):
        if os.path.exists('experiment_results/summary.json'):
            plot_comparison()
        else:
            print("No summary.json found — run main.py first.")
