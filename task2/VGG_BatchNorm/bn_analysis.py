import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import os

from models.vgg import VGG_A, VGG_A_BatchNorm
from data.loaders import get_cifar_loader
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

module_path = os.path.dirname(os.path.abspath(__file__))
codes_path = os.path.dirname(module_path)
figures_path = os.path.join(module_path, 'reports', 'figures')
models_path = os.path.join(codes_path, 'reports', 'models')
os.makedirs(figures_path, exist_ok=True)


def flatten_grads(model):
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.data.view(-1))
        else:
            grads.append(torch.zeros_like(p.data.view(-1)))
    return torch.cat(grads)


def flatten_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def set_params_from_flat(model, flat_vector):
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat_vector[offset:offset + n].view_as(p))
        offset += n


def load_checkpoint(model, checkpoint_path):
    if checkpoint_path is None:
        return
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    model.load_state_dict(checkpoint)


def compute_loss(model, loader, criterion, max_batches=10):
    model.eval()
    total_loss, count = 0.0, 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            count += y.size(0)
            if i >= max_batches - 1:
                break
    return total_loss / count


def compute_loss_and_grad(model, loader, criterion, max_batches=10,
                          train_mode=False):
    if train_mode:
        model.train()
    else:
        model.eval()
    total_loss, count = 0.0, 0
    model.zero_grad()

    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        (loss * x.size(0)).backward()
        total_loss += loss.item() * x.size(0)
        count += y.size(0)
        if i >= max_batches - 1:
            break

    for p in model.parameters():
        if p.grad is not None:
            p.grad.data.div_(count)

    avg_loss = total_loss / count
    grad_vec = flatten_grads(model).clone()

    model.zero_grad()
    return avg_loss, grad_vec


def gradient_predictiveness(model_cls, loader, criterion, device,
                            checkpoint_path=None, max_batches=10,
                            n_steps=21, max_eta=0.003):
    model = model_cls().to(device)
    load_checkpoint(model, checkpoint_path)
    model.eval()

    loss_0, grad_vec = compute_loss_and_grad(model, loader, criterion,
                                              max_batches, train_mode=False)
    flat_0 = flatten_params(model).clone()
    grad_norm = grad_vec.norm().item()

    etas = np.linspace(-max_eta, max_eta, n_steps)
    predicted = []
    actual = []

    for eta in etas:
        new_flat = flat_0 - eta * grad_vec.to(flat_0.device)
        set_params_from_flat(model, new_flat)

        loss_new = compute_loss(model, loader, criterion, max_batches)

        loss_pred = -eta * (grad_norm ** 2)

        predicted.append(loss_pred)
        actual.append(loss_new - loss_0)

    set_params_from_flat(model, flat_0)

    return etas, np.array(predicted), np.array(actual)


def beta_smoothness(model_cls, loader, criterion, device,
                    checkpoint_path=None, max_batches=10, n_steps=21,
                    max_alpha=0.5):
    model = model_cls().to(device)
    load_checkpoint(model, checkpoint_path)
    model.eval()

    _, grad_0 = compute_loss_and_grad(model, loader, criterion, max_batches,
                                      train_mode=False)
    flat_0 = flatten_params(model).clone()
    grad_norm_0 = grad_0.norm().item()
    g_dir = grad_0 / (grad_norm_0 + 1e-10)

    alphas = np.linspace(0, max_alpha, n_steps)
    smoothness = []

    for alpha in alphas:
        param_norm = flat_0.norm().item()
        perturbation = alpha * param_norm * g_dir.to(flat_0.device)
        new_flat = flat_0 - perturbation
        set_params_from_flat(model, new_flat)

        _, grad_new = compute_loss_and_grad(model, loader, criterion,
                                             max_batches, train_mode=False)

        grad_diff_norm = (grad_new.to(flat_0.device) - grad_0.to(flat_0.device)).norm().item()
        pert_norm = perturbation.norm().item()

        if pert_norm > 1e-10:
            smoothness.append(grad_diff_norm / pert_norm)
        else:
            smoothness.append(0.0)

    set_params_from_flat(model, flat_0)

    return alphas[1:], np.array(smoothness[1:])


def plot_gradient_predictiveness(etas, pred_no_bn, act_no_bn,
                                  pred_bn, act_bn, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(etas, pred_no_bn, 'b--', label='Predicted (1st-order Taylor)',
                 linewidth=2)
    axes[0].plot(etas, act_no_bn, 'r-', label='Actual loss', linewidth=2)
    axes[0].fill_between(etas, pred_no_bn, act_no_bn, alpha=0.15, color='purple')
    axes[0].axvline(0, color='gray', linestyle=':', alpha=0.5)
    axes[0].set_xlabel('Step size η (along gradient direction)')
    axes[0].axhline(0, color='gray', linestyle=':', alpha=0.5)
    axes[0].set_ylabel('Loss change ΔL')
    axes[0].set_title('VGG-A WITHOUT BN')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(etas, pred_bn, 'b--', label='Predicted (1st-order Taylor)',
                 linewidth=2)
    axes[1].plot(etas, act_bn, 'r-', label='Actual loss', linewidth=2)
    axes[1].fill_between(etas, pred_bn, act_bn, alpha=0.15, color='purple')
    axes[1].axvline(0, color='gray', linestyle=':', alpha=0.5)
    axes[1].set_xlabel('Step size η (along gradient direction)')
    axes[1].axhline(0, color='gray', linestyle=':', alpha=0.5)
    axes[1].set_ylabel('Loss change ΔL')
    axes[1].set_title('VGG-A WITH BN')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Gradient Predictiveness: How well does gradient predict nearby loss?',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved gradient predictiveness to {save_path}")
    plt.close()


def plot_gradient_predictiveness_combined(etas, pred_no_bn, act_no_bn,
                                           pred_bn, act_bn, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 6))

    error_no_bn = np.abs(act_no_bn - pred_no_bn)
    error_bn = np.abs(act_bn - pred_bn)

    ax.plot(etas, error_no_bn, 'r-', label='Without BN', linewidth=2)
    ax.plot(etas, error_bn, 'b-', label='With BN', linewidth=2)
    ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Step size η')
    ax.set_ylabel('|Actual - Predicted| loss')
    ax.set_title('Gradient Prediction Error: BN vs No BN\n'
                 '(smaller = more predictable / smoother landscape)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved prediction error comparison to {save_path}")
    plt.close()


def plot_beta_smoothness(alphas, smooth_no_bn, smooth_bn, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(alphas, smooth_no_bn, 'r-o', label='Without BN',
            markersize=4, linewidth=2)
    ax.plot(alphas, smooth_bn, 'b-o', label='With BN',
            markersize=4, linewidth=2)
    ax.set_xlabel('Perturbation size α (fraction of ||θ||)')
    ax.set_ylabel('||∇L(θ + d) - ∇L(θ)|| / ||d||')
    ax.set_title('Local Beta-Smoothness: Gradient Lipschitz Constant\n'
                 '(smaller = smoother gradient)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved beta-smoothness to {save_path}")
    plt.close()


def main():
    train_loader = get_cifar_loader(root=os.path.join(module_path, '..', '..', 'data'),
                                     train=True, shuffle=False,
                                     batch_size=64, n_items=2000,
                                     num_workers=0)
    criterion = nn.CrossEntropyLoss()
    max_batches = 5

    no_bn_checkpoint = os.path.join(models_path, 'vgg_a_no_bn_best.pt')
    bn_checkpoint = os.path.join(models_path, 'vgg_a_bn_best.pt')

    print("\n" + "=" * 60)
    print("Computing Gradient Predictiveness ...")
    print("=" * 60)

    etas, pred_no_bn, act_no_bn = gradient_predictiveness(
        VGG_A, train_loader, criterion, device,
        checkpoint_path=no_bn_checkpoint,
        max_batches=max_batches, n_steps=21, max_eta=0.003)

    etas, pred_bn, act_bn = gradient_predictiveness(
        VGG_A_BatchNorm, train_loader, criterion, device,
        checkpoint_path=bn_checkpoint,
        max_batches=max_batches, n_steps=21, max_eta=0.003)

    plot_gradient_predictiveness(
        etas, pred_no_bn, act_no_bn, pred_bn, act_bn,
        save_path=os.path.join(figures_path, 'gradient_predictiveness.png'))

    plot_gradient_predictiveness_combined(
        etas, pred_no_bn, act_no_bn, pred_bn, act_bn,
        save_path=os.path.join(figures_path, 'prediction_error.png'))

    print("\n" + "=" * 60)
    print("Computing Beta-Smoothness ...")
    print("=" * 60)

    alphas, smooth_no_bn = beta_smoothness(
        VGG_A, train_loader, criterion, device,
        checkpoint_path=no_bn_checkpoint,
        max_batches=max_batches, n_steps=21, max_alpha=0.1)

    alphas, smooth_bn = beta_smoothness(
        VGG_A_BatchNorm, train_loader, criterion, device,
        checkpoint_path=bn_checkpoint,
        max_batches=max_batches, n_steps=21, max_alpha=0.1)

    plot_beta_smoothness(
        alphas, smooth_no_bn, smooth_bn,
        save_path=os.path.join(figures_path, 'beta_smoothness.png'))

    print("\nAll analysis plots saved to reports/figures/")
    print("Done.")


if __name__ == '__main__':
    main()
