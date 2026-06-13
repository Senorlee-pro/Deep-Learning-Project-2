import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np

from models import CifarCNN, model_small, model_medium, model_large
from train import (train_model, count_parameters,
                   CrossEntropyWithL1, CrossEntropyWithL2, get_optimizer)


def get_cifar10_loaders(batch_size=128, num_workers=4):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root='../data', train=True, download=True, transform=train_transform)
    test_set = torchvision.datasets.CIFAR10(
        root='../data', train=False, download=True, transform=test_transform)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


def run_experiment(config, train_loader, test_loader, device):
    print(f"\n{'='*60}")
    print(f"Experiment: {config['name']}")
    print(f"{'='*60}")

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

    n_params = count_parameters(model)
    print(f"Trainable parameters: {n_params:,}")

    loss_type = config.get('loss', 'ce')
    if loss_type == 'ce_l1':
        criterion = CrossEntropyWithL1(model, l1_lambda=config.get('l1_lambda', 1e-5))
    elif loss_type == 'ce_l2':
        criterion = CrossEntropyWithL2(model, l2_lambda=config.get('l2_lambda', 1e-4))
    else:
        criterion = nn.CrossEntropyLoss()

    optim_name = config.get('optimizer', 'adam')
    lr = config.get('lr', 0.001)
    optimizer = get_optimizer(optim_name, model, lr,
                              weight_decay=config.get('weight_decay', 0),
                              momentum=config.get('momentum', 0.9))

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.get('epochs', 40))

    history, best_acc = train_model(
        model, train_loader, test_loader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=config.get('epochs', 60),
        device=device,
        scheduler=scheduler,
        verbose=True,
    )

    save_dir = 'experiment_results'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{config['name']}.pt")
    torch.save({
        'config': config,
        'state_dict': model.state_dict(),
        'history': history,
        'best_acc': best_acc,
        'n_params': n_params,
    }, save_path)

    print(f"Best test accuracy: {best_acc:.4f} | Saved to {save_path}")
    return {'name': config['name'], 'best_acc': best_acc, 'n_params': n_params,
            'history': history, 'config': config}


def get_experiment_configs():
    configs = []

    base = {
        'epochs': 30,
        'lr': 0.001,
        'loss': 'ce',
        'optimizer': 'adam',
    }

    cfg = base.copy()
    cfg.update(name='small_baseline', filters=(16, 32, 64),
               num_blocks=(1, 1, 1), dense_units=(128,),
               activation='relu', dropout=0.1)
    configs.append(cfg)

    cfg = base.copy()
    cfg.update(name='medium_baseline', filters=(32, 64, 128),
               num_blocks=(2, 2, 2), dense_units=(256,),
               activation='relu', dropout=0.1)
    configs.append(cfg)

    cfg = base.copy()
    cfg.update(name='large_baseline', filters=(64, 128, 256, 512),
               num_blocks=(2, 2, 2, 2), dense_units=(512, 256),
               activation='relu', dropout=0.2)
    configs.append(cfg)

    for act in ['leaky_relu', 'gelu', 'silu']:
        cfg = base.copy()
        cfg.update(name=f'medium_act_{act}', filters=(32, 64, 128),
                   num_blocks=(2, 2, 2), dense_units=(256,),
                   activation=act, dropout=0.1)
        configs.append(cfg)

    cfg = base.copy()
    cfg.update(name='medium_loss_l1', filters=(32, 64, 128),
               num_blocks=(2, 2, 2), dense_units=(256,),
               activation='relu', dropout=0.1, loss='ce_l1', l1_lambda=1e-5)
    configs.append(cfg)

    cfg = base.copy()
    cfg.update(name='medium_loss_l2', filters=(32, 64, 128),
               num_blocks=(2, 2, 2), dense_units=(256,),
               activation='relu', dropout=0.1, loss='ce_l2', l2_lambda=1e-4)
    configs.append(cfg)

    for opt, opt_kwargs in [
        ('sgd', {'lr': 0.01, 'momentum': 0.9, 'weight_decay': 5e-4}),
        ('adamw', {'lr': 0.001, 'weight_decay': 1e-2}),
    ]:
        cfg = base.copy()
        cfg.update(name=f'medium_opt_{opt}', filters=(32, 64, 128),
                   num_blocks=(2, 2, 2), dense_units=(256,),
                   activation='relu', dropout=0.1,
                   optimizer=opt, lr=opt_kwargs.get('lr', 0.001))
        if 'momentum' in opt_kwargs:
            cfg['momentum'] = opt_kwargs['momentum']
        if 'weight_decay' in opt_kwargs:
            cfg['weight_decay'] = opt_kwargs['weight_decay']
        configs.append(cfg)

    return configs


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_loader, test_loader = get_cifar10_loaders(batch_size=128)

    configs = get_experiment_configs()
    results = []
    os.makedirs('experiment_results', exist_ok=True)

    for config in configs:
        try:
            result = run_experiment(config, train_loader, test_loader, device)
            results.append(result)
        except Exception as e:
            print(f"FAILED: {config['name']} — {e}")

    summary = [{'name': r['name'], 'best_acc': r['best_acc'],
                'n_params': r['n_params']} for r in results]
    with open('experiment_results/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*60)
    print("SUMMARY (sorted by best accuracy)")
    print("="*60)
    for r in sorted(results, key=lambda x: x['best_acc'], reverse=True):
        print(f"  {r['name']:30s} | Acc: {r['best_acc']:.4f} | Params: {r['n_params']:,}")


if __name__ == '__main__':
    main()
