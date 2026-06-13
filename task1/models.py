import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0,
                 activation='relu'):
        super().__init__()
        act_fn = _get_activation(activation)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = act_fn()
        self.drop1 = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = act_fn()
        self.drop2 = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.drop1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + self.shortcut(x)
        out = self.act2(out)
        out = self.drop2(out)
        return out


def _get_activation(name):
    act_map = {
        'relu':    nn.ReLU,
        'leaky_relu': lambda: nn.LeakyReLU(0.1),
        'gelu':    nn.GELU,
        'elu':     nn.ELU,
        'silu':    nn.SiLU,
    }
    if name not in act_map:
        raise ValueError(f"Unknown activation: {name}. Choose from {list(act_map.keys())}")
    return act_map[name]


class CifarCNN(nn.Module):

    def __init__(self, filters=(32, 64, 128), num_blocks=(2, 2, 2),
                 dense_units=(256,), activation='relu', dropout=0.0,
                 use_bn=True, use_residual=True, num_classes=10):
        super().__init__()
        act_fn = _get_activation(activation)

        in_channels = 3
        self.stages = nn.ModuleList()

        for stage_idx, (f, n) in enumerate(zip(filters, num_blocks)):
            modules = []
            stride = 2 if stage_idx > 0 else 1
            for block_idx in range(n):
                s = stride if block_idx == 0 else 1
                if use_residual:
                    modules.append(ResidualBlock(
                        in_channels, f, stride=s, dropout=dropout,
                        activation=activation))
                else:
                    modules.append(_plain_block(
                        in_channels, f, stride=s, dropout=dropout,
                        activation=activation, use_bn=use_bn))
                in_channels = f
            self.stages.append(nn.Sequential(*modules))

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        classifier = []
        in_features = filters[-1]
        for unit in dense_units:
            classifier.extend([
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(in_features, unit),
                act_fn(),
            ])
            in_features = unit
        classifier.append(nn.Linear(in_features, num_classes))
        self.classifier = nn.Sequential(*classifier)

        self._init_weights()

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


def _plain_block(in_c, out_c, stride, dropout, activation, use_bn):
    act_fn = _get_activation(activation)
    layers = [
        nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1,
                  bias=not use_bn),
    ]
    if use_bn:
        layers.append(nn.BatchNorm2d(out_c))
    layers.append(act_fn())
    if dropout > 0:
        layers.append(nn.Dropout2d(dropout))
    layers.append(
        nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1,
                  bias=not use_bn))
    if use_bn:
        layers.append(nn.BatchNorm2d(out_c))
    layers.append(act_fn())
    if dropout > 0:
        layers.append(nn.Dropout2d(dropout))
    return nn.Sequential(*layers)


def model_small(activation='relu', dropout=0.1, **kwargs):
    return CifarCNN(filters=(16, 32, 64), num_blocks=(1, 1, 1),
                    dense_units=(128,), activation=activation,
                    dropout=dropout, **kwargs)

def model_medium(activation='relu', dropout=0.1, **kwargs):
    return CifarCNN(filters=(32, 64, 128), num_blocks=(2, 2, 2),
                    dense_units=(256,), activation=activation,
                    dropout=dropout, **kwargs)

def model_large(activation='relu', dropout=0.2, **kwargs):
    return CifarCNN(filters=(64, 128, 256, 512), num_blocks=(2, 2, 2, 2),
                    dense_units=(512, 256), activation=activation,
                    dropout=dropout, **kwargs)
