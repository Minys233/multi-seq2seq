import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Field
import yaml
import cloudpickle
from pathlib import Path
from datetime import datetime


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_weight(m: nn.Module):
    for name, param in m.named_parameters():
        if 'bias' in name:
            continue
        nn.init.kaiming_normal_(param.data)


def save_model(name: str, model: nn.Module, epoch: int, optimizer, src: Field, trg: Field, base, note: str, **kwargs):
    if isinstance(base, str):
        base = Path(base)
    fname = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{name}_{epoch}_{note}.pt"
    savedict = {'model': model.state_dict(), 'epoch': epoch, 'optimizer': optimizer.state_dict(), 'SRC': src, 'TRG': trg}
    for k, v in kwargs.items():
        savedict[k] = v
    with open(base/fname, 'wb') as fout:
        cloudpickle.dump(savedict, fout)


def load_model(fname):
    with open(fname, 'rb') as fin:
        obj = cloudpickle.load(fin)
    return obj


def save_config(config: dict, fname):
    if isinstance(fname, str):
        fname = Path(fname)
    with open(fname, 'w') as yaml_file:
        yaml.dump(config, yaml_file, default_flow_style=False)


def load_config(fname):
    with open(fname, 'r') as fin:
        config = yaml.load(fin)
    return config