import os
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import time
import argparse
import pytorch_lightning as pl

from model.Functional_Subgraph import FuncSub
from dataset.aig_parser import GraphDataset
from torch_geometric.loader import DataLoader

def get_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--devices', default=[0], type=list)
    parser.add_argument('--train_data', default='Path/to/your/train/dataset', type=str)
    parser.add_argument('--test_data', default='Path/to/your/test/dataset', type=str)
    parser.add_argument('--log_path', default='Path/to/your/log', type=str)

    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--max_epochs', default=100, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--thre', default=0.625, type=float)

    parser.add_argument('--in_channels', default=4, type=int)
    parser.add_argument('--hidden', default=128, type=int)

    return parser.parse_args()

if __name__ == '__main__':
    # init args
    args = get_parse_args()

    model = FuncSub(args)

     # init dataset [aig, sub_aig, syn_aig, pm]
    train_dataset = GraphDataset(args.train_data)
    val_dataset = GraphDataset(args.test_data)

    # init dataloader 
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, 
                                drop_last=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                                drop_last=True, num_workers=args.num_workers)

    trainer = pl.Trainer(default_root_dir=args.log_path, max_epochs=args.max_epochs, devices=args.devices, log_every_n_steps=10) 
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)