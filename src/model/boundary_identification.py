import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from torch.nn import GRU
import torch_geometric.nn as gnn
from torch_geometric.nn import MessagePassing
from .mlp import MLP
import torch_geometric as pyg
import torch_scatter
from .DeepGate2 import DeepGate2
from .dc_model import DeepCell
from info_nce import InfoNCE
import math
from collections import defaultdict, deque
from torch_sparse import SparseTensor


class BoundaryIde(pl.LightningModule):

    def __init__(self,args):
        super().__init__()  
        self.args = args

        self.aig_encoder = DeepGate2(num_rounds=1, dim_hidden=self.args.hidden)
        self.pm_encoder = DeepCell(num_rounds=1, dim_hidden=self.args.hidden)
        self.infonce = InfoNCE(negative_mode='paired')
        self.pm_seg= MLP(2*self.args.hidden, 2*self.args.hidden, 2, num_layer=3)
        
        self.save_hyperparameters()
        
        self.thre = args.thre
        self.training_step_outputs = []
        self.test_step_outputs = []
        self.val_step_outputs = []

    def compute_metrics(self, preds, labels):
        TP = ((preds == 1) & (labels == 1)).sum().item() / preds.shape[0]
        FP = ((preds == 1) & (labels == 0)).sum().item() / preds.shape[0]
        TN = ((preds == 0) & (labels == 0)).sum().item() / preds.shape[0]
        FN = ((preds == 0) & (labels == 1)).sum().item() / preds.shape[0]
        return TP, FP, TN, FN

    def find_nodes_between_start_and_end(self, edge_index, start_nodes, end_nodes):
        device = edge_index.device
        num_nodes = int(edge_index.max().item()) + 1

        adjacency_list = [[] for _ in range(num_nodes)]
        for src, dst in edge_index.t().tolist():
            adjacency_list[src].append(dst)

        start_nodes = torch.as_tensor(start_nodes, device=device).unique()
        end_nodes = torch.as_tensor(end_nodes, device=device).unique()
        end_set = set(end_nodes.tolist())


        visited = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        visited[start_nodes] = True
        queue = start_nodes.tolist() 
        result_set = set(start_nodes.tolist())
   
        while queue:
            current = queue.pop(0) 

            if current in end_set:
                continue

            for neighbor in adjacency_list[current]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    result_set.add(neighbor)
                    queue.append(neighbor)

        return torch.tensor(list(result_set), dtype=torch.long, device=device)

    def forward_boundary(self, batch, batch_idx):
        bs = batch.batch_size
        device = batch.pm_x.device

        #encode aig
        sub_aig_hf_g, sub_aig_hs_g = self.aig_encoder(batch.sub_aig_gate_type, batch.sub_aig_edge_index, batch.sub_aig_forward_level, batch.sub_aig_forward_index, is_hs=True)
        sub_aig_hf = sub_aig_hf_g[torch.logical_and(batch.sub_aig_forward_level!=0, batch.sub_aig_backward_level==0)]
        #encode pm
        pm_hf_g, pm_hs_g = self.pm_encoder(batch.pm_x, batch.pm_edge_index, batch.pm_forward_level, batch.pm_forward_index, is_hs=True)

        pm_fuse = torch.cat([pm_hf_g, sub_aig_hf[batch.pm_batch]], dim=-1)
        pred_logits = self.pm_seg(pm_fuse)

        # for graph segementation
        label = torch.isin(batch.pm_forward_index, batch.sub_aig_to_cell).long().to(device)
        label = F.one_hot(label, num_classes=2).float()

        loss = F.cross_entropy(pred_logits, label)
        pred_label = (F.sigmoid(pred_logits) > self.thre).long()

        acc = (pred_label == label).float().mean()
        union = torch.logical_or(pred_label == 1, label == 1).float().sum()
        intersection = torch.logical_and(pred_label == 1, label == 1).float().sum()
        iou = intersection / (union + 1e-5)
        dice = (2*intersection) / (pred_label.float().sum() + label.float().sum() + 1e-5)
        return loss, acc, iou, dice

    def training_step(self, batch, batch_idx):


        loss, acc, iou, dice= self.forward_boundary(batch, batch_idx)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=False, batch_size=self.args.batch_size)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=False, batch_size=self.args.batch_size)
        self.log('train_iou', iou, on_step=True, on_epoch=True, prog_bar=True, logger=False, batch_size=self.args.batch_size)
        self.log('train_dice', dice, on_step=True, on_epoch=True, prog_bar=True, logger=False, batch_size=self.args.batch_size)
        
        self.training_step_outputs.append({'loss': loss, 'acc': acc, 'iou': iou, 'dice': dice})

        return loss
    
    def validation_step(self, batch, batch_idx):

        loss, acc, iou, dice= self.forward_boundary(batch, batch_idx)
 
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=False, batch_size=self.args.batch_size)
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=False, batch_size=self.args.batch_size)
        self.log('val_iou', iou, on_step=True, on_epoch=True, prog_bar=True, logger=False, batch_size=self.args.batch_size)
        self.log('val_dice', dice, on_step=True, on_epoch=True, prog_bar=True, logger=False, batch_size=self.args.batch_size)
        self.val_step_outputs.append({'loss': loss, 'acc': acc, 'iou': iou, 'dice': dice})

        return loss
    
    def on_train_epoch_end(self):
        total_acc = sum([x['acc'] for x in self.training_step_outputs])/len(self.training_step_outputs)
        total_iou = sum([x['iou'] for x in self.training_step_outputs])/len(self.training_step_outputs)
        total_dice = sum([x['dice'] for x in self.training_step_outputs])/len(self.training_step_outputs)
        self.log('train_acc_epoch', round(float(total_acc),4), on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.batch_size)
        self.log('train_iou_epoch', round(float(total_iou),4), on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.batch_size)
        self.log('train_dice_epoch', round(float(total_dice),4), on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.batch_size)

        self.training_step_outputs.clear()


    def on_validation_epoch_end(self):
        total_acc = sum([x['acc'] for x in self.val_step_outputs])/len(self.val_step_outputs)
        total_iou = sum([x['iou'] for x in self.val_step_outputs])/len(self.val_step_outputs)
        total_dice = sum([x['dice'] for x in self.val_step_outputs])/len(self.val_step_outputs)
        self.log('val_acc_epoch', round(float(total_acc),4), on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.batch_size)
        self.log('val_iou_epoch', round(float(total_iou),4), on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.batch_size)
        self.log('val_dice_epoch', round(float(total_dice),4), on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.batch_size)
        self.val_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        return optimizer
    
