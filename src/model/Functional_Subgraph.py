import os
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.nn import GRU
import torch_geometric.nn as gnn
from torch_geometric.nn import MessagePassing
from .mlp import MLP
import torch_geometric as pyg
from .DeepGate2 import DeepGate2
from .dc_model import DeepCell
from info_nce import InfoNCE

def generate_negative_pair_idx(bs):

    first_idx = torch.arange(0, bs)
    second_idx = torch.randperm(bs)
    
    mask = (first_idx == second_idx)
    while mask.any():
        second_idx[mask] = torch.randperm(bs)[mask]
        mask = (first_idx == second_idx)
    
    return torch.stack([first_idx, second_idx])

class FuncSub(pl.LightningModule):

    def __init__(self,args):
        super().__init__()  
        self.args = args
        self.aig_encoder = DeepGate2(num_rounds=1, dim_hidden=self.args.hidden)
        self.pm_encoder = DeepCell(num_rounds=1, dim_hidden=self.args.hidden)
        self.hf_dec = MLP(2*self.args.hidden, self.args.hidden, 2, num_layer=3)
        self.save_hyperparameters()
        self.infonce = InfoNCE(negative_mode='paired')

        self.training_step_outputs = []
        self.test_step_outputs = []
        self.val_step_outputs = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        return optimizer

    def compute_metrics(self, preds, labels):
        TP = ((preds == 1) & (labels == 1)).sum().item() / preds.shape[0]
        FP = ((preds == 1) & (labels == 0)).sum().item() / preds.shape[0]
        TN = ((preds == 0) & (labels == 0)).sum().item() / preds.shape[0]
        FN = ((preds == 0) & (labels == 1)).sum().item() / preds.shape[0]
        return TP, FP, TN, FN

    def compute_PR(self, predictions, labels):

        predictions = predictions.view(-1)
        labels = labels.view(-1)

        true_positives = torch.logical_and(predictions == 1, labels == 1).sum().item()
        false_positives = torch.logical_and(predictions == 1, labels == 0).sum().item()
        false_negatives = torch.logical_and(predictions == 0, labels == 1).sum().item()

        # Precision: TP / (TP + FP)
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0

        # Recall: TP / (TP + FN)
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0

        # F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics = {
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1
        }
        return precision, recall, f1
    
    def forward(self, batch, batch_idx):
        bs = batch.batch_size
        device = batch.x.device

        #encode aig
        orig_hf_g = self.aig_encoder(batch.gate, batch.edge_index, batch.forward_level, batch.forward_index)
        rd_hf_g = self.aig_encoder(batch.rd_gate, batch.rd_edge_index, batch.rd_forward_level, batch.rd_forward_index)
        syn_hf_g = self.aig_encoder(batch.syn_gate, batch.syn_edge_index, batch.syn_forward_level, batch.syn_forward_index)

        #encode pm
        pm_hf_g = self.pm_encoder(batch.pm_x, batch.pm_edge_index, batch.pm_forward_level, batch.pm_forward_index)

        
        orig_hf = orig_hf_g[torch.logical_and(batch.forward_level!=0, batch.backward_level==0)]
        rd_hf = rd_hf_g[torch.logical_and(batch.rd_forward_level!=0, batch.rd_backward_level==0)]
        syn_hf = syn_hf_g[torch.logical_and(batch.syn_forward_level!=0, batch.syn_backward_level==0)]
        pm_hf = pm_hf_g[torch.logical_and(batch.pm_forward_level!=0, batch.pm_backward_level==0)]

        neg_pair_idx = generate_negative_pair_idx(bs).to(device)

        # Contrastive Embedding Alignment 
        contra_neg_pair = torch.arange(bs).unsqueeze(0).repeat(bs, 1)[torch.eye(bs)==0].view(bs, bs - 1)
        if bs>256: # in case OOM
            contra_neg_pair = contra_neg_pair[:,torch.randperm(bs-1)[:256]]
        # intra-modal alignment 
        L_orig_syn = self.infonce(orig_hf, syn_hf, syn_hf[contra_neg_pair])
        # inter-modal alignment
        L_orig_pm = self.infonce(orig_hf, pm_hf, pm_hf[contra_neg_pair])
       
        #orig & rd
        orig_pos_pair = torch.cat([rd_hf, orig_hf],dim=-1)
        orig_neg_pair = torch.cat([rd_hf[neg_pair_idx[0]], orig_hf[neg_pair_idx[1]]],dim=-1)
        orig_pair = torch.cat([orig_pos_pair, orig_neg_pair],dim=0)
        orig_pred = self.hf_dec(orig_pair)
        orig_label = torch.cat([torch.ones(bs), torch.zeros(bs)],dim=0).long().to(device)
        L_orig = F.cross_entropy(orig_pred, orig_label)
        orig_acc = (orig_pred.argmax(dim=-1) == orig_label).sum().item() / orig_label.shape[0]

        #syn & rd
        syn_pos_pair = torch.cat([rd_hf, syn_hf],dim=-1)
        syn_neg_pair = torch.cat([rd_hf[neg_pair_idx[0]], syn_hf[neg_pair_idx[1]]],dim=-1)
        syn_pair = torch.cat([syn_pos_pair, syn_neg_pair],dim=0)
        syn_label = torch.cat([torch.ones(bs), torch.zeros(bs)],dim=0).long().to(device)
        syn_pred = self.hf_dec(syn_pair)
        L_syn = F.cross_entropy(syn_pred, syn_label)
        syn_pred_label = syn_pred.argmax(dim=-1)
        syn_acc = (syn_pred_label == syn_label).sum().item() / syn_label.shape[0]

        #pm & rd
        pm_pos_pair = torch.cat([rd_hf, pm_hf],dim=-1)
        pm_neg_pair = torch.cat([rd_hf[neg_pair_idx[0]], pm_hf[neg_pair_idx[1]]],dim=-1)
        pm_pair = torch.cat([pm_pos_pair, pm_neg_pair],dim=0)
        pm_label = torch.cat([torch.ones(bs), torch.zeros(bs)],dim=0).long().to(device)
        pm_pred = self.hf_dec(pm_pair)
        L_pm = F.cross_entropy(pm_pred, pm_label)
        pm_pred_label = pm_pred.argmax(dim=-1)
        pm_acc = (pm_pred_label == pm_label).sum().item() / pm_label.shape[0]

        # summarize loss
        loss = L_orig + L_syn + L_pm
        loss_align = L_orig_syn + L_orig_pm

        syn_prec, syn_rec, syn_f1 = self.compute_PR(syn_pred_label, syn_label)
        pm_prec, pm_rec, pm_f1 = self.compute_PR(pm_pred_label, pm_label)
        
        metrics = {
            'syn':
                {
                "Precision": syn_prec,
                "Recall": syn_rec,
                "F1-Score": syn_f1
                },
            'pm':
                {
                "Precision": pm_prec,
                "Recall": pm_rec,
                "F1-Score": pm_f1
                }
        }
        
        return loss, loss_align, orig_acc, syn_acc, pm_acc, metrics

    def training_step(self, batch, batch_idx):

        loss, loss_align, orig_acc, syn_acc, pm_acc, metrics = self.forward(batch, batch_idx)
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=False, batch_size=self.args.batch_size)
        self.log('train_align_loss', loss_align, on_step=False, on_epoch=True, prog_bar=True, logger=False, batch_size=self.args.batch_size)

        self.training_step_outputs.append({'loss': loss+loss_align, 'orig_acc': orig_acc, 'syn_acc': syn_acc, 'pm_acc': pm_acc, 'metrics':metrics})

        return loss + loss_align
    
    def validation_step(self, batch, batch_idx):

        loss, loss_align, orig_acc, syn_acc, pm_acc, metrics = self.forward(batch, batch_idx)
 
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=False, batch_size=self.args.batch_size)
        self.log('val_align_loss', loss_align, on_step=False, on_epoch=True, prog_bar=True, logger=False, batch_size=self.args.batch_size)

        self.val_step_outputs.append({'loss': loss+loss_align, 'orig_acc': orig_acc, 'syn_acc': syn_acc, 'pm_acc': pm_acc,'metrics':metrics})

        return loss + loss_align
    
    def on_train_epoch_end(self):
        orig_acc = sum([x['orig_acc'] for x in self.training_step_outputs])/len(self.training_step_outputs)
        syn_acc = sum([x['syn_acc'] for x in self.training_step_outputs])/len(self.training_step_outputs)
        pm_acc = sum([x['pm_acc'] for x in self.training_step_outputs])/len(self.training_step_outputs)
        pm_prec = sum([x['metrics']['pm']['Precision'] for x in self.training_step_outputs])/len(self.training_step_outputs)
        pm_rec = sum([x['metrics']['pm']['Recall'] for x in self.training_step_outputs])/len(self.training_step_outputs)
        pm_f1 = sum([x['metrics']['pm']['F1-Score'] for x in self.training_step_outputs])/len(self.training_step_outputs)
        syn_prec = sum([x['metrics']['syn']['Precision'] for x in self.training_step_outputs])/len(self.training_step_outputs)
        syn_rec = sum([x['metrics']['syn']['Recall'] for x in self.training_step_outputs])/len(self.training_step_outputs)
        syn_f1 = sum([x['metrics']['syn']['F1-Score'] for x in self.training_step_outputs])/len(self.training_step_outputs)
        self.log('train_orig_accuarcy_epoch', round(float(orig_acc),4), on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.batch_size)
        self.log('train_syn_accuarcy_epoch', round(float(syn_acc),4), on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.batch_size)
        self.log('train_pm_accuarcy_epoch', round(float(pm_acc),4), on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.batch_size)
        self.log('train_syn_precision_epoch', round(float(syn_prec),4), on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.batch_size)
        self.log('train_syn_recall_epoch', round(float(syn_rec),4), on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.batch_size)
        self.log('train_syn_f1_epoch', round(float(syn_f1),4), on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.batch_size)
        self.log('train_pm_precision_epoch', round(float(pm_prec),4), on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.batch_size)
        self.log('train_pm_recall_epoch', round(float(pm_rec),4), on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.batch_size)
        self.log('train_pm_f1_epoch', round(float(pm_f1),4), on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.batch_size)

        self.training_step_outputs.clear()

    def on_validation_epoch_end(self):
        orig_acc = sum([x['orig_acc'] for x in self.val_step_outputs])/len(self.val_step_outputs)
        syn_acc = sum([x['syn_acc'] for x in self.val_step_outputs])/len(self.val_step_outputs)
        pm_acc = sum([x['pm_acc'] for x in self.val_step_outputs])/len(self.val_step_outputs)
        pm_prec = sum([x['metrics']['pm']['Precision'] for x in self.val_step_outputs])/len(self.val_step_outputs)
        pm_rec = sum([x['metrics']['pm']['Recall'] for x in self.val_step_outputs])/len(self.val_step_outputs)
        pm_f1 = sum([x['metrics']['pm']['F1-Score'] for x in self.val_step_outputs])/len(self.val_step_outputs)
        syn_prec = sum([x['metrics']['syn']['Precision'] for x in self.val_step_outputs])/len(self.val_step_outputs)
        syn_rec = sum([x['metrics']['syn']['Recall'] for x in self.val_step_outputs])/len(self.val_step_outputs)
        syn_f1 = sum([x['metrics']['syn']['F1-Score'] for x in self.val_step_outputs])/len(self.val_step_outputs)
        self.log('val_orig_accuarcy_epoch', round(float(orig_acc),4), on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.batch_size)
        self.log('val_syn_accuarcy_epoch', round(float(syn_acc),4), on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.batch_size)
        self.log('val_pm_accuarcy_epoch', round(float(pm_acc),4), on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.batch_size)
        self.log('val_syn_precision_epoch', round(float(syn_prec),4), on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.batch_size)
        self.log('val_syn_recall_epoch', round(float(syn_rec),4), on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.batch_size)
        self.log('val_syn_f1_epoch', round(float(syn_f1),4), on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.batch_size)
        self.log('val_pm_precision_epoch', round(float(pm_prec),4), on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.batch_size)
        self.log('val_pm_recall_epoch', round(float(pm_rec),4), on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.batch_size)
        self.log('val_pm_f1_epoch', round(float(pm_f1),4), on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.batch_size)
        self.val_step_outputs.clear()


    

