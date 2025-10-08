import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from torch.nn import GRU
from .tfmlp import TFMlpAggr
import torch_geometric.nn as gnn
from .mlp import MLP
import numpy as np

def generate_negative_pair_idx(bs):

    first_idx = torch.arange(0, bs)
    second_idx = torch.randperm(bs)
    
    mask = (first_idx == second_idx)
    while mask.any():
        second_idx[mask] = torch.randperm(bs)[mask]
        mask = (first_idx == second_idx)
    
    return torch.stack([first_idx, second_idx])

def get_slices(G):
    device = G.gate.device
    edge_index = G.edge_index
    
    # Index slices
    and_index_slices = []
    not_index_slices = []
    edge_index_slices = []
    and_mask = (G.gate == 1).squeeze(1)
    not_mask = (G.gate == 2).squeeze(1)

    # one gate will only be updated once
    for level in range(0, torch.max(G.forward_level).item() + 1):
        and_level_nodes = torch.nonzero((G.forward_level == level) & and_mask).squeeze(1)
        not_level_nodes = torch.nonzero((G.forward_level == level) & not_mask).squeeze(1)
        and_index_slices.append(and_level_nodes)
        not_index_slices.append(not_level_nodes)
        edge_index_slices.append(edge_index[:,G.forward_level[edge_index[1]]==level])
    
    return and_index_slices, not_index_slices, edge_index_slices
 
class DeepGate2(nn.Module):
    def __init__(self, num_rounds=1, dim_hidden=128, enable_encode=True, enable_reverse=False):
        super(DeepGate2, self).__init__()

        # configuration
        self.num_rounds = num_rounds
        self.enable_encode = enable_encode
        self.enable_reverse = enable_reverse   

        # dimensions
        self.dim_hidden = dim_hidden
        self.dim_mlp = 32

        # Network 
        self.aggr_and_strc = TFMlpAggr(self.dim_hidden*1, self.dim_hidden)
        self.aggr_not_strc = TFMlpAggr(self.dim_hidden*1, self.dim_hidden)
        self.aggr_and_func = TFMlpAggr(self.dim_hidden*2, self.dim_hidden)
        self.aggr_not_func = TFMlpAggr(self.dim_hidden*1, self.dim_hidden)
            
        self.update_and_strc = GRU(self.dim_hidden, self.dim_hidden)
        self.update_and_func = GRU(self.dim_hidden, self.dim_hidden)
        self.update_not_strc = GRU(self.dim_hidden, self.dim_hidden)
        self.update_not_func = GRU(self.dim_hidden, self.dim_hidden)


    def forward(self, gate, edge_index, forward_level, forward_index, is_hs=False):

        device = next(self.parameters()).device
        num_nodes = len(gate)
        max_num_layers = torch.max(forward_level).item() + 1
        min_num_layers = 0
    
        
        hs = torch.zeros(num_nodes, self.dim_hidden)
        hf = torch.zeros(num_nodes, self.dim_hidden)

        hs = hs.to(device)
        hf = hf.to(device)

        # initialize the hidden state
        hf[gate.squeeze()==0] += 0.5
        vectors = torch.randn((gate==0).sum(), self.dim_hidden).to(device)
        hs[gate.squeeze()==0] = vectors / torch.norm(vectors, dim=1, keepdim=True)


        node_state = torch.cat([hs, hf], dim=-1)
        
        for _ in range(self.num_rounds):

            for level in range(min_num_layers, max_num_layers):
                # l_and_node = and_slices[level]
                l_and_node = forward_index[torch.logical_and(gate.squeeze() == 1, forward_level == level)]

                if l_and_node.size(0) > 0:
                    and_mask  = torch.logical_and(gate[edge_index[1]].squeeze() == 1, forward_level[edge_index[1]] == level)
                    and_edge_index = edge_index[:,and_mask]
                    
                    msg = self.aggr_and_strc(hs, and_edge_index)
                    and_msg = torch.index_select(msg, dim=0, index=l_and_node)
                    hs_and = torch.index_select(hs, dim=0, index=l_and_node)
                    _, hs_and = self.update_and_strc(and_msg.unsqueeze(0), hs_and.unsqueeze(0))
                    hs[l_and_node, :] = hs_and.squeeze(0)
                    # Update function hidden state
                    msg = self.aggr_and_func(node_state, and_edge_index)
                    and_msg = torch.index_select(msg, dim=0, index=l_and_node)
                    hf_and = torch.index_select(hf, dim=0, index=l_and_node)
                    _, hf_and = self.update_and_func(and_msg.unsqueeze(0), hf_and.unsqueeze(0))
                    hf[l_and_node, :] = hf_and.squeeze(0)

                # NOT Gate
                l_not_node = forward_index[torch.logical_and(gate.squeeze() == 2, forward_level == level)]
                if l_not_node.size(0) > 0:
                    not_mask = torch.logical_and(gate[edge_index[1]].squeeze() == 2, forward_level[edge_index[1]] == level)
                    not_edge_index = edge_index[:,not_mask]
                    # Update structure hidden state
                    msg = self.aggr_not_strc(hs, not_edge_index)
                    not_msg = torch.index_select(msg, dim=0, index=l_not_node)
                    hs_not = torch.index_select(hs, dim=0, index=l_not_node)
                    _, hs_not = self.update_not_strc(not_msg.unsqueeze(0), hs_not.unsqueeze(0))
                    hs[l_not_node, :] = hs_not.squeeze(0)
                    # Update function hidden state
                    msg = self.aggr_not_func(hf, not_edge_index)
                    not_msg = torch.index_select(msg, dim=0, index=l_not_node)
                    hf_not = torch.index_select(hf, dim=0, index=l_not_node)
                    _, hf_not = self.update_not_func(not_msg.unsqueeze(0), hf_not.unsqueeze(0))
                    hf[l_not_node, :] = hf_not.squeeze(0)
                
                # Update node state
                node_state = torch.cat([hs, hf], dim=-1)
        if is_hs==False:
            return hf
        else: 
            return hf, hs
    
    def load(self, model_path):
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        state_dict_ = checkpoint['state_dict']
        state_dict = {}
        for k in state_dict_:
            if k.startswith('module') and not k.startswith('module_list'):
                state_dict[k[7:]] = state_dict_[k]
            else:
                state_dict[k] = state_dict_[k]
        model_state_dict = self.state_dict()
        
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print('Skip loading parameter {}, required shape{}, loaded shape{}.'.format(
                        k, model_state_dict[k].shape, state_dict[k].shape))
                    state_dict[k] = model_state_dict[k]
            else:
                print('Drop parameter {}.'.format(k))
        for k in model_state_dict:
            if not (k in state_dict):
                print('No param {}.'.format(k))
                state_dict[k] = model_state_dict[k]
        self.load_state_dict(state_dict, strict=False)
    

