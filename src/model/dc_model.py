import torch
import os
from torch import nn
from torch.nn import LSTM, GRU

from .mlp import MLP
from .tfmlp import TFMlpAggr

class DeepCell(nn.Module):
    '''
    Recurrent Graph Neural Networks for Circuits.
    '''
    def __init__(self, 
                 num_rounds = 1, 
                 dim_hidden = 128, 
                 enable_encode = True,
                 enable_reverse = False, 
                 aggr='dg'
                ):
        super(DeepCell, self).__init__()
        
        # configuration
        self.num_rounds = num_rounds
        self.enable_encode = enable_encode
        self.enable_reverse = enable_reverse     
        self.aggr = aggr

        # dimensions
        self.dim_hidden = dim_hidden
        self.dim_mlp = 32

        self.aggr_cell_strc = TFMlpAggr(self.dim_hidden*1, self.dim_hidden)
        self.aggr_cell_func = TFMlpAggr(self.dim_hidden*2, self.dim_hidden)


        self.update_cell_strc = GRU(self.dim_hidden + 64, self.dim_hidden)
        self.update_cell_func = GRU(self.dim_hidden + 64, self.dim_hidden)


    def forward(self, x, edge_index, forward_level, forward_index, is_hs=False):
        device = next(self.parameters()).device
        num_nodes = x.shape[0]

        # print(max(forward_level).item())
        num_layers_f = max(forward_level).item() + 1
        
        # initialize the structure hidden state
        hs = torch.zeros(num_nodes, self.dim_hidden)
        vectors = torch.rand((forward_level==0).sum(), self.dim_hidden) - 0.5
        hs[forward_level==0] = vectors / torch.norm(vectors, dim=1, keepdim=True)
        # initialize the function hidden state
        # hf = self.hf_emd_int(self.one).view(1, -1) # (1 x 1 x dim_hidden)
        # hf = hf.repeat(num_nodes, 1) # (1 x num_nodes x dim_hidden)
        hf = torch.zeros(num_nodes, self.dim_hidden)
        hs = hs.to(device)
        hf = hf.to(device)
        
        edge_index = edge_index

        node_state = torch.cat([hs, hf], dim=-1)

        for _ in range(self.num_rounds):
            for level in range(1, num_layers_f):
                # forward layer
                # layer_mask = forward_level == level
                layer_mask = forward_level == level
                l_node = forward_index[layer_mask]

                if l_node.size(0) > 0:
                    
                    l_edge_index = edge_index[:, forward_level[edge_index[1]] == level]
                    l_x = torch.index_select(x, dim=0, index=l_node)
                    
                    # Update structure hidden state
                    msg = self.aggr_cell_strc(hs, l_edge_index)

                    l_msg = torch.index_select(msg, dim=0, index=l_node)
                    l_hs = torch.index_select(hs, dim=0, index=l_node)
                    _, l_hs = self.update_cell_strc(torch.cat([l_msg, l_x], dim=1).unsqueeze(0), l_hs.unsqueeze(0))

                    hs[l_node, :] = l_hs.squeeze(0)
                    # Update function hidden state
                    msg = self.aggr_cell_func(node_state, l_edge_index)

                    l_msg = torch.index_select(msg, dim=0, index=l_node)
                    l_hf = torch.index_select(hf, dim=0, index=l_node)
                    _, l_hf = self.update_cell_func(torch.cat([l_msg, l_x], dim=1).unsqueeze(0), l_hf.unsqueeze(0))

                    hf[l_node, :] = l_hf.squeeze(0)

                
                # Update node state
                node_state = torch.cat([hs, hf], dim=-1)

        node_embedding = node_state.squeeze(0)
        hs = node_embedding[:, :self.dim_hidden]
        hf = node_embedding[:, self.dim_hidden:]

        # return hs, hf
        if is_hs == False:
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
        
    def load_pretrained(self, pretrained_model_path = ''):
        if pretrained_model_path == '':
            pretrained_model_path = os.path.join(os.path.dirname(__file__), 'pretrained', 'model.pth')
        self.load(pretrained_model_path)
