import numpy as np 
import torch
import os
import copy
import random
import time
import pickle
from torch_geometric.data import Data, InMemoryDataset, Dataset
import torch.nn.functional as F

from typing import Optional, Callable, List
import sys
sys.path.append('./src')
from utils.data_utils import OrderedData, BoundaryData
from data_process import parse_pm_aig_verilog 
from data_process import gen_pkl_for_subgraph_mining
from multiprocessing import Pool, cpu_count


        
class GraphDataset(Dataset):
    def __init__(self, data_dir):
        super(GraphDataset, self).__init__()

        data_list = self.load_graph_pt(data_dir)
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

    def load_graph_pt(self, dir): 
        data = torch.load(dir, weights_only=False)
        return data

if __name__=='__main__':
    data_dir =  None
    TrainDataset = GraphDataset(data_dir)
    print(TrainDataset[0])
