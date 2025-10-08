import numpy as np 
import torch
import os
import copy
import random
import time
import pickle
from torch_geometric.data import Data, InMemoryDataset
import torch.nn.functional as F

from typing import Optional, Callable, List
from gen_pkl_for_subgraph_mining import OrderedData 
from multiprocessing import Pool, cpu_count
# from parse_pm_aig_verilog import BoundaryData

import pickle
import os
import argparse

def get_parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_folder', default='./dataset/boundary', type=str)
    parser.add_argument('--output_file', default='./dataset/boundary', type=str)

    args = parser.parse_args()
    
    return args

class BoundaryData(Data):
    def __init__(self): 
        super().__init__()
    
    def __inc__(self, key, value, *args, **kwargs):
        if key in ['pm_edge_index', 'pm_forward_index', 'aig_to_cell', 'sub_aig_to_cell'] :
            return self.pm_forward_index.shape[0]
        elif key in ['aig_edge_index', 'aig_forward_index'] :
            return self.aig_forward_index.shape[0]
        elif key in ['sub_aig_edge_index', 'sub_aig_forward_index'] :
            return self.sub_aig_forward_index.shape[0]
        elif 'batch' in key:
            return 1
        else:
            return 0

    def __cat_dim__(self, key, value, *args, **kwargs):
        if  "edge_index" in key :
            return 1
        else:
            return 0


def merge_pt_files(input_folder, output_file):

    train_merged_data = []
    test_merged_data = []

    pkl_files = [f for f in os.listdir(input_folder) if f.endswith(".pt")]
    if 'forgeeda' in input_folder:
        all_cir_name = ['_'.join(file.split('_')[:-4]) for file in pkl_files]
        all_cir_name = list(set(all_cir_name))
        val_split_file = './dataset/forgeeda_val_split.txt'
        if  os.path.exists(val_split_file):
            with open(val_split_file, 'r') as f:
                val_list = f.readlines()
                val_list = [x.strip() for x in val_list]
                val_list = list(set(val_list))
        else:
            val_list = random.sample(all_cir_name, int(len(all_cir_name)*0.1))
            with open(val_split_file, 'w') as f:
                for cir in val_list:
                    f.write(cir+'\n')
                    
    for i,file in enumerate(pkl_files):
        if i%100 == 0:
            print(f'process {i}')
        file_path = os.path.join(input_folder, file)
        G = torch.load(file_path)
        
        if 'itc99' in input_folder:
            cir_name = '_'.join(file.split('_')[:3])
            if cir_name in ['b15_opt_C','b20_opt_C']:
                test_merged_data.append(G)
            else:
                train_merged_data.append(G) 
        elif 'openabcd' in input_folder:
            cir_name = '_'.join(file.split('_')[:-4])
            if cir_name in ['dft','wb_conmax','ethernet','bp_be','aes_secworks']:
                test_merged_data.append(G)
            else:
                train_merged_data.append(G)  
        elif 'forgeeda' in input_folder:
            cir_name = '_'.join(file.split('_')[:-4])
            if cir_name in val_list:
                test_merged_data.append(G)
            else:
                train_merged_data.append(G)

    torch.save(test_merged_data, output_file+'_test.pt')
    torch.save(train_merged_data, output_file+'_train.pt')

    print(f"merge {len(pkl_files)} .pt file to {output_file}")

  
if __name__ == "__main__":
    print('merge the data and split train and test')
    args = get_parse_args()
    input_folder = args.input_folder
    output_file = args.output_file
    merge_pt_files(input_folder, output_file)


    

