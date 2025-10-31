import os 
import glob
import numpy as np 
import random
import copy
import time
import argparse
import torch.nn.functional as F

import networkx as nx

import shutil
import sys
sys.path.append('./src')
from utils.utils import run_command
from collections import defaultdict
from multiprocessing import Pool, cpu_count
gate_to_index={'PI': 0, 'AND': 1, 'NOT': 2, 'DFF': 3}

import sys
sys.setrecursionlimit(1000000)

def get_parse_args():
    parser = argparse.ArgumentParser()

    # Range
    parser.add_argument('--num_workers', default=1, type=int)
    
    parser.add_argument('--lib_path', default='./src/data_process/sky130.lib', type=str)
    # Input
    parser.add_argument('--aig_dir', default='./dataset/itc99/aig', type=str)
    # Output
    parser.add_argument('--pm_dir', default='./dataset/itc99/pm', type=str)

    parser.add_argument('--abc_path', default='../abc/abc', type=str)
    
    args = parser.parse_args()
    
    return args


def mapping(data):
    i,aig = data
    aig_file = os.path.join(args.aig_dir, aig + '.aig')
    pm_file = os.path.join(args.pm_dir, aig + '.v')
    cmd = f'{args.abc_path} -c "read {args.lib_path}; read_aiger {aig_file}; strash; map; write_verilog {pm_file}"'
    run_command(cmd)

if __name__ == '__main__':   
    print('2. generate the pm netlist for aig')  
    args = get_parse_args()
    
    aig_files = glob.glob('{}/*.aig'.format(args.aig_dir))
    aig_namelist = []
    for aig_file in aig_files:
        aig_name = os.path.basename(aig_file).replace('.aig', '')
        if aig_name.split('_')[-1] == 'rd' or aig_name.split('_')[-1] == 'syn':
            continue
        aig_namelist.append(aig_name)
    
    no_circuits = len(aig_namelist)

    data = [(i, aig) for i,aig in enumerate(aig_namelist)]

    if args.num_workers==1:
        for i in range(len(data)):
            mapping(data[i])
            if i%100 == 0:
                print(f'process {i} circuits')
    else:
        with Pool(args.num_workers) as pool:
            pool.map(mapping,  data)

    print('finish all')
