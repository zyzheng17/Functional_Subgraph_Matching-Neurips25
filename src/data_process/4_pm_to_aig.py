import os
import argparse
import glob
from multiprocessing import Pool, cpu_count
import sys
sys.path.append('./src')
from utils.utils import run_command

def get_parse_args():
    parser = argparse.ArgumentParser()
    # Range
    parser.add_argument('--num_workers', default=1, type=int)
    
    # Input
    parser.add_argument('--lib_file', default='./src/data_process/sky130.lib', type=str)
    parser.add_argument('--data_path', default='./dataset/pm', type=str)
    
    # Output
    parser.add_argument('--save_path', default='./dataset/pm_aig', type=str)
    args = parser.parse_args()
    
    return args

def convert_to_aig(data):
    """
    Convert Verilog file to AIGER format using Yosys
    """
    idx, v_file, lib_file, save_path = data

    aig_v_file = v_file.split("/")[-1].replace(".v", "_aig.v")
    if os.path.exists(os.path.join(save_path, aig_v_file)):
        print(f"File {aig_v_file} already exists, skipping...")
        return
    cmd = f"yosys -p 'read_liberty -ignore_miss_func {lib_file}; read_verilog {v_file}; synth -auto-top; aigmap; opt_clean; write_verilog {save_path}/{aig_v_file}'"
    os.system(cmd)

if __name__ == "__main__":
    print('1. extract aig for the node in pm netlist')
    args = get_parse_args()
    v_files = glob.glob(os.path.join(args.data_path, "*.v"))

    data = [[i,v_file, args.lib_file, args.save_path] for i,v_file in enumerate(v_files)]

    if args.num_workers==1:
        for i in range(len(data)):
            convert_to_aig(data[i])
            if i%100 == 0:
                print(f'process {i} circuits')
    else:
        with Pool(args.num_workers) as pool:
            pool.map(convert_to_aig,  data)

