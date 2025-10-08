import sys
import os
import glob
import deepgate as dg
import torch
import numpy as np
import time
import argparse
from collections import deque, defaultdict
import math

sys.path.append('./src')
from utils import aiger_utils as aiger_utils
from utils import circuit_utils as circuit_utils
from utils.utils import run_command
import random
from collections import deque, defaultdict
from multiprocessing import Pool, cpu_count
gate_to_index = {'PI': 0, 'AND': 1, 'NOT': 2, 'DFF': 3}

def get_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--aig_dir', type=str, default='path/to/aig_dir')
    parser.add_argument('--depth', type=list, default=[8,12])
    parser.add_argument('--dp_ratio', type=list, default=[0.6,0.95])
    parser.add_argument('--sample_ratio', type=float, default=0.003) # 0.2 for ITC99; 0.05 for OpenABCD, 0.003 for ForgeEDA
    parser.add_argument('--num_workers', default=1, type=int)

    parser.add_argument('--syn_cmd_list', type=list, default=['src_rw', 'src_rs', 'src_rws', 'resyn2rs', 'compress2rs'])
    parser.add_argument('--save_root', type=str, default='path/to/cut_aig')
    parser.add_argument('--abc_path', type=str, default='xxx/abc/abc')

    parser.add_argument('--thre', type=int, default=50)
    args = parser.parse_args()
    return args

def get_fanin_fanout(
    x_data,
    edge_index,
):
    fanout_list = []
    fanin_list = []
    for idx, x_data_info in enumerate(x_data):
        fanout_list.append([])
        fanin_list.append([])
    for edge in edge_index:
        intedge0 = int(edge[0])
        intedge1 = int(edge[1])
        fanout_list[intedge0].append(intedge1)
        fanin_list[intedge1].append(intedge0)
    return fanin_list, fanout_list

def save_bench(file, x_data, fanin_list, fanout_list, gate_to_idx={'PI': 0, 'AND': 1, 'NOT': 2, 'DFF': 3}):
    PI_list = []
    PO_list = []
    for idx, ele in enumerate(fanin_list):
        if len(fanin_list[idx]) == 0:
            PI_list.append(idx)
    for idx, ele in enumerate(fanout_list):
        if len(fanout_list[idx]) == 0:
            PO_list.append(idx)

    f = open(file, 'w')
    f.write('# {:} inputs\n'.format(len(PI_list)))
    f.write('# {:} outputs\n'.format(len(PO_list)))
    f.write('\n')
    # Input
    for idx in PI_list:
        f.write('INPUT({})\n'.format(x_data[idx][0]))
    f.write('\n')
    # Output
    for idx in PO_list:
        f.write('OUTPUT({})\n'.format(x_data[idx][0]))
    f.write('\n')
    # Gates
    for idx, x_data_info in enumerate(x_data):
        if idx not in PI_list:
            gate_type = None
            for ele in gate_to_idx.keys():
                if gate_to_idx[ele] == x_data_info[1]:
                    gate_type = ele
                    break
            line = '{} = {}('.format(x_data_info[0], gate_type)
            for k, fanin_idx in enumerate(fanin_list[idx]):
                if k == len(fanin_list[idx]) - 1:
                    line += '{})\n'.format(x_data[fanin_idx][0])
                else:
                    line += '{}, '.format(x_data[fanin_idx][0])
            f.write(line)
    f.write('\n')
    f.close()

    return PI_list, PO_list

def bfs_directed_graph_with_subgraph(nodes, edges, start_index, depth):
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
    gate_type = defaultdict(list)
    for node, gate in nodes:
        gate_type[node] = gate
    visited = set()
    queue = deque([(start_index, 0)])  # (node, current_depth)
    result_nodes = set()
    result_edges = set()
    max_depth = 0

    while queue:
        node, current_depth = queue.popleft()

        if node not in visited:
            max_depth = max(max_depth, current_depth)
            visited.add(node)
            result_nodes.add(node)

            if current_depth < depth:
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        if gate_type[neighbor]==2:
                            queue.append((neighbor, current_depth))
                        else:
                            queue.append((neighbor, current_depth + 1))
                        result_edges.add((node, neighbor))
                        result_nodes.add(neighbor)
    if max_depth < depth-1:
        return set(), set()
    else:
        return result_nodes, result_edges

def bfs_directed_graph_with_subgraph_origin(x_data, start_index, edge, depth, FLAG):
    graph = defaultdict(list)
    graph_back = defaultdict(list)
    for u, v in edge:
        graph[u].append(v)
    for u, v in edge:
        graph_back[v].append(u)
    visited = set()
    queue = deque([(start_index, 0)])  # (node, current_depth)
    result_nodes = set()
    result_edges = set()

    while queue:
        node, current_depth = queue.popleft()

        if node not in visited:
            visited.add(node)
            result_nodes.add(node)

            if current_depth < depth:
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        if x_data[int(neighbor)][1] == gate_to_index['NOT']:
                            queue.append((neighbor, current_depth))
                            # queue.append((neighbor, current_depth + 1))
                        else:
                            queue.append((neighbor, current_depth + 1))
                        result_edges.add((node, neighbor))

                        if FLAG:
                            for back_node in graph_back[neighbor]:
                                result_edges.add((back_node, neighbor))
                                result_nodes.add(back_node)
                        result_nodes.add(neighbor)

    return result_nodes, result_edges

def count_predecessors_dag(graph):

    predecessors_count = {}
    in_degree = {}
    for u in graph:
        predecessors_count[u] = 0
        in_degree[u] = 0
        for v in graph[u]:
            predecessors_count[v] = 0
            in_degree[v] = 0

    for u in graph:
        for v in graph[u]:
            in_degree[v] += 1

    from collections import deque
    queue = deque([node for node in graph if in_degree[node] == 0])

    while queue:
        u = queue.popleft()
        for v in graph[u]:
            predecessors_count[v] += predecessors_count[u] + 1
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    return predecessors_count

def random_bfs_sample_node(sub_x_data, sub_edge, root, visit_ratio=0.8):
    graph = defaultdict(list)
    ori_graph = defaultdict(list)
    for u, v in sub_edge:
        ori_graph[u].append(v)
        graph[v].append(u)

    node_cnt = count_predecessors_dag(ori_graph)

    # with 50% probability, we start from the root node
    # Otherwise we start from a random predecessor node 
    if random.random() < 0.5:
        pre = graph[root]
        cnt = [node_cnt[p] for p in pre]
        new_root = torch.argmax(torch.tensor(cnt)).item()
        root = pre[new_root]

    # total_nodes = sub_x_data.shape[0]
    total_nodes = len(sub_x_data)
    target_visit_count = int(total_nodes * visit_ratio)

    visited = set()
    queue = deque([root])
    result_nodes = set()
    result_edges = set()

    while queue and len(result_nodes) < target_visit_count:
        node = queue.popleft()

        if node not in visited:
            visited.add(node)
            result_nodes.add(node)

            neighbors = graph[node]
            random.shuffle(neighbors) 
            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.append(neighbor)
                    result_edges.add((node, neighbor))
                    result_nodes.add(neighbor)

    return result_nodes, result_edges, root

def random_sample_node(sub_x_data, sub_edge, ratio=0.8):

    graph = defaultdict(list)
    reverse_graph = defaultdict(list)
    for u, v in sub_edge:
        graph[u].append(v)
        reverse_graph[v].append(u)

    total_nodes = sub_x_data.shape[0]
    target_node_count = int(total_nodes * ratio)

    nodes = set(sub_x_data)
    edges = set(map(tuple, sub_edge))


    node_to_index = {node: idx for idx, node in enumerate(nodes)}
    index_to_node = {idx: node for node, idx in node_to_index.items()}

    out_degree = torch.zeros(total_nodes, dtype=torch.int)
    in_degree = torch.zeros(total_nodes, dtype=torch.int)

    for u, v in sub_edge:
        out_degree[node_to_index[u]] += 1
        in_degree[node_to_index[v]] += 1

    while len(nodes) > target_node_count:
        zero_in_out_degree_indices = torch.where((out_degree == 0) | (in_degree == 0))[0].tolist()
        zero_in_out_degree_nodes = [index_to_node[idx] for idx in zero_in_out_degree_indices if index_to_node[idx] in nodes]

        if not zero_in_out_degree_nodes:
            break

        node_to_remove = random.choice(zero_in_out_degree_nodes)
        node_to_remove_idx = node_to_index[node_to_remove]

        nodes.remove(node_to_remove)
        for neighbor in graph[node_to_remove]:
            if (node_to_remove, neighbor) in edges:
                neighbor_idx = node_to_index[neighbor]
                in_degree[neighbor_idx] -= 1
                edges.remove((node_to_remove, neighbor))
        for neighbor in reverse_graph[node_to_remove]:
            if (neighbor, node_to_remove) in edges:
                neighbor_idx = node_to_index[neighbor]
                out_degree[neighbor_idx] -= 1
                edges.remove((neighbor, node_to_remove))
        del graph[node_to_remove]
        del reverse_graph[node_to_remove]

    return nodes, edges

def to_dense(sub_x_data,sub_edge,idx):
    map_dict = {}
    for i, k in enumerate(sub_x_data[:, 0]):
        if k == idx:
            root = i
        map_dict[k] = i
    for i in range(sub_x_data.shape[0]):
        sub_x_data[i][0] = map_dict[sub_x_data[i][0]]
    for i in range(sub_edge.shape[0]):
        sub_edge[i][0] = map_dict[sub_edge[i][0]]
        sub_edge[i][1] = map_dict[sub_edge[i][1]]
    return sub_x_data,sub_edge,root
    
def getsubgraph(idx, x_data, forward_edge_index, backward_edge_index, depth, cir_name, args):

    node, edge = bfs_directed_graph_with_subgraph(x_data, backward_edge_index, int(idx), depth)

    if edge==set():
        return

    node = torch.tensor(list(node))
    edge = torch.tensor(list(edge))
    edge = torch.stack([edge[:, 1], edge[:, 0]]).T

    if node.shape[0] < args.thre:
        return
    
    dp_ratio = random.uniform(args.dp_ratio[0],args.dp_ratio[1])
    rd_sub_node,rd_sub_edge,rd_root = random_bfs_sample_node(node.numpy().tolist(), edge.numpy().tolist(), int(idx), dp_ratio)
    rd_sub_node = torch.tensor(list(rd_sub_node))
    rd_sub_edge = torch.tensor(list(rd_sub_edge))
    rd_sub_edge = torch.stack([rd_sub_edge[:, 1], rd_sub_edge[:, 0]]).T

    npx_data = np.array(x_data)


    #for sub graph
    node = torch.unique(node)
    sub_x_data = npx_data[node]
    sub_edge = edge.numpy()

    #for rd sub graph
    rd_sub_node = torch.unique(rd_sub_node)
    rd_sub_x_data = npx_data[rd_sub_node]
    rd_sub_edge = rd_sub_edge.numpy()

    # to dense
    sub_x_data,sub_edge,root = to_dense(sub_x_data,sub_edge,idx)
    rd_sub_x_data,rd_sub_edge,rd_root = to_dense(rd_sub_x_data,rd_sub_edge,rd_root)


    save_root = args.save_root
    #save subgraph
    fanin_list, fanout_list = get_fanin_fanout(sub_x_data, sub_edge)
    sub_x_data = sub_x_data.astype(np.int32)
    tmp_bench_path = './tmp/{}_{}_{}_{}.bench'.format(cir_name,cir_name, idx, root, depth)
    fanin_list, fanout_list = save_bench(tmp_bench_path, sub_x_data, fanin_list, fanout_list)
    aig_file_path = '{}/{}_{}_{}_{}.aig'.format(save_root,cir_name, idx, root, depth)

    abc_cmd = '{} -c "read_bench {}; strash; write_aiger {}"'.format(args.abc_path,tmp_bench_path, aig_file_path)
    stdout, _ = run_command(abc_cmd)
    os.remove(tmp_bench_path)


    # save syn subgraph
    aig_file_path = '{}/{}_{}_{}_{}.aig'.format(save_root, cir_name, idx, root, depth)
    syn_aig_file_path = '{}/{}_{}_{}_{}_syn.aig'.format(save_root,cir_name, idx, root, depth)
    syn_cmd = random.sample(args.syn_cmd_list,1)[0]
    abc_cmd = f'{args.abc_path} -c "read_aiger {aig_file_path}; source {args.abc_path+'.rc'}; {syn_cmd}; write_aiger {syn_aig_file_path}"'
 
    stdout, _ = run_command(abc_cmd)

    #save rd droped subgraph
    fanin_list, fanout_list = get_fanin_fanout(rd_sub_x_data, rd_sub_edge)
    rd_sub_x_data = rd_sub_x_data.astype(np.int32)
    tmp_bench_path = './tmp/{}_{}_{}_{}_rd.bench'.format(cir_name, cir_name, idx, rd_root, depth)
    fanin_list, fanout_list = save_bench(tmp_bench_path, rd_sub_x_data, fanin_list, fanout_list)
    aig_file_path = '{}/{}_{}_{}_{}_rd.aig'.format(save_root, cir_name, idx, root, depth)
    abc_cmd = '{} -c "read_bench {}; strash; write_aiger {}"'.format(args.abc_path, tmp_bench_path, aig_file_path)
    stdout, _ = run_command(abc_cmd)
    os.remove(tmp_bench_path)

def get_aig_namelist(aig_dir):
    aig_namelist = []
    for root, dirs, files in os.walk(aig_dir):
        for file in files:
            if file.endswith('.aig'):
                aig_namelist.append(file.replace('.aig',''))
    return aig_namelist

def cutmain():
    args = get_parse_args()

    min_depth, max_depth = args.depth
    aig_namelist = get_aig_namelist(args.aig_dir)

    no_circuits = len(aig_namelist)
    tot_time = 0
    graphs = {}
    for aig_idx, cir_name in enumerate(aig_namelist):
        aig_file = os.path.join(args.aig_dir, cir_name + '.aig')

        start_time = time.time()
        try:
            x_data, edge_index = aiger_utils.aig_to_xdata(aig_file)
        except Exception as e:
            print(f"Error processing {aig_file}: {e}")
            continue
        print('Parse: {} ({:} / {:}), Size: {:}, Time: {:.2f}s, ETA: {:.2f}s, Succ: {:}'.format(cir_name, aig_idx, no_circuits, len(x_data), tot_time, tot_time / ((aig_idx + 1) / no_circuits) - tot_time, len(graphs)))
        fanin_list, fanout_list = get_fanin_fanout(x_data, edge_index)
        # Replace DFF as PPI and PPO
        no_ff = 0
        for idx in range(len(x_data)):
            if x_data[idx][1] == gate_to_index['DFF']:
                no_ff += 1
                x_data[idx][1] = gate_to_index['PI']
                for fanin_idx in fanin_list[idx]:
                    fanout_list[fanin_idx].remove(idx)
                fanin_list[idx] = []
        # Get x_data and edge_index
        edge_index = []
        for idx in range(len(x_data)):
            for fanin_idx in fanin_list[idx]:
                edge_index.append([fanin_idx, idx])
        # x_one_hot = dg.construct_node_feature(x_data, 3)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        # get level of gate
        try:
            if edge_index.shape[1] <= 2 or len(x_data) <= 5:
                continue
        except Exception as e:
            print(f"Error processing edge_index: {e}")
            continue

        forward_level, forward_index, backward_level, backward_index = dg.return_order_info(edge_index, len(x_data))

        forward_edge_index = np.array(edge_index.T)
        backward_edge_index = np.array(torch.stack([edge_index.T[:, 1], edge_index.T[:, 0]]).T)

        # for idx in forward_index:
        select_index = forward_index[forward_level>=min_depth]
        node_num = math.ceil(select_index.shape[0]*args.sample_ratio)
        select_index = select_index[torch.randperm(len(select_index))[:node_num]]
        depth = random.randint(min_depth, max_depth)
        data = [(idx, x_data, forward_edge_index, backward_edge_index, depth, cir_name, args) for idx in select_index]

        if args.num_workers == 1:
            for i in range(len(data)):
                worker(data[i])
        else:
            with Pool(args.num_workers) as pool:
                pool.map(worker,  data)

    print('finish all')

def worker(data):
    
    idx, x_data, forward_edge_index, backward_edge_index, depth, cir_name, args = data
    try:
        getsubgraph(idx, x_data, forward_edge_index, backward_edge_index, depth, cir_name, args)
    except Exception as e:
        print(f"Error processing {idx}: {e}")

    
if __name__ == '__main__':
    print('1. cut the raw data to generate aig, syn_aig, and sub_aig')
    cutmain()