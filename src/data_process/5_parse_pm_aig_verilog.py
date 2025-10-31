import re
import networkx as nx
import torch_geometric
import csv
import torch
import deepgate as dg
from torch_geometric.data import Data
import numpy as np
import copy
import random
from collections import deque, defaultdict
import glob
import os
from torch_geometric.utils import to_networkx
import argparse

def get_parse_args():
    parser = argparse.ArgumentParser()
    
    # Input
    parser.add_argument('--pm_root', default='./dataset/pm', type=str)
    parser.add_argument('--pm_aig_root', default='./dataset/pm_aig', type=str)
    
    # Output
    parser.add_argument('--save_root', default='./dataset/boundary', type=str)
    args = parser.parse_args()
    
    return args


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


def top_sort(edge_index, graph_size):

    cell_ids = torch.arange(graph_size, dtype=int)

    node_order = torch.zeros(graph_size, dtype=int)
    unevaluated_nodes = torch.ones(graph_size, dtype=bool)

    parent_nodes = edge_index[0]
    child_nodes = edge_index[1]

    n = 0
    while unevaluated_nodes.any():
        # Find which parent nodes have not been evaluated
        unevaluated_mask = unevaluated_nodes[parent_nodes]

        # Find the child nodes of unevaluated parents
        unready_children = child_nodes[unevaluated_mask]

        # Mark nodes that have not yet been evaluated
        # and which are not in the list of children with unevaluated parent nodes
        nodes_to_evaluate = unevaluated_nodes & ~torch.isin(cell_ids, unready_children)

        node_order[nodes_to_evaluate] = n
        unevaluated_nodes[nodes_to_evaluate] = False

        n += 1

    return node_order.long()

def return_order_info(edge_index, num_nodes):
    ns = torch.LongTensor([i for i in range(num_nodes)])
    forward_level = top_sort(edge_index, num_nodes)
    ei2 = torch.LongTensor([list(edge_index[1]), list(edge_index[0])])
    backward_level = top_sort(ei2, num_nodes)
    forward_index = ns
    backward_index = torch.LongTensor([i for i in range(num_nodes)])
    
    return forward_level, forward_index, backward_level, backward_index

def parse_cell_to_aig(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    modules = re.findall(r'module\s+(\w+)\s*\((.*?)\);(.*?)endmodule', content, re.S)

    graphs = {}

    for module_name, ports, body in modules:
        inputs = re.findall(r'input\s+(\w+);', body)
        outputs = re.findall(r'output\s+(\w+);', body)
        wires = re.findall(r'wire\s+(\w+);', body)

        nodes = wires
        node_indices = {node: i for i, node in enumerate(nodes)}

        edge_index = []

        gate_type = {}
        assigns = re.findall(r'assign\s+(\w+)\s*=\s*(.*?);', body)
        for target, expr in assigns:
            expr = expr.strip()

            if '&' in expr:  # AND 
                gate_type[target] = 'AND'
                sources = [s.strip() for s in expr.split('&')]
                for source in sources:
                    edge_index.append([node_indices[source], node_indices[target]])
            elif '~' in expr:  # NOT

                gate_type[target] = 'NOT'
                source = expr.replace('~', '').strip()
                edge_index.append([node_indices[source], node_indices[target]])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
       
        gate_type_list = []
        
        type2id = {
            'AND': 1,
            'NOT': 2,
        }

        for node in nodes:
            if node in gate_type:
                gate_type_list.append(type2id[gate_type[node]])
            elif node in inputs:
                gate_type_list.append(node)

        graph = Data(
            forward_index=torch.arange(0,len(nodes)),  
            edge_index=edge_index,  
            gate_type = gate_type_list,
        )
        if edge_index.shape[0] == 0:
            return None
        forward_level, forward_index, backward_level, backward_index = return_order_info(graph.edge_index, graph.forward_index.shape[0])
        graph.forward_level = forward_level
        graph.forward_index = forward_index
        graph.backward_level = backward_level
        graph.backward_index = backward_index
        graphs[module_name] = graph

    return graphs

def remove_pin_node(graph):

    pin_mask = ~np.isin(graph.gate_type, np.array(['input','1','2']))
    assign_nodes = graph.forward_index[pin_mask]
    edge_index = graph.edge_index.t()  

    new_edges = []

    for node in assign_nodes:

        in_edges = edge_index[edge_index[:, 1] == node]  
        out_edges = edge_index[edge_index[:, 0] == node] 

        for in_edge in in_edges:
            for out_edge in out_edges:
                new_edges.append([in_edge[0].item(), out_edge[1].item()])

    edge_index = edge_index[
        (edge_index[:, 0] != assign_nodes.unsqueeze(1)).all(dim=0) &
        (edge_index[:, 1] != assign_nodes.unsqueeze(1)).all(dim=0)
    ]

    new_edges = torch.tensor(new_edges, dtype=torch.long)
    edge_index = torch.cat([edge_index, new_edges], dim=0).t()

    graph.edge_index = edge_index
    graph.gate_type = graph.gate_type[~pin_mask]
    graph.forward_index = graph.forward_index[~pin_mask]
    graph.aig_to_cell = graph.aig_to_cell[~pin_mask]
    
    return graph

def replace_node_with_subgraph_pyg(graph, cell_id, subgraph, subgraph_name):

    entry_nodes = subgraph.forward_index[torch.logical_and(subgraph.forward_level==0 , subgraph.backward_level!=0)]
    exit_nodes = subgraph.forward_index[torch.logical_and(subgraph.forward_level!=0 , subgraph.backward_level==0)]

    edge_index = graph.edge_index
    in_edges = edge_index[:, edge_index[1] == cell_id]  
    out_edges = edge_index[:, edge_index[0] == cell_id]  

    out_edges_pin = np.array(graph.edge_pin_o)[graph.edge_index[0] == cell_id]
    pins_i = np.array(graph.edge_pin_o)[graph.edge_index[1] == cell_id]
    pins_o = np.array(subgraph.gate_type)[(subgraph.forward_level==0) & (subgraph.backward_level!=0)]
    assert len(pins_i) == len(pins_o)

    mask = (edge_index[0] != cell_id) & (edge_index[1] != cell_id)
    edge_index = edge_index[:, mask]
    graph.edge_pin_i = np.array(graph.edge_pin_i)[mask]
    graph.edge_pin_o = np.array(graph.edge_pin_o)[mask]

    num_original_nodes = graph.forward_index.shape[0]
    subgraph_edge_index = subgraph.edge_index + num_original_nodes
    edge_index = torch.cat([edge_index, subgraph_edge_index], dim=1)
    graph.edge_pin_i = np.concatenate([graph.edge_pin_i, np.array(len(subgraph_edge_index[1]) * ['None'])])
    graph.edge_pin_o = np.concatenate([graph.edge_pin_o, np.array(len(subgraph_edge_index[1]) * ['None'])])

    for pin_i,u in zip(pins_i,in_edges[0]):
        for pin_o,entry_node in zip(pins_o,entry_nodes):
            if pin_i == pin_o: 
                edge_index = torch.cat([edge_index, torch.tensor([[u, entry_node + num_original_nodes]], dtype=torch.long).t()], dim=1)
                graph.edge_pin_i = np.concatenate([graph.edge_pin_i, np.array(['None'])])
                graph.edge_pin_o = np.concatenate([graph.edge_pin_o, np.array(['None'])])

    for pin, v in zip(out_edges_pin,out_edges[1]):
        for exit_node in exit_nodes:
            edge_index = torch.cat([edge_index, torch.tensor([[exit_node + num_original_nodes, v]], dtype=torch.long).t()], dim=1)
            graph.edge_pin_i = np.concatenate([graph.edge_pin_i, np.array(['None'])])
            graph.edge_pin_o = np.concatenate([graph.edge_pin_o, np.array([pin])])
            
    graph.gate_type = graph.gate_type + subgraph.gate_type
    graph.aig_to_cell = torch.cat([graph.aig_to_cell, cell_id * torch.ones_like(subgraph.forward_index,dtype=int)], dim=0)
    graph.forward_index = torch.cat([graph.forward_index, subgraph.forward_index + num_original_nodes], dim=0)
    graph.edge_index = edge_index

    return graph

def parse_verilog_to_graph(file_path):

    G = nx.DiGraph()
    AIG = Data()

    with open(file_path, 'r') as file:
        content = file.read()

        module_match = re.search(r"module\s+(\S+)\s*\(", content)
        if module_match:
            module_name = module_match.group(1)
            G.graph['module_name'] = module_name

        inputs_list = re.findall(r"input\s+([\w,\s]+);", content)
        outputs_list = re.findall(r"output\s+([\w,\s]+);", content)

        inputs_list = [item.strip() for sublist in inputs_list for item in sublist.split(',')]
        outputs_list = [item.strip() for sublist in outputs_list for item in sublist.split(',')]

        
        for input_signal in inputs_list:
            G.add_node(input_signal, gate_type='input')

        net_dict = {}

        gate_pattern = re.compile(r"(\w+)\s+(\w+)\s*\(([^;]+)\);")
        for match in gate_pattern.finditer(content):

            gate_type, gate_name, connections = match.groups()

            G.add_node(gate_name, gate_type=gate_type)
            connection_pattern = re.compile(r"\.(\w+)\((\w+)\)")

            for conn_match in connection_pattern.finditer(connections):
                pin, net = conn_match.groups()
                if net in inputs_list:
                    G.add_edge(net, gate_name,edge_pin_i=net, edge_pin_o=pin)
                elif net in outputs_list:
                    # print(net)
                    continue
                else:
                    if net not in net_dict:
                        net_dict[net] = {'i':[],'o':[]}
                    if pin in ['X','Y','z']:
                        net_dict[net]['i'].append([gate_name,pin])
                    else:
                        net_dict[net]['o'].append([gate_name,pin])
            
            for net in net_dict:
                for i,pin_i in net_dict[net]['i']:
                    for o,pin_o in net_dict[net]['o']:
                        G.add_edge(i, o, edge_pin_i=pin_i, edge_pin_o=pin_o)

    return G

def bitstring_to_tensor(bitstring):

    assert len(bitstring) == 64, "input length must be 64 "
    assert all(c in '01' for c in bitstring), "input string should only contain 0 1"

    bit_list = [int(c) for c in bitstring]

    tensor = torch.tensor(bit_list, dtype=torch.float32)
    return tensor

def read_csv(file_path):

    result_dict = {}

    with open(file_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            if len(row) >= 3:
                key = row[0].strip()  
                value = row[2].strip()  
                result_dict[key] = value * (64// len(value))  
    result_dict['input'] = '0' * 64

    for k in result_dict:
        result_dict[k] = bitstring_to_tensor(result_dict[k])
    return result_dict

def verilog2graph(file_path):
    csv_path = './src/data_process/sky130.csv'
    cell_lib = read_csv(csv_path)
    graph = parse_verilog_to_graph(file_path)
    graph = torch_geometric.utils.from_networkx(graph)
    graph.x = torch.stack([cell_lib[i] for i in graph.gate_type])
    del graph.module_name
    forward_level, forward_index, backward_level, backward_index = return_order_info(graph.edge_index, graph.x.shape[0])
    graph.forward_level = forward_level
    graph.forward_index = forward_index
    graph.backward_level = backward_level
    graph.backward_index = backward_index
    return graph


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


if __name__ == "__main__":
    print('2. generate aig from pm netlist')
    args = get_parse_args()
    pm_root = args.pm_root
    pm_aig_root = args.pm_aig_root
    save_root = args.save_root

    pm_files = glob.glob(pm_root + '*.v')
    for idx,pm_file in enumerate(pm_files):
        
        pm_aig_file = pm_file.replace('.v', '_aig.v').split('/')[-1]
        pm_aig_path = os.path.join(pm_aig_root, pm_aig_file)

        if os.path.exists(os.path.join(save_root, pm_aig_file.replace('_aig.v', '.pt'))):
            continue

        cell_lib_aig = parse_cell_to_aig(pm_aig_path)
        if cell_lib_aig is None:
            continue
        graph = verilog2graph(pm_file)
        if graph.edge_index.shape[1] <= 1:
            continue
        pm_aig_netlist = BoundaryData()
        pm_aig_netlist.pm_edge_index = graph.edge_index
        pm_aig_netlist.pm_x = graph.x
        pm_aig_netlist.pm_forward_index = graph.forward_index
        pm_aig_netlist.pm_batch = torch.zeros(pm_aig_netlist.pm_x.shape[0], dtype=torch.long)

        graph.aig_to_cell = torch.zeros_like(graph.forward_index,dtype=int) - 1
        for cell_id, cell in zip(graph.forward_index, graph.gate_type):
            if cell == 'input':
                continue
            cell_aig = cell_lib_aig[cell]
            graph = replace_node_with_subgraph_pyg(graph=graph, cell_id=cell_id, subgraph=cell_aig, subgraph_name = cell)
        
        gate_mask = torch.tensor(np.array(graph.gate_type) == 'input') | (graph.aig_to_cell != -1)
        cell_mask = ~ gate_mask
        assert not torch.isin(graph.forward_index[cell_mask], graph.edge_index.flatten()).any()

        new_graph = Data()
        new_graph.edge_index = graph.edge_index
        new_graph.gate_type = np.array(graph.gate_type)[gate_mask]
        new_graph.forward_index = graph.forward_index[gate_mask]
        new_graph.aig_to_cell = graph.aig_to_cell[gate_mask]

        # delete pin node in the graph
        new_graph = remove_pin_node(new_graph)
        assert torch.isin(new_graph.edge_index.flatten(),new_graph.forward_index).all()
        gate_dict = {'input':0,'1':1,'2':2}
        new_graph.gate_type = [gate_dict[i] for i in new_graph.gate_type]
        new_graph.gate_type = torch.tensor(new_graph.gate_type, dtype=torch.long)

        # map node index to dense index
        to_dense_index = torch.zeros(new_graph.forward_index.max() + 1, dtype=torch.long) - 1
        to_dense_index[new_graph.forward_index] = torch.arange(new_graph.forward_index.shape[0])
        new_graph.edge_index = to_dense_index[new_graph.edge_index]
        new_graph.forward_index = to_dense_index[new_graph.forward_index]
        assert not torch.isin(-1,new_graph.forward_index).any()
        assert not torch.isin(-1,new_graph.edge_index.flatten()).any()

        aig_netlist = copy.deepcopy(new_graph)
        
        forward_level, forward_index, backward_level, backward_index = return_order_info(pm_aig_netlist.pm_edge_index, pm_aig_netlist.pm_forward_index.shape[0])
        pm_aig_netlist.pm_forward_level = forward_level
        pm_aig_netlist.pm_forward_index = forward_index
        pm_aig_netlist.pm_backward_level = backward_level

        forward_level, forward_index, backward_level, backward_index = return_order_info(aig_netlist.edge_index, aig_netlist.forward_index.shape[0])
        aig_netlist.forward_level = forward_level
        aig_netlist.forward_index = forward_index
        aig_netlist.backward_level = backward_level
        aig_netlist.backward_index = backward_index

        pm_aig_netlist.aig_edge_index = aig_netlist.edge_index
        pm_aig_netlist.aig_gate_type = aig_netlist.gate_type
        pm_aig_netlist.aig_to_cell = aig_netlist.aig_to_cell
        pm_aig_netlist.aig_forward_level = forward_level
        pm_aig_netlist.aig_forward_index = forward_index
        pm_aig_netlist.aig_backward_level = backward_level
        pm_aig_netlist.aig_batch = torch.zeros(pm_aig_netlist.aig_forward_index.shape[0], dtype=torch.long)

        #sample subgraph from aig
        vis_ratio = random.uniform(0.6, 0.95)
        node = aig_netlist.forward_index
        edge = aig_netlist.edge_index.t()
        root = aig_netlist.forward_index[(aig_netlist.forward_level != 0) & (aig_netlist.backward_level == 0)]

        rd_sub_node,rd_sub_edge,rd_root = random_bfs_sample_node(node.numpy().tolist(), edge.numpy().tolist(), int(root), vis_ratio)
        if len(list(rd_sub_edge)) == 0:
            continue
        rd_sub_node = torch.tensor(list(rd_sub_node))
        rd_sub_edge = torch.tensor(list(rd_sub_edge))
        rd_sub_edge = torch.stack([rd_sub_edge[:, 1], rd_sub_edge[:, 0]]).T
        
        sub_aig_netlist = Data()
        sub_aig_netlist.edge_index = rd_sub_edge
        sub_aig_netlist.forward_index = rd_sub_node
        sub_aig_netlist.gate_type = aig_netlist.gate_type[rd_sub_node]
        sub_aig_netlist.aig_to_cell = aig_netlist.aig_to_cell[rd_sub_node]

        to_dense_index = torch.zeros(sub_aig_netlist.edge_index.max() + 1, dtype=torch.long) - 1
        to_dense_index[sub_aig_netlist.forward_index] = torch.arange(sub_aig_netlist.forward_index.shape[0])
        sub_aig_netlist.edge_index = to_dense_index[sub_aig_netlist.edge_index].T
        sub_aig_netlist.forward_index = to_dense_index[sub_aig_netlist.forward_index]

        # has_cycle_flag = has_cycle(sub_aig_netlist.edge_index, sub_aig_netlist.forward_index.shape[0])
        forward_level, forward_index, backward_level, backward_index = return_order_info(sub_aig_netlist.edge_index, sub_aig_netlist.forward_index.shape[0])
        sub_aig_netlist.forward_level = forward_level
        sub_aig_netlist.forward_index = forward_index
        sub_aig_netlist.backward_level = backward_level
        sub_aig_netlist.backward_index = backward_index

        pm_aig_netlist.sub_aig_edge_index = sub_aig_netlist.edge_index
        pm_aig_netlist.sub_aig_gate_type = sub_aig_netlist.gate_type
        pm_aig_netlist.sub_aig_to_cell = sub_aig_netlist.aig_to_cell
        pm_aig_netlist.sub_aig_forward_level = sub_aig_netlist.forward_level
        pm_aig_netlist.sub_aig_forward_index = sub_aig_netlist.forward_index
        pm_aig_netlist.sub_aig_backward_level = sub_aig_netlist.backward_level
        pm_aig_netlist.sub_aig_batch = torch.zeros(pm_aig_netlist.sub_aig_forward_index.shape[0], dtype=torch.long)

        torch.save(pm_aig_netlist, os.path.join(save_root, pm_aig_file.replace('_aig.v', '.pt')))
        # if idx%100 ==0:
        print(f"{idx} Saved: {pm_aig_file.replace('_aig.v', '.pt')}")