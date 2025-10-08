import re
import networkx as nx
import torch_geometric
import csv
import torch
import deepgate as dg

def parse_verilog_to_graph(file_path):

    G = nx.DiGraph()

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
                    G.add_edge(net, gate_name)
                elif net in outputs_list:
                
                    continue
                else:
                    if net not in net_dict:
                        net_dict[net] = {'i':[],'o':[]}
                    if pin in ['X','Y','z']:
                        net_dict[net]['i'].append(gate_name)
                    else:
                        net_dict[net]['o'].append(gate_name)
            
            for net in net_dict:
                for i in net_dict[net]['i']:
                    for o in net_dict[net]['o']:
                        G.add_edge(i, o)

    return G

def bitstring_to_tensor(bitstring):

    assert len(bitstring) == 64, "input length must be 64"
    assert all(c in '01' for c in bitstring), "input string can only contain '0' and '1'"

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
    del graph.gate_type
    forward_level, forward_index, backward_level, backward_index = dg.return_order_info(graph.edge_index, graph.x.shape[0])
    graph.forward_level = forward_level
    graph.forward_index = forward_index
    graph.backward_level = backward_level
    graph.backward_index = backward_index
    return graph



if __name__ == "__main__":

    file_path = None
    graph = verilog2graph(file_path)
    print(graph)