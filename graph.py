import networkx as nx
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split


def hf_to_nx(edge_index, num_nodes):
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    edges = list(zip(edge_index[0], edge_index[1]))
    G.add_edges_from(edges)
    return G


def preprocess_binary(labels, dataset_name):
    labels = np.array(labels)
    if dataset_name in ["cora", "citeseer"]:
        # Follow common binary conversion: Class 0 vs. rest
        pos_class = 0
    elif dataset_name == "pubmed":
        pos_class = 3
    else:
        pos_class = 1
    return np.array([1 if l == pos_class else -1 for l in labels])


def load_and_process_dataset(hf_name):
    dataset = load_dataset(f"graphbenchmark/{hf_name}")
    data = dataset['train'][0]
    edge_index = data['edge_index']
    labels = preprocess_binary(data['labels'])
    num_nodes = len(labels)
    G = hf_to_nx(edge_index, num_nodes)
    nodes = np.arange(num_nodes)
    train_nodes, test_nodes = train_test_split(nodes, test_size=0.2, random_state=42, shuffle=True)

    true_labels = {i: labels[i] for i in range(num_nodes)}
    split = {
        'train': set(train_nodes),
        'test': set(test_nodes)
    }
    return G, true_labels, split


def load_cora():
    return load_and_process_dataset("cora")


def load_citeseer():
    return load_and_process_dataset("citeseer")


def load_pubmed():
    return load_and_process_dataset("pubmed")


def load_dataset(name):
    if name == 'cora':
        return load_cora()
    elif name == 'citeseer':
        return load_citeseer()
    elif name == 'pubmed':
        return load_pubmed()
    else:
        raise ValueError(f"Unknown dataset {name}")

