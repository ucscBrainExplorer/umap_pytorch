import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

def get_graph_elements(graph_, n_epochs):

    graph = graph_.tocoo()
    # eliminate duplicate entries by summing them together
    graph.sum_duplicates()
    # number of vertices in dataset
    n_vertices = graph.shape[1]
    # get the number of epochs based on the size of the dataset
    if n_epochs is None:
        # For smaller datasets we can use more epochs
        if graph.shape[0] <= 10000:
            n_epochs = 500
        else:
            n_epochs = 200
    # remove elements with very low probability
    graph.data[graph.data < (graph.data.max() / float(n_epochs))] = 0.0
    graph.eliminate_zeros()
    # get epochs per sample based upon edge probability
    epochs_per_sample = n_epochs * graph.data

    head = graph.row
    tail = graph.col
    weight = graph.data

    return graph, epochs_per_sample, head, tail, weight, n_vertices

class UMAPDataset(Dataset):
    def __init__(self, data, graph_, batch_size=256, n_epochs=200):
        print("Start UMAPDataset initiation ...", flush=True)
        print ("batch_size:", batch_size)
        graph, epochs_per_sample, head, tail, weight, n_vertices = get_graph_elements(graph_, n_epochs)

        self.data = torch.tensor(data, device="cuda")
        self.probs = torch.tensor(epochs_per_sample / epochs_per_sample.sum(), device="cuda")
        self.head = torch.tensor(head, device="cuda")
        self.tail = torch.tensor(tail, device="cuda")
        self.batch_size = batch_size
        self.total = int(epochs_per_sample.sum())

        '''        
        self.edges_to_exp, self.edges_from_exp = (
            np.repeat(head, epochs_per_sample.astype("int")),
            np.repeat(tail, epochs_per_sample.astype("int")),
        )
        shuffle_mask = np.random.permutation(np.arange(len(self.edges_to_exp)))
        self.edges_to_exp = self.edges_to_exp[shuffle_mask].astype(np.int64)
        self.edges_from_exp = self.edges_from_exp[shuffle_mask].astype(np.int64)
        self.total = len(self.edges_to_exp)
        '''
        print("End UMAPDataset.", self.total, self.data.shape[0], flush=True)
        
    def __len__(self):
        #return int(self.total/self.batch_size)
        return int(self.data.shape[0]/self.batch_size)
    
    def __getitem__(self, index):
        # Instead of expanding, sample
        chunk_size = 10_000_000
        if len(self.probs) < chunk_size:
            # Dataset too small for chunking - use all edges
            edge_idx = torch.multinomial(self.probs, self.batch_size, replacement=True)
        else:
            N = len(self.probs)//chunk_size
            chunk_size = len(self.probs)//N
            chunk_idx = torch.randint(0, N, (1,), device="cuda")
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, len(self.probs))
            edge_idx = torch.multinomial(self.probs[start:end], self.batch_size, replacement=True) + start

        edges_to_exp= self.data[self.head[edge_idx]]
        edges_from_exp = self.data[self.tail[edge_idx]]

        return (edges_to_exp, edges_from_exp)
    
class MatchDataset(Dataset):
    def __init__(self, data, embeddings):
        self.embeddings = torch.Tensor(embeddings)
        self.data = data
    def __len__(self):
        return int(self.data.shape[0])
    def __getitem__(self, index):
        return self.data[index], self.embeddings[index]
