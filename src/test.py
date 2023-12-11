from src import data_processing, GNN_model
import torch
import torch_geometric
from torch import nn, optim, Tensor
from tqdm.notebook import tqdm
from src import metrics
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import numpy as np
from sklearn.utils import gen_batches

mylist = np.array([[2,3,4,7,5.5,7,6,82,8], [5,7,4,35,0,5 ,20, 6,7]]).T
labels = np.array([1,15,23,4,20,34,55,42,3])

mylist, labels = shuffle(mylist, labels)
#labels = np.reshape(labels, [len(labels),1])

batched = gen_batches(len(mylist), 4)
batched2 = gen_batches(len(mylist), 4)
batches_label = [labels[batch] for batch in batched]
batches = [mylist[batch].T for batch in batched2]






class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, edge_index, edge_values):
        self.data = edge_index
        self.label = edge_values

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.data[:,idx], self.label[idx]

train_edge_index = torch.LongTensor([[2,4,6,8,10,13,3], [2.5,4.5,6.5,8.5,10.5,7,8]])
train_edge_values = torch.LongTensor([3,6,3,6,7,8.8,0])

dataset = SimpleDataset(train_edge_index, train_edge_values)
#train_edge_index = torch.transpose(train_edge_index,0,1)

# put data into loaders
train_loader = torch.utils.data.DataLoader(dataset, batch_size=3, shuffle=True)

train_loader = enumerate(train_loader)
for batch_idx, (data, target) in train_loader:
    print (batch_idx)
    print (torch.transpose(data,0,1))
    print (target)

print ("")