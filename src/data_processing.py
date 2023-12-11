import pandas as pd
import numpy as np
from sklearn import model_selection, metrics, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.utils import gen_batches
import torch
from torch_sparse import SparseTensor, matmul
from sklearn.utils import shuffle

def encode_data(data_path, ratings_threshold, csv=True):
    # Load into a pandas dataframe, the dataset that has users and the ratings they gave to particular movies
    if csv:
        ratings = pd.read_csv(data_path)  # for csv
    else:
        rnames = ['userId', 'movieId', 'rating', 'timestamp']
        ratings = pd.read_csv(data_path, delimiter='::', engine='python', header=None,names=rnames)  # for .dat file

    #Encode the userId, and MovieId values (from index 0 to the no. of unique IDs)
    labelencode = preprocessing.LabelEncoder()
    ratings.userId, ratings.movieId = labelencode.fit_transform(ratings.userId.values), labelencode.fit_transform(ratings.movieId.values)

    #Get the edge values (i.e. the ratings between a user and a movie)
    src = [user_id for user_id in ratings['userId']]
    num_users = len(ratings['userId'].unique())
    dst = [(movie_id) for movie_id in ratings['movieId']]

    link_vals = ratings['rating'].values

    #Only consider ratings that are higher than the specified threshold
    edge_attr = torch.from_numpy(ratings['rating'].values).view(-1, 1).to(torch.long) >= ratings_threshold

    edge_values = []
    edge_index = [[], []]
    for i in range(edge_attr.shape[0]):
        if edge_attr[i]:
            edge_index[0].append(src[i])
            edge_index[1].append(dst[i])
            edge_values.append(link_vals[i])

    # edge_values is the label we will use for compute training loss, edge index is the list of user-movie pairings
    return ratings, edge_index, edge_values

def convert_to_longtensor(data):
    data = torch.LongTensor(data)
    return data

def data_split(edge_index, edge_values, validation_ratio, test_ratio):
    num_interactions = edge_index.shape[1]  #Obtain no. of edges
    all_indices = [i for i in range(num_interactions)]

    #Obtain indices that will be used for solitting the data
    train_indices, test_indices = train_test_split(all_indices, test_size=validation_ratio + test_ratio, random_state=1)
    val_indices, test_indices = train_test_split(test_indices, test_size=test_ratio/(validation_ratio + test_ratio), random_state=1)

    #Split the data
    train_edge_index, train_edge_value = edge_index[:, train_indices], edge_values[train_indices]
    val_edge_index, val_edge_value = edge_index[:, val_indices], edge_values[val_indices]
    test_edge_index, test_edge_value = edge_index[:, test_indices], edge_values[test_indices]

    return [train_edge_index, train_edge_value], [val_edge_index, val_edge_value], [test_edge_index, test_edge_value]


class BatchDataset(torch.utils.data.Dataset):
    def __init__(self, edge_index, edge_values):
        self.data = edge_index
        self.label = edge_values

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.data[:,idx], self.label[idx]

def batch_data (training_dataset, batchsize):
    training_data, training_label = training_dataset
    training_data, training_label = shuffle(torch.transpose(training_data, 0, 1), training_label)
    batched = gen_batches(len(training_data), batchsize)
    batched2 = gen_batches(len(training_data), batchsize)
    batches_label = [training_label[batch] for batch in batched]
    batches_data = [training_data[batch].T for batch in batched2]

    return batches_data, batches_label




