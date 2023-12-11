from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import degree
from torch import nn, optim, Tensor
import torch.nn.functional as F
import torch
import src.data_processing as data_processing

# defines LightGCN model using the matrix formula from the paper
class LightGCN_Matrixform(MessagePassing):
    def __init__(self, num_users, num_items, embedding_dim=64, K=3, add_self_loops=False, dropout_rate=0.1):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.K = K
        self.add_self_loops = add_self_loops
        self.users_emb = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.embedding_dim)  # e_u^0
        self.items_emb = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.embedding_dim)  # e_i^0
        # "Initialize with normal distribution"
        nn.init.normal_(self.users_emb.weight, std=0.1)
        nn.init.normal_(self.items_emb.weight, std=0.1)
        # create a linear layer (fully connected layer) so we can output a single value (predicted_rating)
        self.out = nn.Linear(embedding_dim + embedding_dim, 1)

    def forward(self, edge_index: Tensor):
        """Forward propagation of LightGCN Mode. Args:edge_index (SparseTensor): adjacency matrix
            compute \tilde{A}: symmetrically normalized adjacency matrix. \tilde_A = D^(-1/2) * A * D^(-1/2)  """
        edge_index_norm = gcn_norm(edge_index=edge_index,add_self_loops=self.add_self_loops)
        # concat the user_emb and item_emb as the layer0 embing matrix
        emb_0 = torch.cat([self.users_emb.weight, self.items_emb.weight])  # E^0
        embs = [emb_0]  # save the layer0 emb to the embs list
        emb_k = emb_0 # emb_k is the emb that we are actually going to push it through the graph layers
        # push the embedding of all users and items through the Graph Model K times.
        # K here is the number of layers
        for i in range(self.K):
            emb_k = self.propagate(edge_index=edge_index_norm[0], x=emb_k, norm=edge_index_norm[1])
            embs.append(emb_k)
        embs = torch.stack(embs, dim=1)
        emb_final = torch.mean(embs, dim=1)  # E^K

        src = edge_index[0][:int(len(edge_index[0])/2)]
        dest = edge_index[1][:int(len(edge_index[0])/2)]

        # applying embedding lookup to get embeddings for src nodes and dest nodes in the edge list
        user_embeds = emb_final[src]
        item_embeds = emb_final[dest]

        # output dim: edge_index_len x 128 (given 64 is the original emb_vector_len)
        output = torch.cat([user_embeds, item_embeds], dim=1)
        output = self.out(output) # push it through the linear layer   #ALTERNATIVE METHOD

        #output = torch.sum((user_embeds * item_embeds), 1).view(-1,1)
        return output

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class NGCFConv(MessagePassing):
    def __init__(self, num_users, num_items, embedding_dim=64, K=3, add_self_loops=False, dropout_rate=0.1, bias=True,
                 **kwargs):
        super(NGCFConv, self).__init__(aggr='add', **kwargs)
        self.embedding_dim = embedding_dim
        self.num_users = num_users
        self.num_items = num_items
        self.embedding = nn.Embedding(self.num_users + self.num_items, self.embedding_dim)
        self.K = K
        self.add_self_loops = add_self_loops
        self.dropout = dropout_rate
        self.lin_1 = nn.Linear(self.embedding_dim, self.embedding_dim, bias=bias)
        self.lin_2 = nn.Linear(self.embedding_dim, self.embedding_dim, bias=bias)
        # create a linear layer (fully connected layer) so we can output a single value (predicted_rating)
        self.final = nn.Linear((self.embedding_dim + self.embedding_dim) * (self.K + 1), 1)
        # self.final = nn.Linear((self.embedding_dim + self.embedding_dim), 1)

        nn.init.xavier_uniform_(self.lin_1.weight)
        nn.init.xavier_uniform_(self.lin_2.weight)
        nn.init.xavier_uniform_(self.embedding.weight, gain=1)

    def forward(self, edge_index):
        emb0 = self.embedding.weight
        embs = [emb0]  # save the layer0 emb to the embs list
        emb_k = emb0  # emb_k is the emb that we are actually going to push it through the graph layers

        # Calculate the normalization
        from_, to_ = edge_index
        # edge_index_norm = gcn_norm(edge_index=edge_index, add_self_loops=self.add_self_loops)
        deg = degree(to_, emb_k.size(0), dtype=emb_k.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[from_] * deg_inv_sqrt[to_]

        # Start propagating messages
        for i in range(self.K):
            out = self.propagate(edge_index=edge_index, x=(emb_k, emb_k), norm=norm)
            # out = self.propagate(edge_index=edge_index_norm[0], x=(emb_k, emb_k), norm=edge_index_norm[1])
            # Perform update after aggregation
            out += self.lin_1(emb_k)
            out = F.dropout(out, self.dropout, self.training)
            out = F.leaky_relu(out)
            emb_k = out
            embs.append(emb_k)

        emb_final = torch.cat(embs, dim=-1)
        #emb_final = torch.mean(torch.stack(embs, dim=1), dim=1)  #Only use if imitating LightGCN congregation method

        src = edge_index[0][:int(len(edge_index[0]) / 2)]
        dest = edge_index[1][:int(len(edge_index[0]) / 2)]
        # applying embedding lookup to get embeddings for src nodes and dest nodes in the edge list
        user_embeds = emb_final[src]
        item_embeds = emb_final[dest]

        '''output = torch.cat([user_embeds, item_embeds], dim=1)
        output = self.final(output)  # push it through the linear layer   #ALTERNATIVE METHOD'''

        output = torch.sum((user_embeds * item_embeds), 1).view(-1,1)
        return output

    def message(self, x_j, x_i, norm):
        return norm.view(-1, 1) * (self.lin_1(x_j) + self.lin_2(x_j * x_i))


class LightGCNConv(MessagePassing):
    def __init__(self, num_users, num_items, embedding_dim=64, K=3, add_self_loops=False, dropout_rate=0.1, bias=True,
                 **kwargs):
        super().__init__(aggr='add')
        #super().__init__()
        self.embedding_dim = embedding_dim
        self.num_users = num_users
        self.num_items = num_items
        self.embedding = nn.Embedding(self.num_users + self.num_items, self.embedding_dim)
        self.K = K
        self.add_self_loops = add_self_loops
        self.dropout = dropout_rate

        # create a linear layer (fully connected layer) so we can output a single value (predicted_rating)
        self.out = nn.Linear(self.embedding_dim + self.embedding_dim, 1)
        nn.init.normal_(self.embedding.weight, std=0.1)

    def forward(self, edge_index):
        emb0 = self.embedding.weight
        embs = [emb0]  # save the layer0 emb to the embs list
        emb_k = emb0  # emb_k is the emb that we are actually going to push it through the graph layers

        # Compute normalization
        from_, to_ = edge_index
        #edge_index_norm = gcn_norm(edge_index=edge_index, add_self_loops=self.add_self_loops)  #Alternative method
        deg = degree(to_, emb_k.size(0), dtype=emb_k.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[from_] * deg_inv_sqrt[to_]

        # Start propagating messages
        for i in range(self.K):
            emb_k = self.propagate(edge_index=edge_index, x=emb_k, norm=norm)
            #emb_k = self.propagate(edge_index=edge_index_norm[0], x=emb_k, norm=edge_index_norm[1])  #Alternative method
            embs.append(emb_k)

        emb_final = torch.mean(torch.stack(embs, dim=0), dim=0)
        '''embs = torch.stack(embs, dim=1)
        emb_final = torch.mean(embs, dim=1)  # E^K'''  #Alternative method

        src = edge_index[0][:int(len(edge_index[0]) / 2)]
        dest = edge_index[1][:int(len(edge_index[0]) / 2)]
        # applying embedding lookup to get embeddings for src nodes and dest nodes in the edge list
        user_embeds = emb_final[src]
        item_embeds = emb_final[dest]

        output = torch.cat([user_embeds, item_embeds], dim=1)
        output = self.out(output)  # push it through the linear layer   #ALTERNATIVE METHOD

        #output = torch.sum((user_embeds * item_embeds), 1).view(-1,1)
        return output

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j




