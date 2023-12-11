from src import data_processing, GNN_model
import torch
from torch import nn, optim, Tensor
from tqdm.notebook import tqdm
from src import metrics
import matplotlib.pyplot as plt


def train(net, optimizer, epoch, eval=True):

    batched_data, batched_label = data_processing.batch_data(trainset, BATCH_SIZE)
    for batch_idx in range(len(batched_data)):
        data, target = batched_data[batch_idx], batched_label[batch_idx] #Alternative way of batching the data

    #for batch_idx, (data, target) in enumerate(train_loader):       #First method of batching (It makes code slow)
        #data = torch.transpose(data, 0, 1)  # This would not be needed if we use the alternative way of batching

        net.train()  # prepare model for training (only important for dropout, batch norm, etc.)
        data, target = data.to(device), target.to(device)
        data[1] = data[1] + num_users
        data = torch.stack((torch.cat([data[0], data[1]]), torch.cat([data[1], data[0]])))
        pred_ratings = net.forward(data)
        train_loss = loss_func(pred_ratings, target.view(-1, 1))
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        batchidx = batch_idx

    if eval == True:
        # going over validation set
        net.eval()
        with torch.no_grad():
            val_pred_ratings = net.forward(val_edge_index)
            val_loss = loss_func(val_pred_ratings, val_edge_values.view(-1, 1)).sum()
            recall_at_k, precision_at_k = metrics.get_recall_at_k(valset[0], valset[1], val_pred_ratings, k=20)
            val_recall_at_ks.append(round(recall_at_k, 5))
            train_losses.append(train_loss.item())
            val_losses.append(val_loss.item())

            print(
                f"[epoch {epoch + 1}/No. of iterations: {batchidx + 1}], train_loss: {round(train_loss.item(), 5)}, "
                f"val_loss: {round(val_loss.item(), 5)},  recall_at_k {round(recall_at_k, 5)}, "
                f"precision_at_k {round(precision_at_k, 5)}")
    else:
        eval_marker = 10
        print(f"[epoch {epoch + 1}/No. of iterations: {batchidx + 1}], train_loss: {round(train_loss.item(), 5)}, ")
        if epoch % eval_marker == 0 and epoch != 0:  # Evaluation slows down training time. Only do it at intervals
            net.eval()
            with torch.no_grad():
                val_pred_ratings = net.forward(val_edge_index)
                val_loss = loss_func(val_pred_ratings, val_edge_values.view(-1, 1)).sum()
                recall_at_k, precision_at_k = metrics.get_recall_at_k(valset[0], valset[1], val_pred_ratings, k=20)
                val_recall_at_ks.append(round(recall_at_k, 5))
                train_losses.append(train_loss.item())
                val_losses.append(val_loss.item())

                print(
                    f"[epoch {epoch + 1}/No. of iterations: {batchidx + 1}], train_loss: {round(train_loss.item(), 5)}, "
                    f"val_loss: {round(val_loss.item(), 5)},  recall_at_k {round(recall_at_k, 5)}, "
                    f"precision_at_k {round(precision_at_k, 5)}")



#Load data
data_path = './data/ml-1m/ratings.dat'
#data_path = './data/ml-latest-small/ratings.csv'
ratings_threshold = 1

#Make sure to set csv to True if the data file is a csv file
ratings, edge_index, edge_values = data_processing.encode_data(data_path, ratings_threshold, csv=False) #Encode data and select only user-movie ratings >= threshold

num_users, num_movies = len(ratings['userId'].unique()), len(ratings['movieId'].unique())
print (f"No of users is: {num_users} and No of Movies is: {num_movies}")

# Convert to tensor
# LongTensor is used because the .propagate() method in the model needs either LongTensor or SparseTensor
edge_index = data_processing.convert_to_longtensor(edge_index)  #longTensor is simply int64 format
edge_values = torch.FloatTensor(edge_values)  #The edge values (ratings) have to remain as floats

#Split the data into training, test and validation sets
validation_ratio, test_ratio = 0.1, 0.1
trainset, valset, testset = data_processing.data_split(edge_index, edge_values, validation_ratio, test_ratio)

#Convert the indices of the data into an adjacency matrix format so that each node has its own unique identifier (index)
#Doing this will prevent a user node and movie node from having the same index
#Change the movie indices by simply adding the no. of users while leaving the user indices untouched
val_edge_index, val_edge_values = valset[0], valset[1]
val_edge_index[1] = val_edge_index[1] + num_users
test_edge_index, test_edge_values = testset[0], testset[1]
test_edge_index[1] = test_edge_index[1] + num_users

#Represent the graph as a sparse list of node pairs and since the graph is undirected, we need to include each edge twice
#Once for the edges from the users to the movies and vice versa (This is the adjacency matrix sparse list)
val_edge_index = torch.stack((torch.cat([val_edge_index[0], val_edge_index[1]]), torch.cat([val_edge_index[1], val_edge_index[0]])))
test_edge_index = torch.stack((torch.cat([test_edge_index[0], test_edge_index[1]]), torch.cat([test_edge_index[1], test_edge_index[0]])))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_edge_index, test_edge_values = test_edge_index.to(device), test_edge_values.to(device)
val_edge_index, val_edge_values = val_edge_index.to(device), val_edge_values.to(device)
loss_func = nn.MSELoss()

'''Train using LightGCN'''
#Initialize Model
layers = 2
model1 = GNN_model.LightGCN_Matrixform(num_users=num_users, num_items=num_movies, K=layers)
model1 = model1.to(device)
# define parameters
EPOCHS = 500
BATCH_SIZE = 10240 #len(edge_values) #if we don't want batches
LR = 1e-3
EPOCH_PER_LR_DECAY = 100
K = 10
LAMBDA = 1e-6

optimizer = optim.Adam(model1.parameters(), lr=LR, weight_decay=0.01)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

# training loop
train_losses, val_losses, val_recall_at_ks = [],[], []
# Prepare the data to be put into a loader so that we can use batching
#train_dataset = data_processing.BatchDataset(trainset[0], trainset[1])  #comment this line if we use a different way of batching

'''for epoch in tqdm(range(EPOCHS)):
    #train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True) #comment if we use the alternative way of batching
    train(model1, optimizer, epoch, eval=False)
    if epoch % EPOCH_PER_LR_DECAY == 0 and epoch != 0:
        scheduler.step()


#Make the Learning plots
iters = [epch + 1 for epch in range(len(train_losses))]
plt.plot(iters, train_losses, label='train')
plt.plot(iters, val_losses, label='validation')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.title('training and validation loss curves')
plt.legend()
plt.show()

f2 = plt.figure()
plt.plot(iters, val_recall_at_ks, label='recall_at_k')
plt.xlabel('iteration')
plt.ylabel('recall_at_k')
plt.title('recall_at_k curves')
plt.show()

#Evaluate the model on the test set
model1.eval()
with torch.no_grad():
    pred_ratings = model1.forward(test_edge_index)
    recall_at_k, precision_at_k = metrics.get_recall_at_k(testset[0],testset[1],pred_ratings, 20)
    print(f"Test dataset recall_at_k {round(recall_at_k, 5)}, precision_at_k {round(precision_at_k, 5)}")'''

'''Train Using NGCF'''
model2 = GNN_model.NGCFConv(num_users=num_users, num_items=num_movies, embedding_dim=64, K=layers)
#model2 = GNN_model.LightGCNConv(num_users=num_users, num_items=num_movies, embedding_dim=64, K=layers)
model2 = model2.to(device)
optimizer = optim.Adam(model2.parameters(), lr=LR, weight_decay=0.01)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

# training loop
train_losses, val_losses, val_recall_at_ks = [],[], []

for epoch in tqdm(range(EPOCHS)):
    train(model2, optimizer, epoch, eval=False)
    if epoch % EPOCH_PER_LR_DECAY == 0 and epoch != 0:
        scheduler.step()


#Make the Learning plots
iters = [epch + 1 for epch in range(len(train_losses))]
plt.plot(iters, train_losses, label='train')
plt.plot(iters, val_losses, label='validation')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.title('training and validation loss curves')
plt.legend()
plt.show()

f2 = plt.figure()
plt.plot(iters, val_recall_at_ks, label='recall_at_k')
plt.xlabel('iteration')
plt.ylabel('recall_at_k')
plt.title('recall_at_k curves')
plt.show()

#Evaluate the model on the test set
model2.eval()
with torch.no_grad():
    pred_ratings = model2.forward(test_edge_index)
    recall_at_k, precision_at_k = metrics.get_recall_at_k(testset[0],testset[1],pred_ratings, 20)
    print(f"Test dataset recall_at_k {round(recall_at_k, 5)}, precision_at_k {round(precision_at_k, 5)}")