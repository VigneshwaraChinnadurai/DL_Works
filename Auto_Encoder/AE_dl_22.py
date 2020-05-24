# AutoEncoders

# AE cannot have same or more no of hidden layers than input or visible layers as they wont function properly.
# It just copies the node information to another node by dedicating the nodes to input ie 1st hidden node to 1st visible node.

# Types of AE: Sparse, Denoising, Contractive, Stacked, Deep.

# In sparse encoder, a regularisation technique, sparcity, which is applied to use only selected no of hidden layers for a case,
# at a given time, which is less than the no of input layers wich makes the model robust and efficient working.

# In  Denoising encoder, it replaces the input by modified input in which some of th inputs are made as zeros
# which is according to our wish and it can be changed by our coding. And at last the output is compared with 
# original input and not modified input values.

# Contractive AE is very complex. So refer additional resources

# Stacked AE is adding another hidden layer to the network.

# Stacked AE <> Deep AE. Deep AE is a RBM (restricted boltsman machine) stacked to normal auto encoder,
# but looks similar to stacked AE

# In this problem we're going to model based upon the users rating to movies which they've watched and
# predicting the movies's ratings they probabily give to the movies they have'nt watched.

# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Importing the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# Preparing the training set and the test set
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

# Getting the number of users and movies
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# Converting the data into an array with users in lines and movies in columns
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Creating the architecture of the Neural Network
class SAE(nn.Module):
    # Here we're defining that we're inheriting the features of parent class (nn.Module from pyTorch)
    def __init__(self, ):
        super(SAE, self).__init__()
        # This is to inherit the features from parent class.
        # This architecture is similar to the Deep Auto Encoders.
        self.fc1 = nn.Linear(nb_movies, 20)
        # Here te input is nb_movies and the output is 20 nodes.
        # Try different values in the place of 20 to get a better result.(Practical experienc)
        self.fc2 = nn.Linear(20, 10)
        # This is first hidden layer
        # 20 is the input to this layer, 10 is the output from this layer.
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, nb_movies)
        self.activation = nn.Sigmoid()
        # Declaring the activation function(rectifier activation function can also be given)
        # Upon testing, Sigmoid gave more accuracy.
    def forward(self, x):
        x = self.activation(self.fc1(x))
        # The above step returns first encoded vector
        # Where x is the first input vector of features (nb_movies).
        x = self.activation(self.fc2(x))
        # The above step returns second encoded vector
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        # The above step returns decoded vector
        return x
sae = SAE()
# For objects, try using non capital letters.
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)
# lr is Learning Rate which is arbitary, try giving different values according to business problems.
# Found best lr as 0.01

# Training the SAE
nb_epoch = 200
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    # S is the no of person who atleast gave rating for 1 movie and initializing it to be zero.
    for id_user in range(nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0)
        # Adding a new dimension with the variable function. 
        # In the above step, we're creating the batch of single input vector (which is expected by pyTorch).
        # Batch creation is essential, else it wont work.
        target = input.clone()
        # Cloning the input as target.
        if torch.sum(target.data > 0) > 0:
            # This If condition is to remove users who didn't vote even a single movie as they are of no use.
            output = sae(input)
            target.require_grad = False
            # Defining the network not to use this for computing gradient.
            output[target == 0] = 0
            loss = criterion(output, target)
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
            # As the torch.sum(target.data) is denominator, we should not make it as zero at any condition.
            # Hence adding 1e-10 a number nearly equal to zero but not absolute zero to it.
            # Hence by this we're avoiding the infinite computation error.
            loss.backward()
            train_loss += np.sqrt(loss.data*mean_corrector)
            # That's because in PyTorch>=0.5, the index of 0-dim tensor is invalid. 
            # The master branch is designed for PyTorch 0.4.1, loss_val.data[0] works well.
            s += 1.
            optimizer.step()
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))

# Testing the SAE
test_loss = 0
s = 0.
for id_user in range(nb_users):
    input = Variable(training_set[id_user]).unsqueeze(0)
    target = Variable(test_set[id_user]).unsqeeze(0)
    if torch.sum(target.data > 0) > 0:
        output = sae(input)
        target.require_grad = False
        output[target == 0] = 0
        loss = criterion(output, target)
        mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
        test_loss += np.sqrt(loss.item()*mean_corrector)
        s += 1.
print('test loss: '+str(test_loss/s))