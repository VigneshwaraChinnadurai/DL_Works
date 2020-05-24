# Boltzmann Machines

# Restricted Boltzmann machines have received a lot of attention recently after being proposed as 
# building blocks of multi-layer learning architectures called deep belief networks.
# These features can serve as input to another RBM. 
# By stacking RBMs in this way, one can learn features from features in the hope of arriving at a high level representation. 

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
# ml-1m/movies.dat is the key to get into that sub directory and fetch the dataset.
# And seperator in this .data file is ::
# We specify engine to get the import correctly.
# Encoding is used as some of the movie title have some special character.
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
# In this rating.dat, 1st column corresponds to users, 2nd to movies, 3rd to ratings and 4th is timestamp (not used.)

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
        # After data[:,1], this is given as condition
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        # As the id_movies starts from 1 and program starts index from zero, to avoid mismatching, giving -1
        new_data.append(list(ratings))
        # As torch expeccts list of lists, confirming that itis a list.
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)
# By the above procedure,we're converting the data in the dataframe to list

# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)
# Here we're converting the data in LIST to torch tensors.
# After converting the datas to tensors, it may disappear in variable explorer as it is much advanced.

# Converting the ratings into binary ratings 1 (Liked) or 0 (Not Liked)
# As we are giving the ratings in zeros and ones, the present zeros which represents the ratings which is not given by the user
# must be changed to another value, hence changing it to -1
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1
test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

# Creating the architecture of the Neural Network
# Note: Creating a class should always start with capital letter.
class RBM():
    # Should always start with init function as it says the objects once the class is made
    # Default and compulsory
    def __init__(self, nv, nh):
        # Self is default which corresponds to the objects which will be created afterwards.
        # where nh is no of hidden nodes and nv is no of visible nodes
        # Then initialize the parameters of our future objects.(Initialize the objects of the class)
        self.W = torch.randn(nh, nv)
        # Here the weights(W) are initialized by the torch tensors.
        self.a = torch.randn(1, nh)
        # Here the bias is initialized for hidden nodes and we'll do same for visible nodes.
        self.b = torch.randn(1, nv)
    def sample_h(self, x):
        # Taking the samples of hidden neuron where x is visible neurons v in a given probability p(h) given v
        wx = torch.mm(x, self.W.t())
        # This is to calculate weight (w) of given neurons (x) plus the bias (a)
        # torch.mm is to multiply the tensors and t is to take transpose.
        activation = wx + self.a.expand_as(wx)
        # Explaining the activation function.
        # Basically, Activation= Weight + Bias   
        # Expand function is used to confirm whether it is applied for each line of mini batch considered.
        p_h_given_v = torch.sigmoid(activation) 
        return p_h_given_v, torch.bernoulli(p_h_given_v)
        # Since we're predicting binary outcome, liked or not, we use Bernoulli distribution
        # As per the bernoulli, we finally get sampling of the hidden nodes which are activated when the p_h_given_v<0.7
    def sample_v(self, y):
        # Taking the samples of visible neuron where y is hidden neurons h in a given probability p(v) given h
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    def train(self, v0, vk, ph0, phk):
        # v0 is input vector, vk is visible node obtained after k samplings, ph0 is probability of hidden node
        # and phk is probability of hidden node after k samplings.
        self.W += (torch.mm(v0.t(),ph0) - torch.mm(vk.t(),phk)).t()
        # Doing exactly as per reference paper instruction (Pg:28) for W,b,a
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)
nv = len(training_set[0])
nh = 100 
# giving 100 for the no of features to predict
batch_size = 100
# Batch size is the interval at which the weights to be updated.
# To get a fasyer prediction, giving batch size as 100.
rbm = RBM(nv, nh)
# It creates the nb_movies which is the movies which is not actually rated.

# Training the RBM
nb_epoch = 10
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    # S is the no of person who atleast gave rating for 1 movie and initializing it to be zero.
    for id_user in range(0, nb_users - batch_size, batch_size):
        # range: stop at 843(nb_users - batch_size) as we are stpping every 100, ie 1 to 100 then 100 to 200.
        # So by this steping, we can update the weights after 100 inputs.
        vk = training_set[id_user:id_user+batch_size]
        v0 = training_set[id_user:id_user+batch_size]
        ph0,_ = rbm.sample_h(v0)
        # ph0,_ is the trick to get the first element of the function return.
        for k in range(10):
            _,hk = rbm.sample_h(vk)
            # _,hk is the trick to get the second element of the function return.
            _,vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0]
            # By this we're not touching the column where the users didn't rate the movies (-1)
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        s += 1.
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))
    # Str is to convert the epoc value to number.

# Testing the RBM
test_loss = 0
s = 0.
for id_user in range(nb_users):
    v = training_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1]
    if len(vt[vt>=0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        s += 1.
print('test loss: '+str(test_loss/s))



"""
To calculate the loss using Root Mean Square value, Simply change with the below.

In Training Phase:

nb_epoch = 10
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(0, nb_users - batch_size, batch_size):
        vk = training_set[id_user:id_user+batch_size]
        v0 = training_set[id_user:id_user+batch_size]
        ph0,_ = rbm.sample_h(v0)
        for k in range(10):
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0]
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += np.sqrt(torch.mean((v0[v0>=0] - vk[v0>=0])**2)) # RMSE here
        s += 1.
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))
    
    
    
In Test Phase:

test_loss = 0
s = 0.
for id_user in range(nb_users):
    v = training_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1]
    if len(vt[vt>=0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += np.sqrt(torch.mean((vt[vt>=0] - v[vt>=0])**2)) # RMSE here
        s += 1.
print('test loss: '+str(test_loss/s))


The current method we use is Average Distance method. """