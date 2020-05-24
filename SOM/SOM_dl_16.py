# Self Organizing Map

# These are unsupervised deep learning in which the output is not given in the dataset which is found with the 
# previous inputs.

# In this dataset, we are going to predict the no of frauds who applied and received credit cards.

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
# Feature range is to give the values between 0 and 1
X = sc.fit_transform(X)

# Training the SOM
# Or you can simply pip the command
# pip install -i https://test.pypi.org/simple/ MiniSom==1.0
from minisom import MiniSom
# Importing the minisom file which is kept in working directory.(minisom.py)
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
# As it is unsupervised DL, the input and output for the prediction is same(X)
# As we are having 14 columns, 14 dimentions, we selecting a random value as x and y (Say 10)
# Higher the learning rate faster the convergence (fits to data)
# Lower the learning rate higher the model takes (Doesnot procduce prediction correctly)
# Hence taking default of 0.5
som.random_weights_init(X)
# In this algorithm, we need to initialize the weights close to zero.
som.train_random(data = X, num_iteration = 100)
# choosing the number of iterations to 100.

# Visualizing the results
# Here we are not using the default maps, but we are goingto create one of our own desire to fit SOM Model.
from pylab import bone, pcolor, colorbar, plot, show
bone()
# It will just create a white window. (Just run bone() to see the output)
pcolor(som.distance_map().T)
# To show the value as color, we do the above step. (Like color gradien in Tableau)
# dotT is used to transpose the matrix as required to input it to pcolor
colorbar()
# It gives the legends.
# Hence the graph is showing the Mean Interneuron Distance.
markers = ['o', 's']
# Circle(o) is the customers who didn't get approval and the Squares(s) is the one who got approval.
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

# Finding the frauds
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(8,1)], mappings[(6,8)]), axis = 0)
# By seeing the graph, giving the co-ordinate to fetch the details using the dict.
# Need to give which ever cell has pure or near pure white with values ie circle or square. 
frauds = sc.inverse_transform(frauds)