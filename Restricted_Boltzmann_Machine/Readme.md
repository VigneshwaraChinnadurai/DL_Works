Hi All,

This is a recommender system using Boltzmann Machineusing movie lens dataset (available here: https://grouplens.org/datasets/movielens/ )

We have evaluated our RBM model using Average Distance method in calculating loss function.

If needed, we can also create model using Root Mean Squared Error method by simply changing the below.


train_loss += np.sqrt(torch.mean((v0[v0>=0] - vk[v0>=0])**2))

Similarly for test phase.

Happy Coding.
