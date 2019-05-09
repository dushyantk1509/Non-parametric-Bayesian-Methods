import os
import _pickle as cPickle
import gzip
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
from scipy.stats import multivariate_normal
from scipy import stats
import seaborn as sns
from tqdm import tqdm
import sys
from sklearn.utils import shuffle
import pickle
from collections import Counter

import itertools

sigma = 150                   
mean_0 = np.array([0,0])                  # mean for sampling means for clusters
cov_0 = np.array( [[sigma, 0],[0, sigma]] )             # covariance for sampling means for clusters
cov = [[1,0], [0, 1]]

## TO DO 
'''
	1. Color code the original plots						DONE
	2. Save dataset in pickle 							DONE 	
	3. After every 5 epochs, save mu, z in pickle file				DONE 	
	4. Read X, mu, z and plot using color coding the different plots		DONE
'''


def dataset():
	
  # Creates a 2d mixture of 10 gaussians
  # Plots the scatter plot of the dataset

  # 20 clusters of total = 16000 points
  cluster_points = np.array([8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]) * 100
  K = len(cluster_points)

  fig, ax = plt.subplots()

  # c = cm.rainbow(np.linspace(0,1,K))

  for i in range(K):                 								# K different cluster means chosen
    cluster_mean = np.random.multivariate_normal(mean_0, cov_0)     # Choose a mean for new cluster 
    X_new = np.random.multivariate_normal(cluster_mean, cov, cluster_points[i]) # Allocate points for the cluster   

    ax.scatter(X_new[:, 0], X_new[:, 1]) # color=c[i]

    X = np.concatenate((X, X_new)) if i != 0 else X_new     
  
  plt.savefig("original_plot.jpg")

  np.save("X_train.npy", X)

  return X



def GibbsSampler(X_train, alpha, K):
  
  print("Running experiment for alpha, K values: Alpha = {0}, K = {1}".format(alpha, K))
  print("In gibbs sampler, shape of X_train : ", X_train.shape)

  # Create a folder to write the experiments results
  folder_name = "results_alpha_{0}_K_{1}".format(alpha, K)
  if os.path.exists(folder_name):
  	os.system("rm -r {0}".format(folder_name))
  os.system("mkdir {0}".format(folder_name))

  # Open a file to write results of the experiment
  f = open("{2}/cluster_counts_alpha_{0}_z_{1}.txt".format(alpha, K, folder_name), 'w+')
  
  alpha = float(alpha)
  K = int(K)

  N, D = X_train.shape
  maxIter = 31
  mu_0 = 0.0
  var_0 = 10.0
  var = 1.0

  mu_ = np.random.rand(K, D) # initial mean for K clusters
  z_ = np.random.randint(K, size = N) # initial cluster assignments for N data points
  n_ = Counter(z_)

  K_old = K


  for t in tqdm(range(maxIter)):

    # Every few iterations, save cluster means, cluster ids and show K value
    if t%1 == 0:
      print("Saving mean and z_ before " + str(t) + " iters")
      np.save("{1}/mean_iter_{0}.npy".format(t, folder_name), mu_)        # Save the cluster means  
      np.save("{1}/z_iter_{0}.npy".format(t, folder_name), z_)          # Save the clusters ids

      # Print to file
      n_ = Counter(z_)
      print("Print K value : " + str(K) + " actual no. of clusters : " + str(len(n_)), file=f)
      print("Printing cluster counts before " + str(t) + " iters", file=f)
      print(n_, file=f)                   # Save the cluster counts
      print("\n", file=f)

    for i in tqdm(range(N)):
      
      if K != K_old:
        print("K changed at for N = " + str(i) + " at iteration : " + str(t) + " New K = " + str(K))
        K_old = K

      phat_i_ = np.zeros(K + 1)

      for k in range(K):
        phat_i_[k] = np.log(n_[k]) - 0.5 * D * np.log(2 * np.pi) - 0.5 * np.log(var) - (1 / (2 * var)) * (np.linalg.norm(X_train[i] - mu_[k]) ** 2)
        
      phat_i_[K] = np.log(alpha) - 0.5 * D * np.log(2 * np.pi) - 0.5 * np.log(var_0 + var) - (1 / (2 * (var_0 + var))) * (np.linalg.norm(X_train[i] - mu_0) ** 2)

      phat_i_ = np.exp(phat_i_)
      phat_i_ = phat_i_ / np.sum(phat_i_)

      z_[i] = np.nonzero(np.random.multinomial(1,phat_i_))[0]

      # check if need to create a new cluster or not
      if(z_[i] == K):
        K += 1
        muhat = (var * mu_0 + var_0 * X_train[i]) / (var + var_0 )
        varhat = (var_0 * var) / (var + var_0 )

        cov = varhat * np.identity(D)
        new_mean = np.random.multivariate_normal(muhat, cov)
        new_mean = new_mean.reshape(1,D)
        mu_ = np.concatenate((mu_, new_mean), axis=0)

      n_ = Counter(z_)

    print(n_)
      
    # sample mean for the new clusters

    for k in range(K):

      i = z_ == k
      i = i.reshape(N,1)

      sumx = np.sum(np.multiply(X_train,i), axis=0)
      muhat = (var * mu_0 + var_0 * sumx) / (var + var_0 * n_[k])
      varhat = (var_0 * var) / (var + var_0 * n_[k])

      cov = varhat * np.identity(D)
      mu_[k] = np.random.multivariate_normal(muhat, cov)

    # Every few iterations, save cluster means, cluster ids and show K value
    # if t%5 == 0:
    #   print("Saving mean and z_ after " + str(t) + " iters")
    #   np.save("{1}/mean_iter_{0}.npy".format(t, folder_name), mu_)				# Save the cluster means  
    #   np.save("{1}/z_iter_{0}.npy".format(t, folder_name), z_)					# Save the clusters ids

    #   # Print to file
    #   print("Printing cluster counts after " + str(t) + " iters", file=f)
    #   print(n_, file=f)										# Save the cluster counts

  f.close()			# Close the file

  return mu_, z_
      

X_train = dataset()


# Decide on a range of values using cross product

alpha_range = ['1e-2']
K_range = ['3', '1', '5']

value_range = list(itertools.product(alpha_range, K_range))

print("value_range : ", value_range)

for _, (alpha, K) in enumerate(value_range): 
	GibbsSampler(X_train, alpha, K)

