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

def plot_original():
	X_train = np.load("X_train.npy")
	
	fig, ax = plt.subplots()
	c = cm.rainbow(np.linspace(0, 1, 20))
		
	for k in range(20):
		start_index = 800*k
		end_index = 800*(k+1)
		ax.scatter(X_train[start_index:end_index, 0], X_train[start_index:end_index, 1], color=c[k])

	plt.title("Original Plot")
	plt.savefig("original_x_train_plot.jpg")
	plt.close()


def read_and_plot_X_and_Z():
	X_train = np.load("X_train.npy")
	
	for folder in os.listdir("."):
		
		if folder.startswith("results"):
	
			values = folder.split("results_")[1].split("alpha_")[1].split("_K_")
			print("Alpha = {0}, K = {1}".format(values[0], values[1]))		
			
			# For every epoch's saved values, plot the clusters
			for t in range(0, 31, 1):
				z_ = np.load("{1}/z_iter_{0}.npy".format(t, folder))
				n_ = Counter(z_)
				
				fig, ax = plt.subplots()
				c = cm.rainbow(np.linspace(0, 1, len(n_)))
				
				i = 0		
				for k in n_:
					X_train_sub = X_train[np.where(z_ == k)]
					ax.scatter(X_train_sub[:, 0], X_train_sub[:, 1], color=c[i])
					i+=1
				
				plt.title("Alpha = {0}, K = {1}, Iteration #{2}".format(values[0], values[1], t))
				plt.savefig("{1}/z_iter_{0}_PLOT.jpg".format(t, folder))
				plt.close()

read_and_plot_X_and_Z()
plot_original()
