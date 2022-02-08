from __future__ import division
import numpy as np
import pandas as pd

from scipy.linalg import cholesky, cho_solve, cho_factor
from scipy.stats import multivariate_normal

from scipy.spatial import distance

import itertools
import matplotlib.cm as cm


import matplotlib.pyplot as plt
import matplotlib 
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

#Kernel hyperparameters
theta_0 = 1
theta_1 = 512
theta_2 = 0
theta_3 = 2

def kernel(x_n,x_m ,theta_0, theta_1, theta_2, theta_3):
    part1 = (-0.5*theta_1)*distance.cdist(x_n[:,np.newaxis], x_m[:,np.newaxis], 'sqeuclidean')
    part2 = theta_3*np.dot(x_n[:,np.newaxis],np.transpose(x_m[:,np.newaxis]))
    return theta_0* np.exp(part1) + theta_2 + part2


def gp_prior():
    m = 100
    x = np. linspace (-1,1,m) # Test input vector
    mx = np.zeros(m) # Zero mean vector


    K = kernel(x, x, theta_0, theta_1, theta_2, theta_3)#Covariance kernel

    draws_from_prior = 10 # Draw samples from the prior
    f_prior = multivariate_normal (mean=mx ,cov=K , allow_singular =True).rvs(draws_from_prior).T

    colors = iter(cm.rainbow(np.linspace(0, 1, draws_from_prior)))

    plt.plot(x,mx, 'green')
    for a in range(draws_from_prior):
        plt.plot(x, f_prior[:,a] ,color =next(colors) ) # Plot the samples
    plt. fill_between (x ,mx + np.sqrt(np.diag(K)),mx - np.sqrt(np.diag(K)),color='lightgray')
    plt.legend(['Gaussian process mean', 'Prior function']) 
    plt.show ()

def Gaussian_process_algorithm(X_train, Y_train):

    #Prediction data
    X_star = np. linspace (0,1,5000) 
    #Predict data till maximum of given lineage
    X_star = np.array([float(x) for x in X_star if x >= min(X_train)])
    X_star = np.array([float(x) for x in X_star if x <= max(X_train)])

    K = kernel(X_train, X_train, theta_0, theta_1, theta_2, theta_3)#Covariance kernel of train data
    K_star =kernel(X_train, X_star, theta_0, theta_1, theta_2, theta_3)#Covariance kernel of train data and predict data
    K_star_star =kernel(X_star, X_star, theta_0, theta_1, theta_2, theta_3)#Covariance kernel of prediction data
    # Prediction
    sigma_n = 0.01 #Noise
    A = K + sigma_n**2*np.eye(len(X_train))
    # print(K)

    # L = cholesky(A, lower = True)
    c, low = cho_factor(A)
    alpha = cho_solve((c, low ), Y_train)
    # print(alpha)
    f_star = np.dot(K_star.T, alpha)

    c, low = cho_factor(A)
    v = cho_solve((c, low ), K_star)
    cov_star = K_star_star - np.dot(K_star.T, v)

    # print(np.diag(cov_star))

    return f_star, cov_star, X_star



