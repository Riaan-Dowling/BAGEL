"""
Core functions for running Palantir
"""
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import networkx as nx
import time
import random
import copy

# from sklearn.externals import joblib
import joblib


from sklearn.metrics import pairwise_distances
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed
from scipy.sparse.linalg import eigs

from scipy.sparse import csr_matrix, find, csgraph
from scipy.stats import entropy, pearsonr, norm
from numpy.linalg import inv
from copy import deepcopy
# from presults import PResults

import warnings
warnings.filterwarnings(action="ignore", message="scipy.cluster")
warnings.filterwarnings(action="ignore", module="scipy",
                        message="Changing the sparsity")


def run_palantir(ms_data, early_cell, terminal_states=None,
                 knn=30, num_waypoints=1200, n_jobs=-1,
                 scale_components=True, use_early_cell_as_start=False):
    """Function for max min sampling of waypoints

    :param data: Multiscale space diffusion components
    :param early_cell: Start cell for pseudotime construction
    :param terminal_states: List/Series of user defined terminal states
    :param knn: Number of nearest neighbors for graph construction
    :param num_waypoints: Number of waypoints to sample
    :param num_jobs: Number of jobs for parallel processing
    :param max_iterations: Maximum number of iterations for pseudotime convergence
    :return: PResults object with pseudotime, entropy, branch probabilities and waypoints
    """


def _max_min_sampling(data, num_waypoints):
    """Function for max min sampling of waypoints
    :param data: Data matrix along which to sample the waypoints,
                 usually diffusion components
    :param num_waypoints: Number of waypoints to sample
    :param num_jobs: Number of jobs for parallel processing
    :return: pandas Series reprenting the sampled waypoints
    """

    waypoint_set = list()
    no_iterations = int((num_waypoints) / data.shape[1])

    # Sample along each component
    N = data.shape[0]
    for ind in data.columns:
        # Data vector
        vec = np.ravel(data[ind])

        # Random initialzlation
        iter_set = random.sample(range(N), 1)

        # Distances along the component
        dists = np.zeros([N, no_iterations])
        dists[:, 0] = abs(vec - data[ind].values[iter_set])
        for k in range(1, no_iterations):
            # Minimum distances across the current set
            min_dists = dists[:, 0:k].min(axis=1)

            # Point with the maximum of the minimum distances is the new waypoint
            new_wp = np.where(min_dists == min_dists.max())[0][0]
            iter_set.append(new_wp)

            # Update distances
            dists[:, k] = abs(vec - data[ind].values[new_wp])

        # Update global set
        waypoint_set = waypoint_set + iter_set

    # Unique waypoints
    waypoints = data.index[waypoint_set].unique()

    return waypoints


def _compute_pseudotime(data, start_cell, knn, waypoints, n_jobs, max_iterations=25):
    """Function for compute the pseudotime
    :param data: Multiscale space diffusion components
    :param start_cell: Start cell for pseudotime construction
    :param knn: Number of nearest neighbors for graph construction
    :param waypoints: List of waypoints
    :param n_jobs: Number of jobs for parallel processing
    :param max_iterations: Maximum number of iterations for pseudotime convergence
    :return: pseudotime and weight matrix
    """

    # ################################################
    # Shortest path distances to determine trajectories
    print("Shortest path distances using {}-nearest neighbor graph...".format(knn))
    start = time.time()
    nbrs = NearestNeighbors(n_neighbors=knn, metric="euclidean", n_jobs=n_jobs).fit(
        data
    )
    adj = nbrs.kneighbors_graph(data, mode="distance")

    # Connect graph if it is disconnected
    adj = _connect_graph(adj, data, np.where(data.index == start_cell)[0][0])

    # Distances
    dists = Parallel(n_jobs=n_jobs, max_nbytes=None)(
        delayed(_shortest_path_helper)(np.where(data.index == cell)[0][0], adj)
        for cell in waypoints
    )

    # Convert to distance matrix
    D = pd.DataFrame(0.0, index=waypoints, columns=data.index)
    for i, cell in enumerate(waypoints):
        D.loc[cell, :] = pd.Series(
            np.ravel(dists[i]), index=data.index[dists[i].index]
        )[data.index]
    end = time.time()
    print("Time for shortest paths: {} minutes".format((end - start) / 60))

    # ###############################################
    # Determine the perspective matrix

    print("Iteratively refining the pseudotime...")
    # Waypoint weights
    sdv = np.std(np.ravel(D)) * 1.06 * len(np.ravel(D)) ** (-1 / 5)
    W = np.exp(-0.5 * np.power((D / sdv), 2))
    # Stochastize the matrix
    W = W / W.sum()

    # Initalize pseudotime to start cell distances
    pseudotime = D.loc[start_cell, :]
    converged = False

    # Iteratively update perspective and determine pseudotime
    iteration = 1
    while not converged and iteration < max_iterations:
        # Perspective matrix by alinging to start distances
        P = deepcopy(D)
        for wp in waypoints[1:]:
            # Position of waypoints relative to start
            idx_val = pseudotime[wp]

            # Convert all cells before starting point to the negative
            before_indices = pseudotime.index[pseudotime < idx_val]
            P.loc[wp, before_indices] = -D.loc[wp, before_indices]

            # Align to start
            P.loc[wp, :] = P.loc[wp, :] + idx_val

        # Weighted pseudotime
        new_traj = P.multiply(W).sum()

        # Check for convergence
        corr = pearsonr(pseudotime, new_traj)[0]
        # print("Correlation at iteration %d: %.4f" % (iteration, corr))
        if corr > 0.9999:
            converged = True

        # If not converged, continue iteration
        pseudotime = new_traj
        iteration += 1

    return pseudotime, W


# def _max_min_sampling(data, num_waypoints):
#     """Function for max min sampling of waypoints

#     :param data: Data matrix along which to sample the waypoints,
#                  usually diffusion components
#     :param num_waypoints: Number of waypoints to sample
#     :param num_jobs: Number of jobs for parallel processing
#     :return: pandas Series reprenting the sampled waypoints
#     """
#     print("waypoints")
#     waypoint_set = list()
#     no_iterations = int((num_waypoints) / data.shape[1])
#     print("Data.shape[1] " + str(data.shape[1]))
#     print("no it"+ str(no_iterations))
#     # Sample along each component
#     N = data.shape[0]
#     print("Data.shape[0] " + str(data.shape[0]))
#     print("colom" + str(data.columns))
#     for ind in data.columns:
#         # Data vector
#         vec = np.ravel(data[ind])
#         # print("vec" + str(vec))
#         # Random initialzlation
#         iter_set = random.sample(range(N), 1)

#         # Distances along the component
#         dists = np.zeros([N, no_iterations])
#         dists[:, 0] = abs(vec - data[ind].values[iter_set])
#         print(" data[ind].values[iter_set]" + str ( iter_set))
#         print(dists[:, 0])
    #     for k in range(1, no_iterations):
    #         # Minimum distances across the current set
    #         min_dists = dists[:, 0:k].min(axis=1)

    #         # Point with the maximum of the minimum distances is the new waypoint
    #         new_wp = np.where(min_dists == min_dists.max())[0][0]
    #         iter_set.append(new_wp)

    #         # Update distances
    #         dists[:, k] = abs(vec - data[ind].values[new_wp])

    #     # Update global set
    #     waypoint_set = waypoint_set + iter_set

    # # Unique waypoints
    # waypoints = data.index[waypoint_set].unique()

    # return waypoints


# def _compute_pseudotime(data, start_cell, knn,
#                         waypoints, n_jobs, max_iterations=25):
#     """Function for compute the pseudotime

#     :param data: Multiscale space diffusion components
#     :param start_cell: Start cell for pseudotime construction
#     :param knn: Number of nearest neighbors for graph construction
#     :param waypoints: List of waypoints
#     :param n_jobs: Number of jobs for parallel processing
#     :param max_iterations: Maximum number of iterations for pseudotime convergence
#     :return: pseudotime and weight matrix
#     """

#     # ################################################
#     # Shortest path distances to determine trajectories
#     print('Shortest path distances using {}-nearest neighbor graph...'.format(knn))
#     start = time.time()
#     nbrs = NearestNeighbors(n_neighbors=knn,
#                             metric='euclidean', n_jobs=n_jobs).fit(data)
#     adj = nbrs.kneighbors_graph(data, mode='distance')

#     # Connect graph if it is disconnected
#     adj = _connect_graph(adj, data, np.where(data.index == start_cell)[0][0])

#     # Distances
#     dists = Parallel(n_jobs=n_jobs)(
#         delayed(_shortest_path_helper)(np.where(data.index == cell)[0][0], adj)
#         for cell in waypoints)

#     # print("dists:   " + str(dists))

#     # Convert to distance matrix
#     D = pd.DataFrame(0.0, index=waypoints, columns=data.index)
#     for i, cell in enumerate(waypoints):
#         D.loc[cell, :] = pd.Series(np.ravel(dists[i]),
#                                    index=data.index[dists[i].index])[data.index]
#     end = time.time()
#     print('Time for shortest paths: {} minutes'.format((end - start) / 60))

#     # ###############################################
#     # Determine the perspective matrix

#     print('Iteratively refining the pseudotime...')
#     # Waypoint weights
#     sdv = np.std(np.ravel(D)) * 1.06 * len(np.ravel(D)) ** (-1 / 5)
#     W = np.exp(-.5 * np.power((D / sdv), 2))
#     # Stochastize the matrix
#     W = W / W.sum()

#     # Initalize pseudotime to start cell distances
#     pseudotime = D.loc[start_cell, :]
#     # print("pseudo time start: " + str(pseudotime))
#     converged = False

#     # Iteratively update perspective and determine pseudotime
#     iteration = 1
#     while not converged and iteration < max_iterations:
#         # Perspective matrix by alinging to start distances
#         P = deepcopy(D)
#         for wp in waypoints[1:]:
#             # Position of waypoints relative to start
#             idx_val = pseudotime[wp]

#             # Convert all cells before starting point to the negative
#             before_indices = pseudotime.index[pseudotime < idx_val]
#             P.loc[wp, before_indices] = -D.loc[wp, before_indices]

#             # Align to start
#             P.loc[wp, :] = P.loc[wp, :] + idx_val

#         # Weighted pseudotime
#         new_traj = P.multiply(W).sum()

#         # Check for convergence
#         corr = pearsonr(pseudotime, new_traj)[0]
#         # print(corr)
#         print('Correlation at iteration %d: %.4f' % (iteration, corr))
#         if corr > 0.9999:
#             converged = True

#         # If not converged, continue iteration
#         pseudotime = new_traj
#         iteration += 1

#     return pseudotime, W



def _shortest_path_helper(cell, adj):
    return pd.Series(csgraph.dijkstra(adj, False, cell))


def _connect_graph(adj, data, start_cell):

    # Create graph and compute distances
    graph = nx.Graph(adj)
    dists = pd.Series(nx.single_source_dijkstra_path_length(graph, start_cell))
    dists = pd.Series(dists.values, index=data.index[dists.index])

    # Idenfity unreachable nodes
    unreachable_nodes = data.index.difference(dists.index)
    if len(unreachable_nodes) > 0:
        print('Warning: Some of the cells were unreachable. Consider increasing the k for \n \
            nearest neighbor graph construction.')

    # Connect unreachable nodes
    while len(unreachable_nodes) > 0:
        farthest_reachable = np.where(data.index == dists.idxmax())[0][0]

        # Compute distances to unreachable nodes
        unreachable_dists = pairwise_distances(data.iloc[farthest_reachable, :].values.reshape(1, -1),
                                               data.loc[unreachable_nodes, :])
        unreachable_dists = pd.Series(
            np.ravel(unreachable_dists), index=unreachable_nodes)

        # Add edge between farthest reacheable and its nearest unreachable
        add_edge = np.where(data.index == unreachable_dists.idxmin())[0][0]
        adj[farthest_reachable, add_edge] = unreachable_dists.min()

        # Recompute distances to early cell
        graph = nx.Graph(adj)
        dists = pd.Series(
            nx.single_source_dijkstra_path_length(graph, start_cell))
        dists = pd.Series(dists.values, index=data.index[dists.index])

        # Idenfity unreachable nodes
        unreachable_nodes = data.index.difference(dists.index)

    return adj

