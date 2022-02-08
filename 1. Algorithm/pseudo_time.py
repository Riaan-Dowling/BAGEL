from sklearn.metrics import pairwise_distances
import os
import numpy as np
import matplotlib

import matplotlib.pyplot as plt

import pandas as pd


import networkx as nx
import time
import random
import copy

from sklearn.metrics import pairwise_distances
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed
from scipy.sparse.linalg import eigs

from scipy.sparse import csr_matrix, find, csgraph
from scipy.stats import entropy, pearsonr, norm
from numpy.linalg import inv
from copy import deepcopy
from sklearn.decomposition import PCA

import joblib


import PCA_combine_datasets
import pseudo_time_calculations
import preprocess
import dimension_reduction

import os


def _clean_up(df):
    # Data frame = Access a group of rows and columns by label(s) or a boolean array.
    # df = Is the values in the data frame that where the rows> 0 or columns > 0(non zero)
    df = df.loc[df.index[df.sum(axis=1) > 0], :]  # rows - >cells
    df = df.loc[:, df.columns[df.sum() > 0]]  # colums  -> genes
    return df


def from_csv(counts_csv_file, delimiter=","):
    # Read in csv file
    try:
        df = pd.read_csv(
            counts_csv_file, sep=delimiter, index_col=[0]
        )  # obtain data frame
        clean_df = _clean_up(df)
    except:
        print(
            "The input data set does not exist. Please ensure that the correct file name is specified (Hint: Confirm spelling)."
        )
        os._exit(1)
    return clean_df


def palantir_pseudo_time(
    early_cell,
    diffusion_components,
    new_manifold_FLAG,
    Main_data_file,
    Secondary_data_file,
):

    if new_manifold_FLAG == True:

        if Main_data_file != Secondary_data_file:
            print("Input: Two datasets.")

            PCA_combine_datasets.combine(Main_data_file, Secondary_data_file)

            norm_df = joblib.load("norm_df.pkl")  # Calculated pseudo_time
            # PCA of all data
            print("Compute manifold.")
            # Dimensionality reduction
            print("PCA (1/4)")
            # Import secondary dataset
            data_file = "Combined_PCA.csv"
            THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(THIS_FOLDER, data_file)
            pca_projections = pd.read_csv(
                data_dir, sep=",", index_col=[0]
            )  # , index_col=[0]) #obtain data frame

            # pca_projections = dimension_reduction.run_pca(combined_PCA)

            two_data_set_FLAG = True
            joblib.dump(two_data_set_FLAG, "two_data_set_FLAG.pkl", compress=3)

        else:
            print("Input: One dataset.")
            # Import MAIN data set
            THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
            Main_data_dir = os.path.join(THIS_FOLDER, Main_data_file)
            MAIN_counts = from_csv(Main_data_dir)

            # Filter data set
            filtered_counts = preprocess.filter_counts_data(
                MAIN_counts, cell_min_molecules=1000, genes_min_cells=10
            )
            norm_df = preprocess.normalize_counts(filtered_counts)
            norm_df = preprocess.log_transform(norm_df)

            joblib.dump(norm_df, "norm_df.pkl", compress=3)

            # PCA of all data

            print("Compute manifold.")
            # Dimensionality reduction
            print("PCA (1/4)")
            pca_projections = dimension_reduction.run_pca(norm_df)

            two_data_set_FLAG = False
            joblib.dump(two_data_set_FLAG, "two_data_set_FLAG.pkl", compress=3)

        val = 1
        while val != 2:

            print("Diffusion map (2/4)")
            dm_res = dimension_reduction.run_diffusion_maps(
                pca_projections, n_components=diffusion_components
            )

            print("Multi scale distance (3/4)")
            ms_data = dimension_reduction.determine_multiscale_space(dm_res)

            print("t-SNE (4/4)")
            # tsne = dimension_reduction.run_tsne(ms_data)

            # Select PCA
            pca = PCA(n_components=2, svd_solver="randomized")
            pca_projections = pca.fit_transform(ms_data)
            pca_projections = pd.DataFrame(pca_projections, index=norm_df.index)
            pca_projections.columns = ["x", "y"]
            tsne = pca_projections

            # TODO magic_imputation
            # print("magic_imputation (5/5)")
            # imp_df = dimension_reduction.run_magic_imputation(norm_df, dm_res)

            if Main_data_file != Secondary_data_file:
                total_Secondary_cells_used = joblib.load(
                    "total_Secondary_cells_used.pkl"
                )
                j_1 = pca_projections.tail(total_Secondary_cells_used)
                plt.scatter(j_1["x"], j_1["y"], s=20, c="k")
            sizes = norm_df.sum(axis=1)  # Define the expressions per cell
            plt.scatter(
                pca_projections["x"],
                pca_projections["y"],
                s=3,
                c=sizes,
                cmap=matplotlib.cm.Spectral_r,
            )
            plt.axis("off")
            plt.title("Phenotypic manifold")
            plt.colorbar()
            plt.show()

            val = input("Was the manifold continuous? (Type 1 for NO and 2 for YES) : ")

            test_pass = False
            while test_pass == False:
                try:
                    test = int(val)
                    if (test == 1) or (test == 2):
                        test_pass = True
                    else:
                        temp = int("a")
                except ValueError:
                    print("Input error!")
                    val = input(
                        "Was the manifold continuous? (Type 1 for NO and 2 for YES) : "
                    )
            val = int(val)

        # ################################################
        # Determine the boundary cell closest to user defined early cell
        data = ms_data
        dm_boundaries = pd.Index(set(data.idxmax()).union(data.idxmin()))
        dists = pairwise_distances(
            data.loc[dm_boundaries, :], data.loc[early_cell, :].values.reshape(1, -1)
        )
        start_cell = pd.Series(np.ravel(dists), index=dm_boundaries).idxmin()

        # Sample waypoints
        waypoints = pseudo_time_calculations._max_min_sampling(data, 500)
        waypoints = waypoints.union(dm_boundaries)

        # No terminal states
        waypoints = pd.Index(waypoints.difference([start_cell]).unique())

        # Append start cell
        waypoints = pd.Index([start_cell]).append(waypoints)

        # Calculate pseudo time
        KNN = 30
        n_jobs = 1
        # :param num_jobs: Number of jobs for parallel processing = -1

        # pseudotime, W = pseudo_time_calculations._compute_pseudotime(imp_df, start_cell, 30, waypoints, n_jobs)
        pseudotime, W = pseudo_time_calculations._compute_pseudotime(
            ms_data, start_cell, KNN, waypoints, n_jobs
        )

        # Markov chain
        wp_data = data.loc[waypoints, :]
        T = _construct_markov_chain(wp_data, KNN, pseudotime, n_jobs)

        # Terminal states
        terminal_states = _terminal_states_from_markov_chain(T, wp_data, pseudotime)

        # Excluded diffusion map boundaries
        dm_boundaries = pd.Index(set(wp_data.idxmax()).union(wp_data.idxmin()))
        excluded_boundaries = dm_boundaries.difference(terminal_states).difference(
            [start_cell]
        )

        # tsen data of
        wp_data_TSNE_ROW = tsne.loc[terminal_states]
        # Dump data to increase performance time
        tsne.to_pickle("sample_tsne.p")

        joblib.dump(pseudotime, "pseudo_time.pkl", compress=3)
        joblib.dump(W, "waypoints.pkl", compress=3)
        joblib.dump(wp_data, "wp_data.pkl", compress=3)
        joblib.dump(T, "T_markov_chain.pkl", compress=3)
        joblib.dump(terminal_states, "terminal_states.pkl", compress=3)
        joblib.dump(excluded_boundaries, "excluded_boundaries.pkl", compress=3)

        sizes = norm_df.sum(axis=1)  # Define the expressions per cell

        """
        PLOT output
        """
        # TODO plot output for debug
        # plt.scatter(tsne["x"], tsne["y"], s=3, c=sizes, cmap=matplotlib.cm.Spectral_r)
        # plt.scatter(wp_data_TSNE_ROW["x"], wp_data_TSNE_ROW["y"], s=20, c="k")
        # plt.axis("off")
        # plt.title("Phenotypic manifold")
        # plt.colorbar()
        # plt.show()

        # Set FLAG to false
        new_manifold_FLAG = False
    else:
        return


def _construct_markov_chain(wp_data, knn, pseudotime, n_jobs):

    # Markov chain construction
    print("Markov chain construction...")
    waypoints = wp_data.index

    # kNN graph
    n_neighbors = knn
    nbrs = NearestNeighbors(
        n_neighbors=n_neighbors, metric="euclidean", n_jobs=n_jobs
    ).fit(wp_data)
    kNN = nbrs.kneighbors_graph(wp_data, mode="distance")
    dist, ind = nbrs.kneighbors(wp_data)

    # Standard deviation allowing for "back" edges
    adpative_k = np.min([int(np.floor(n_neighbors / 3)) - 1, 30])
    adaptive_std = np.ravel(dist[:, adpative_k])

    # Directed graph construction
    # pseudotime position of all the neighbors
    traj_nbrs = pd.DataFrame(
        pseudotime[np.ravel(waypoints.values[ind])].values.reshape(
            [len(waypoints), n_neighbors]
        ),
        index=waypoints,
    )

    # Remove edges that move backwards in pseudotime except for edges that are within
    # the computed standard deviation
    rem_edges = traj_nbrs.apply(
        lambda x: x < pseudotime[traj_nbrs.index] - adaptive_std
    )
    rem_edges = rem_edges.stack()[rem_edges.stack()]

    # Determine the indices and update adjacency matrix
    cell_mapping = pd.Series(range(len(waypoints)), index=waypoints)
    x = list(cell_mapping[rem_edges.index.get_level_values(0)])
    y = list(rem_edges.index.get_level_values(1))
    # Update adjacecy matrix
    kNN[x, ind[x, y]] = 0

    # Affinity matrix and markov chain
    x, y, z = find(kNN)
    aff = np.exp(
        -(z ** 2) / (adaptive_std[x] ** 2) * 0.5
        - (z ** 2) / (adaptive_std[y] ** 2) * 0.5
    )
    W = csr_matrix((aff, (x, y)), [len(waypoints), len(waypoints)])

    # Transition matrix
    D = np.ravel(W.sum(axis=1))
    x, y, z = find(W)
    T = csr_matrix((z / D[x], (x, y)), [len(waypoints), len(waypoints)])

    return T


def _terminal_states_from_markov_chain(T, wp_data, pseudotime):
    print("Identification of terminal states...")

    # Identify terminal statses
    waypoints = wp_data.index
    dm_boundaries = pd.Index(set(wp_data.idxmax()).union(wp_data.idxmin()))
    vals, vecs = eigs(T.T, 10)

    ranks = np.abs(np.real(vecs[:, np.argsort(vals)[-1]]))
    ranks = pd.Series(ranks, index=waypoints)

    # Cutoff and intersection with the boundary cells
    cutoff = norm.ppf(
        0.9999,
        loc=np.median(ranks),
        scale=np.median(np.abs((ranks - np.median(ranks)))),
    )

    # Connected components of cells beyond cutoff
    cells = ranks.index[ranks > cutoff]

    # Find connected components
    T_dense = pd.DataFrame(T.todense(), index=waypoints, columns=waypoints)
    graph = nx.from_pandas_adjacency(T_dense.loc[cells, cells])
    cells = [pseudotime[i].idxmax() for i in nx.connected_components(graph)]

    # Nearest diffusion map boundaries
    terminal_states = [
        pd.Series(
            np.ravel(
                pairwise_distances(
                    wp_data.loc[dm_boundaries, :],
                    wp_data.loc[i, :].values.reshape(1, -1),
                )
            ),
            index=dm_boundaries,
        ).idxmin()
        for i in cells
    ]

    terminal_states = np.unique(terminal_states)

    # excluded_boundaries = dm_boundaries.difference(terminal_states)
    return terminal_states
