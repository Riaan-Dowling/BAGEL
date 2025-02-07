"""
This scirpt contains the phenotypic manifold
"""

import time
import random
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs
from copy import deepcopy
import matplotlib
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix, find, issparse, linalg, csgraph
from scipy.stats import pearsonr, norm
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

from tqdm import tqdm


class PhenotypicManifold(object):
    """
    Manifold development
    """

    def __init__(
        self,
        total_datasets,
        main_df,
        secondary_df,
        result_folder,
        early_cell,
        bagel_config,
        load_old_manifold,
    ) -> None:
        self.total_datasets = total_datasets
        self.main_df = main_df
        self.secondary_df = secondary_df
        self.result_folder = result_folder
        self.early_cell = early_cell
        self.bagel_config = bagel_config
        self.load_old_manifold = load_old_manifold

        self.diffusion_components = self.bagel_config["phenotypic_manifold_config"][
            "palantir"
        ]["diffusion_components"]
        self.total_secondary_cells_used = 0

    def pca_combine_two_datasets(
        self,
        n_components_pca,
        _cell_min_molecules,
        _genes_min_cells,
    ):
        """
        -----------------------------------------------------------------
        combine two single cell datasets
        -----------------------------------------------------------------
        """

        print("Combining data sets.")
        print("(1/4) - Filter datasets.")

        # """
        # -----------------------------------------------------------------
        # Select gene columns in common and missing genes
        # -----------------------------------------------------------------
        # """

        genes_in_common = self.secondary_df.loc[
            :, self.secondary_df.columns.isin(self.main_df.columns.tolist())
        ].copy()
        missing_genes = self.main_df.loc[
            :, ~self.main_df.columns.isin(genes_in_common.columns.tolist())
        ].copy()

        missing_genes_columns = pd.DataFrame(columns=missing_genes.columns.tolist())

        # Persentage gene overlapp
        persentage = (len(genes_in_common.columns) / len(self.main_df.columns)) * 100
        print("Gene percentage overlapp: " + str(persentage) + "%")
        # Append missing gene columns
        secondary_df_with_missing_data = pd.concat(
            [genes_in_common, missing_genes_columns], axis=1, sort=False
        ).fillna(0)

        # """
        # -----------------------------------------------------------------
        # Scale secondary dataframe to main dataframe
        # -----------------------------------------------------------------
        # """

        # Scale secondary df to fit main df
        # Scale secondary df to fit main df
        # 1) Row count sum
        secondary_df_cell_counts_sum = secondary_df_with_missing_data.sum(axis=1)
        main_df_cell_count_sum = self.main_df.sum(axis=1)

        # 2) Scale secondary counts to fit max min of main counts
        scaler_max_min = MinMaxScaler(
            feature_range=(main_df_cell_count_sum.min(), main_df_cell_count_sum.max())
        )
        max_min_secondary_df_cell_counts_sum = scaler_max_min.fit_transform(
            secondary_df_cell_counts_sum.values.reshape(-1, 1)
        )

        # 3) Scale
        secondary_df_with_missing_data_norm = (
            secondary_df_with_missing_data.div(secondary_df_cell_counts_sum, axis=0)
            .fillna(0)
            .mul(max_min_secondary_df_cell_counts_sum, axis=0)
        )

        # NOTE old bug:
        # secondary_df_with_missing_data_norm = secondary_df_with_missing_data.fillna(0)

        # Rerange data in correct column order and MERGE with dataset 1
        secondary_df_with_missing_data_ordered = secondary_df_with_missing_data_norm[
            self.main_df.columns
        ].copy()

        # """
        # -----------------------------------------------------------------
        # Combince dataframes one cell at a time
        # -----------------------------------------------------------------
        # """
        projected_secondary_pca_df = pd.DataFrame()

        # Set cell ID number to each cell to enable singel cell selection
        total_secondary_cells = len(secondary_df_with_missing_data_ordered.index)
        secondary_cell_id_number = range(total_secondary_cells)
        secondary_df_with_missing_data_ordered.loc[:, "secondary_cell_id_number"] = (
            secondary_cell_id_number
        )

        total_secondary_cells_used = 0
        for _ in tqdm(
            range(len(secondary_df_with_missing_data_ordered.index)),
            desc="PCA data set combining.",
        ):
            # Select one cell
            select_one = secondary_df_with_missing_data_ordered.head(1)
            # Remove selected cell from data
            secondary_df_with_missing_data_ordered = (
                secondary_df_with_missing_data_ordered[
                    ~secondary_df_with_missing_data_ordered[
                        "secondary_cell_id_number"
                    ].isin(select_one["secondary_cell_id_number"].values)
                ]
            ).copy()

            # Delete cell ID column
            del select_one["secondary_cell_id_number"]

            # Merge 1 cell data to Dataset 1
            data = pd.concat([self.main_df.copy(), select_one], axis=0, sort=False)

            # Preform Palantir data pre-processing
            filtered_counts = self.palantir_filter_counts_data(
                data,
                cell_min_molecules=_cell_min_molecules,
                genes_min_cells=_genes_min_cells,
            )
            norm_df = self.palantir_normalize_counts(filtered_counts)
            log_norm_df = self.palantir_log_transform(norm_df)

            # Select PCA
            pca_projections = self.run_pca(log_norm_df, n_components=n_components_pca)
            pca_projections = pd.DataFrame(pca_projections, index=log_norm_df.index)

            computed_pca_1 = pca_projections.tail(1)

            # Only append if pca projection is the same cell as the selected one
            if computed_pca_1.index == select_one.index:
                # Count how many cells are used
                total_secondary_cells_used = total_secondary_cells_used + 1
                # Append one cell data
                if projected_secondary_pca_df.empty:
                    projected_secondary_pca_df.reset_index(drop=True, inplace=True)
                    projected_secondary_pca_df = computed_pca_1

                else:
                    projected_secondary_pca_df = pd.concat(
                        [projected_secondary_pca_df, computed_pca_1], axis=0, sort=False
                    )  # Append new lineage data

        joblib.dump(
            total_secondary_cells_used,
            f"{self.result_folder}/total_secondary_cells_used.pkl",
            compress=3,
        )

        # Palantir pre-process data MAIN DF and secondary df
        filtered_counts_main = self.palantir_filter_counts_data(
            self.main_df,
            cell_min_molecules=_cell_min_molecules,
            genes_min_cells=_genes_min_cells,
        )
        norm_df_main = self.palantir_normalize_counts(filtered_counts_main)
        log_norm_main_df = self.palantir_log_transform(norm_df_main)
        log_norm_secondary_df = self.palantir_log_transform(
            secondary_df_with_missing_data_norm
        )

        log_norm_bagel_df = pd.concat(
            [log_norm_main_df, log_norm_secondary_df], axis=0
        ).fillna(0)
        # Select PCA
        print("(2/4) - PCA")
        pca_projections_main = self.run_pca(
            log_norm_main_df, n_components=n_components_pca
        )
        pca_projections_main_df = pd.DataFrame(
            pca_projections_main, index=norm_df_main.index
        )

        # obtain indexes
        pca_projections_main_df_index = pd.DataFrame(
            {"index_1": pca_projections_main_df.index}
        )
        projected_secondary_pca_df_index = pd.DataFrame(
            {"index_1": projected_secondary_pca_df.index}
        )

        combined_pca_index_df = pd.concat(
            [pca_projections_main_df_index, projected_secondary_pca_df_index],
            axis=0,
            sort=False,
        )  # Append new lineage data

        pca_projections_main_df.reset_index(drop=True, inplace=True)
        projected_secondary_pca_df.reset_index(drop=True, inplace=True)

        combined_main_and_secondary_pca = pd.concat(
            [pca_projections_main_df, projected_secondary_pca_df], axis=0, sort=False
        )  # Append new lineage data

        combined_main_and_secondary_pca.reset_index(drop=True, inplace=True)

        # Export combined data
        export_combined_main_and_secondary_pca = pd.DataFrame(
            combined_main_and_secondary_pca.values,
            index=combined_pca_index_df.loc[:, "index_1"],
        )
        export_combined_main_and_secondary_pca.to_csv(
            f"{self.result_folder}/export_combined_main_and_secondary_pca.csv"
        )

        print("Combining data sets end.")
        return export_combined_main_and_secondary_pca, log_norm_bagel_df

    def palantir_filter_counts_data(
        self, data, cell_min_molecules=1000, genes_min_cells=10
    ):
        """Remove low molecule count cells and low detection genes

        :param data: Counts matrix: Cells x Genes
        :param cell_min_molecules: Minimum number of molecules per cell
        :param genes_min_cells: Minimum number of cells in which a gene is detected
        :return: Filtered counts matrix

        @article{Palantir_2019,
            title = {Characterization of cell fate probabilities in single-cell data with Palantir},
            author = {Manu Setty and Vaidotas Kiseliovas and Jacob Levine and Adam Gayoso and Linas Mazutis and Dana Pe'er},
            journal = {Nature Biotechnology},
            year = {2019},
            month = {march},
            url = {https://doi.org/10.1038/s41587-019-0068-4},
            doi = {10.1038/s41587-019-0068-4}
        }
        """

        # Molecule and cell counts
        ms = data.sum(axis=1)
        cs = data.sum()

        # Filter
        return data.loc[
            ms.index[ms > cell_min_molecules], cs.index[cs > genes_min_cells]
        ]

    def palantir_normalize_counts(self, data):
        """Correct the counts for molecule count variability

        :param data: Counts matrix: Cells x Genes
        :return: Normalized matrix

        @article{Palantir_2019,
            title = {Characterization of cell fate probabilities in single-cell data with Palantir},
            author = {Manu Setty and Vaidotas Kiseliovas and Jacob Levine and Adam Gayoso and Linas Mazutis and Dana Pe'er},
            journal = {Nature Biotechnology},
            year = {2019},
            month = {march},
            url = {https://doi.org/10.1038/s41587-019-0068-4},
            doi = {10.1038/s41587-019-0068-4}
        }
        """
        ms = data.sum(axis=1)
        norm_df = data.div(ms, axis=0).mul(np.median(ms), axis=0)
        return norm_df

    def palantir_log_transform(self, data, pseudo_count=0.1):
        """Log transform the matrix

        :param data: Counts matrix: Cells x Genes
        :return: Log transformed matrix

        @article{Palantir_2019,
            title = {Characterization of cell fate probabilities in single-cell data with Palantir},
            author = {Manu Setty and Vaidotas Kiseliovas and Jacob Levine and Adam Gayoso and Linas Mazutis and Dana Pe'er},
            journal = {Nature Biotechnology},
            year = {2019},
            month = {march},
            url = {https://doi.org/10.1038/s41587-019-0068-4},
            doi = {10.1038/s41587-019-0068-4}
        }
        """
        return np.log2(data + pseudo_count)

    def run_pca(self, data, n_components=300):
        """Run PCA

        :param data: Dataframe of cells X genes. Typicaly multiscale space diffusion components
        :param n_components: Number of principal components
        :return: PCA projections of the data and the explained variance
        """
        pca = PCA(n_components=n_components, svd_solver="randomized")
        data = data.apply(lambda x: pd.to_numeric(x))
        pca_projections = pca.fit_transform(data)
        pca_projections = pd.DataFrame(pca_projections, index=data.index)
        return pca_projections  # , pca.explained_variance_ratio_

    def palantir_run_diffusion_maps(self, data_df, n_components, knn=30, n_jobs=-1):
        """Run Diffusion maps using the adaptive anisotropic kernel

        :param data_df: PCA projections of the data or adjancency matrix
        :param n_components: Number of diffusion components
        :return: Diffusion components, corresponding eigen values and the diffusion operator

        @article{Palantir_2019,
            title = {Characterization of cell fate probabilities in single-cell data with Palantir},
            author = {Manu Setty and Vaidotas Kiseliovas and Jacob Levine and Adam Gayoso and Linas Mazutis and Dana Pe'er},
            journal = {Nature Biotechnology},
            year = {2019},
            month = {march},
            url = {https://doi.org/10.1038/s41587-019-0068-4},
            doi = {10.1038/s41587-019-0068-4}
        }
        """

        # Determine the kernel
        N = data_df.shape[0]
        if not issparse(
            data_df
        ):  # Sparse matrix is a matrix which contains very few non-zero elements
            # print('Determing nearest neighbor graph...')
            nbrs = NearestNeighbors(
                n_neighbors=int(knn), metric="euclidean", n_jobs=n_jobs
            ).fit(
                data_df.values
            )  # Get training data from nearest neigbhours
            kNN = nbrs.kneighbors_graph(
                data_df.values, mode="distance"
            )  # Return distances accourding to neighbours
            # Returns a sparse matrix in CSR format
            # -sparse matrix or sparse array: is a matrix in which most of the elements are zero
            # -CSR: compressed sparse row (CSR)
            #    : The CSR format stores a sparse m × n matrix M in row form using three (one-dimensional) arrays (V, COL_INDEX, ROW_INDEX). Let NNZ denote the number of nonzero entries in M.

            # Adaptive k
            adaptive_k = int(np.floor(knn / 3))
            # Floor = largest integer
            # KNN; K = 3ka (Ka = k-addaptive)
            nbrs = NearestNeighbors(
                n_neighbors=int(adaptive_k), metric="euclidean", n_jobs=n_jobs
            ).fit(data_df.values)
            adaptive_std = nbrs.kneighbors_graph(data_df.values, mode="distance").max(
                axis=1
            )  # Maximum distance = distance to ka neighbour
            # returns maximum value of each row
            adaptive_std = np.ravel(adaptive_std.todense())
            # todense returns a matrix
            # ravel converts [[]...[]] to [row one matrix 1,row two matrix 1,,,,row final matrix 2]

            # Kernel
            x, y, dists = find(kNN)
            # Return the indices and values of the nonzero elements of a matrix

            # X, y specific stds
            dists = dists / adaptive_std[x]
            # Adaptive_std = scalling factor (sigma)

            W = csr_matrix(
                (np.exp(-dists), (x, y)), shape=[N, N]
            )  # CSR increase computational time

            # Diffusion map
            # Symmetric affinity matrix as
            kernel = W + W.T
        else:
            # Assume data is already sparse kernel matrix
            kernel = data_df

        # Markov
        D = np.ravel(kernel.sum(axis=1))
        D[D != 0] = 1 / D[D != 0]
        # Affinity to markov transition matrix
        T = csr_matrix((D, (range(N), range(N))), shape=[N, N]).dot(kernel)
        # create affinity matrix
        # Eigen value decomposition
        D, V = linalg.eigs(T, n_components, tol=1e-4, maxiter=1000)
        D = np.real(D)
        # Must be real
        V = np.real(V)
        inds = np.argsort(D)[::-1]
        # Argsort
        # >>> x = np.array([3, 1, 2])
        # >>> np.argsort(x)
        # array([1, 2, 0])

        # Returns the indices that would sort an array.

        # Negative values also work to make a copy of the same list in reverse order:
        # L[::-1]
        # [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        D = D[inds]
        V = V[:, inds]

        # Normalize
        for i in range(V.shape[1]):
            V[:, i] = V[:, i] / np.linalg.norm(V[:, i])

        # Create a results dictionary
        res = {"T": T, "EigenVectors": V, "EigenValues": D}
        res["EigenVectors"] = pd.DataFrame(res["EigenVectors"])
        if not issparse(data_df):
            res["EigenVectors"].index = data_df.index
        res["EigenValues"] = pd.Series(res["EigenValues"])

        return res

    def palantir_determine_multiscale_space(self, dm_res, n_eigs=None):
        """Determine multi scale space of the data

        :param dm_res: Diffusion map results from run_diffusion_maps
        :param n_eigs: Number of eigen vectors to use. If None specified, the number
                of eigen vectors will be determined using eigen gap
        :return: Multi scale data matrix

        @article{Palantir_2019,
            title = {Characterization of cell fate probabilities in single-cell data with Palantir},
            author = {Manu Setty and Vaidotas Kiseliovas and Jacob Levine and Adam Gayoso and Linas Mazutis and Dana Pe'er},
            journal = {Nature Biotechnology},
            year = {2019},
            month = {march},
            url = {https://doi.org/10.1038/s41587-019-0068-4},
            doi = {10.1038/s41587-019-0068-4}
        }
        """
        if n_eigs is None:
            vals = np.ravel(dm_res["EigenValues"])
            n_eigs = np.argsort(vals[: (len(vals) - 1)] - vals[1:])[-1] + 1
            if n_eigs < 3:
                n_eigs = np.argsort(vals[: (len(vals) - 1)] - vals[1:])[-2] + 1

        # Scale the data
        use_eigs = list(range(1, n_eigs))
        eig_vals = np.ravel(dm_res["EigenValues"][use_eigs])
        data = dm_res["EigenVectors"].values[:, use_eigs] * (eig_vals / (1 - eig_vals))
        data = pd.DataFrame(data, index=dm_res["EigenVectors"].index)

        return data

    def palantir_max_min_sampling(self, data, num_waypoints):
        """Function for max min sampling of waypoints
        :param data: Data matrix along which to sample the waypoints,
                    usually diffusion components
        :param num_waypoints: Number of waypoints to sample
        :param num_jobs: Number of jobs for parallel processing
        :return: pandas Series reprenting the sampled waypoints

        @article{Palantir_2019,
            title = {Characterization of cell fate probabilities in single-cell data with Palantir},
            author = {Manu Setty and Vaidotas Kiseliovas and Jacob Levine and Adam Gayoso and Linas Mazutis and Dana Pe'er},
            journal = {Nature Biotechnology},
            year = {2019},
            month = {march},
            url = {https://doi.org/10.1038/s41587-019-0068-4},
            doi = {10.1038/s41587-019-0068-4}
        }
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

    def palantir_connect_graph(self, adj, data, start_cell):
        """
        @article{Palantir_2019,
            title = {Characterization of cell fate probabilities in single-cell data with Palantir},
            author = {Manu Setty and Vaidotas Kiseliovas and Jacob Levine and Adam Gayoso and Linas Mazutis and Dana Pe'er},
            journal = {Nature Biotechnology},
            year = {2019},
            month = {march},
            url = {https://doi.org/10.1038/s41587-019-0068-4},
            doi = {10.1038/s41587-019-0068-4}
        }
        """

        # Create graph and compute distances
        graph = nx.Graph(adj)
        dists = pd.Series(nx.single_source_dijkstra_path_length(graph, start_cell))
        dists = pd.Series(dists.values, index=data.index[dists.index])

        # Idenfity unreachable nodes
        unreachable_nodes = data.index.difference(dists.index)
        if len(unreachable_nodes) > 0:
            print(
                "Warning: Some of the cells were unreachable. Consider increasing the k for \n \
                nearest neighbor graph construction."
            )

        # Connect unreachable nodes
        while len(unreachable_nodes) > 0:
            farthest_reachable = np.where(data.index == dists.idxmax())[0][0]

            # Compute distances to unreachable nodes
            unreachable_dists = pairwise_distances(
                data.iloc[farthest_reachable, :].values.reshape(1, -1),
                data.loc[unreachable_nodes, :],
            )
            unreachable_dists = pd.Series(
                np.ravel(unreachable_dists), index=unreachable_nodes
            )

            # Add edge between farthest reacheable and its nearest unreachable
            add_edge = np.where(data.index == unreachable_dists.idxmin())[0][0]
            adj[farthest_reachable, add_edge] = unreachable_dists.min()

            # Recompute distances to early cell
            graph = nx.Graph(adj)
            dists = pd.Series(nx.single_source_dijkstra_path_length(graph, start_cell))
            dists = pd.Series(dists.values, index=data.index[dists.index])

            # Idenfity unreachable nodes
            unreachable_nodes = data.index.difference(dists.index)

        return adj

    def palantir_shortest_path_helper(self, cell, adj):
        """
        @article{Palantir_2019,
            title = {Characterization of cell fate probabilities in single-cell data with Palantir},
            author = {Manu Setty and Vaidotas Kiseliovas and Jacob Levine and Adam Gayoso and Linas Mazutis and Dana Pe'er},
            journal = {Nature Biotechnology},
            year = {2019},
            month = {march},
            url = {https://doi.org/10.1038/s41587-019-0068-4},
            doi = {10.1038/s41587-019-0068-4}
        }
        """
        return pd.Series(csgraph.dijkstra(adj, False, cell))

    def palantir_clean_up_cell_data_files(self, df_in):
        """
        palantir_clean_up_cell_data_files

        @article{Palantir_2019,
            title = {Characterization of cell fate probabilities in single-cell data with Palantir},
            author = {Manu Setty and Vaidotas Kiseliovas and Jacob Levine and Adam Gayoso and Linas Mazutis and Dana Pe'er},
            journal = {Nature Biotechnology},
            year = {2019},
            month = {march},
            url = {https://doi.org/10.1038/s41587-019-0068-4},
            doi = {10.1038/s41587-019-0068-4}
        }
        """
        # Data frame = Access a group of rows and columns by label(s) or a boolean array.
        # df = Is the values in the data frame that where the rows> 0 or columns > 0(non zero)
        df_in = df_in.loc[df_in.index[df_in.sum(axis=1) > 0], :]  # rows - >cells
        df_in = df_in.loc[:, df_in.columns[df_in.sum() > 0]]  # colums  -> genes
        return df_in

    def palantir_construct_markov_chain(self, wp_data, knn, pseudo_time, n_jobs):
        """
        @article{Palantir_2019,
            title = {Characterization of cell fate probabilities in single-cell data with Palantir},
            author = {Manu Setty and Vaidotas Kiseliovas and Jacob Levine and Adam Gayoso and Linas Mazutis and Dana Pe'er},
            journal = {Nature Biotechnology},
            year = {2019},
            month = {march},
            url = {https://doi.org/10.1038/s41587-019-0068-4},
            doi = {10.1038/s41587-019-0068-4}
        }
        """
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
        # pseudo_time position of all the neighbors
        traj_nbrs = pd.DataFrame(
            pseudo_time[np.ravel(waypoints.values[ind])].values.reshape(
                [len(waypoints), n_neighbors]
            ),
            index=waypoints,
        )

        # Remove edges that move backwards in pseudo_time except for edges that are within
        # the computed standard deviation
        rem_edges = traj_nbrs.apply(
            lambda x: x < pseudo_time[traj_nbrs.index] - adaptive_std
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
            -(z**2) / (adaptive_std[x] ** 2) * 0.5
            - (z**2) / (adaptive_std[y] ** 2) * 0.5
        )
        W = csr_matrix((aff, (x, y)), [len(waypoints), len(waypoints)])

        # Transition matrix
        D = np.ravel(W.sum(axis=1))
        x, y, z = find(W)
        T = csr_matrix((z / D[x], (x, y)), [len(waypoints), len(waypoints)])

        return T

    def palantir_compute_pseudo_time(
        self, data, start_cell, knn, waypoints, n_jobs, max_iterations=25
    ):
        """Function for compute the pseudo_time
        :param data: Multiscale space diffusion components
        :param start_cell: Start cell for pseudo_time construction
        :param knn: Number of nearest neighbors for graph construction
        :param waypoints: List of waypoints
        :param n_jobs: Number of jobs for parallel processing
        :param max_iterations: Maximum number of iterations for pseudo_time convergence
        :return: pseudo_time and weight matrix

        @article{Palantir_2019,
            title = {Characterization of cell fate probabilities in single-cell data with Palantir},
            author = {Manu Setty and Vaidotas Kiseliovas and Jacob Levine and Adam Gayoso and Linas Mazutis and Dana Pe'er},
            journal = {Nature Biotechnology},
            year = {2019},
            month = {march},
            url = {https://doi.org/10.1038/s41587-019-0068-4},
            doi = {10.1038/s41587-019-0068-4}
        }
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
        adj = self.palantir_connect_graph(
            adj, data, np.where(data.index == start_cell)[0][0]
        )

        # Distances
        dists = joblib.Parallel(n_jobs=n_jobs, max_nbytes=None)(
            joblib.delayed(self.palantir_shortest_path_helper)(
                np.where(data.index == cell)[0][0], adj
            )
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

        print("Iteratively refining the pseudo_time...")
        # Waypoint weights
        sdv = np.std(np.ravel(D)) * 1.06 * len(np.ravel(D)) ** (-1 / 5)
        W = np.exp(-0.5 * np.power((D / sdv), 2))
        # Stochastize the matrix
        W = W / W.sum()

        # Initalize pseudo_time to start cell distances
        pseudo_time = D.loc[start_cell, :]
        converged = False

        # Iteratively update perspective and determine pseudo_time
        iteration = 1
        while not converged and iteration < max_iterations:
            # Perspective matrix by alinging to start distances
            P = deepcopy(D)
            for wp in waypoints[1:]:
                # Position of waypoints relative to start
                idx_val = pseudo_time[wp]

                # Convert all cells before starting point to the negative
                before_indices = pseudo_time.index[pseudo_time < idx_val]
                P.loc[wp, before_indices] = -D.loc[wp, before_indices]

                # Align to start
                P.loc[wp, :] = P.loc[wp, :] + idx_val

            # Weighted pseudo_time
            new_traj = P.multiply(W).sum()

            # Check for convergence
            corr = pearsonr(pseudo_time, new_traj)[0]
            # print("Correlation at iteration %d: %.4f" % (iteration, corr))
            if corr > 0.9999:
                converged = True

            # If not converged, continue iteration
            pseudo_time = new_traj
            iteration += 1

        return pseudo_time, W

    def palantir_terminal_states_from_markov_chain(self, T, wp_data, pseudo_time):
        """
        @article{Palantir_2019,
            title = {Characterization of cell fate probabilities in single-cell data with Palantir},
            author = {Manu Setty and Vaidotas Kiseliovas and Jacob Levine and Adam Gayoso and Linas Mazutis and Dana Pe'er},
            journal = {Nature Biotechnology},
            year = {2019},
            month = {march},
            url = {https://doi.org/10.1038/s41587-019-0068-4},
            doi = {10.1038/s41587-019-0068-4}
        }
        """
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
        cells = [pseudo_time[list(i)].idxmax() for i in nx.connected_components(graph)]
        for i in nx.connected_components(graph):
            print(i)

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

    def create_phenotypic_manifold(self):
        """
        create_phenotypic_manifold
        """
        if self.load_old_manifold is False:
            _cell_min_molecules = self.bagel_config["phenotypic_manifold_config"][
                "palantir"
            ]["_cell_min_molecules"]
            _genes_min_cells = self.bagel_config["phenotypic_manifold_config"][
                "palantir"
            ]["_genes_min_cells"]
            n_components_pca = self.bagel_config["phenotypic_manifold_config"][
                "palantir"
            ]["n_components_pca"]

            print("Compute manifold.")

            two_data_set_flag = False

            if self.total_datasets == 2:
                project_one_secondary_only = False  # Testing
                if project_one_secondary_only is True:
                    self.secondary_df = self.secondary_df.head(10)
                self.total_secondary_cells_used = 2
                phenotypic_manifold_pca_projections, log_norm_bagel_df = (
                    self.pca_combine_two_datasets(
                        n_components_pca, _cell_min_molecules, _genes_min_cells
                    )
                )
                two_data_set_flag = True

            else:

                print("(1/4) - Filter dataset.")
                # Filter data set
                filterd_main_df = self.palantir_filter_counts_data(
                    self.main_df,
                    cell_min_molecules=_cell_min_molecules,
                    genes_min_cells=_genes_min_cells,
                )
                norm_main_df = self.palantir_normalize_counts(filterd_main_df)
                log_norm_bagel_df = self.palantir_log_transform(norm_main_df)

                # Dimensionality reduction
                print("(2/4) - PCA")
                phenotypic_manifold_pca_projections = self.run_pca(log_norm_bagel_df)
            joblib.dump(
                log_norm_bagel_df,
                f"{self.result_folder}/log_norm_bagel_df.pkl",
                compress=3,
            )

            joblib.dump(
                two_data_set_flag,
                f"{self.result_folder}/two_data_set_flag.pkl",
                compress=3,
            )

            # Loop until continuous manifold is obtained
            continuous_manifold_flag = False
            phenotypic_manifold_pca_projections_main = (
                phenotypic_manifold_pca_projections.copy()
            )
            while continuous_manifold_flag is False:

                print("(3/4) Diffusion map.)")
                dm_res = self.palantir_run_diffusion_maps(
                    phenotypic_manifold_pca_projections_main,
                    n_components=self.diffusion_components,
                )

                print("(4/4) Multi scale distance.")
                ms_data = self.palantir_determine_multiscale_space(dm_res)

                # print("t-SNE (4/4) - removed")
                # tsne = dimension_reduction.run_tsne(ms_data)

                # Select PCA
                pca = PCA(n_components=2, svd_solver="randomized")
                phenotypic_manifold_pca_projections = pca.fit_transform(ms_data)
                phenotypic_manifold_pca_projections = pd.DataFrame(
                    phenotypic_manifold_pca_projections, index=log_norm_bagel_df.index
                )
                phenotypic_manifold_pca_projections.columns = ["x", "y"]

                # print("magic_imputation (5/5)")
                # imp_df = dimension_reduction.run_magic_imputation(norm_df, dm_res)
                if self.total_secondary_cells_used == 0:
                    j_1 = phenotypic_manifold_pca_projections.tail(
                        self.total_secondary_cells_used
                    )
                    plt.scatter(j_1["x"], j_1["y"], s=20, c="k")
                sizes = log_norm_bagel_df.sum(axis=1)  # Define the expressions per cell
                plt.scatter(
                    phenotypic_manifold_pca_projections["x"],
                    phenotypic_manifold_pca_projections["y"],
                    s=3,
                    c=sizes,
                    cmap=matplotlib.cm.Spectral_r,
                )
                plt.axis("off")
                plt.title("Phenotypic manifold")
                plt.colorbar()
                plt.show()
                time.sleep(1)
                plt.close()

                # Debug plot
                # temp = phenotypic_manifold_pca_projections.copy().reset_index()
                # temp_data1 = temp[temp["index"].str.contains("|".join(["Run"]))]
                # temp_data2 = temp[~temp["index"].str.contains("|".join(["Run"]))]
                # plt.scatter(
                #     temp_data1["x"],
                #     temp_data1["y"],
                #     s=3,
                #     c="r"
                # )
                # plt.scatter(
                #     temp_data2["x"],
                #     temp_data2["y"],
                #     s=3,
                #     c="g"
                # )
                # plot_label = f"{self.result_folder}/test_own.png"
                # plt.savefig(
                # plot_label,
                # bbox_inches="tight",
                # )
                # plt.close()

                continuous_manifold_flag = (
                    input("Was the manifold continuous? (Type True or False) : ")
                    .strip()
                    .lower()
                )

                while True:
                    if continuous_manifold_flag == "true":
                        continuous_manifold_flag = True
                        break
                    elif continuous_manifold_flag == "false":
                        continuous_manifold_flag = False
                        break
                    else:
                        print("Invalid input. Please enter 'true' or 'false'.")
                        continuous_manifold_flag = (
                            input(
                                "Was the manifold continuous? (Type True or False) : "
                            )
                            .strip()
                            .lower()
                        )

            # ################################################
            # Determine the boundary cell closest to user defined early cell
            data = ms_data
            dm_boundaries = pd.Index(set(data.idxmax()).union(data.idxmin()))
            dists = pairwise_distances(
                data.loc[dm_boundaries, :],
                data.loc[self.early_cell, :].values.reshape(1, -1),
            )
            start_cell = pd.Series(np.ravel(dists), index=dm_boundaries).idxmin()

            # Sample waypoints
            waypoints = self.palantir_max_min_sampling(
                data,
                self.bagel_config["phenotypic_manifold_config"]["palantir"][
                    "n_waypoints"
                ],
            )
            waypoints = waypoints.union(dm_boundaries)

            # No terminal states
            waypoints = pd.Index(waypoints.difference([start_cell]).unique())

            # Append start cell
            waypoints = pd.Index([start_cell]).append(waypoints)

            # Calculate pseudo time
            n_jobs = 1
            # :param num_jobs: Number of jobs for parallel processing = -1

            # pseudo_time, W = pseudo_time_calculations.palantir_compute_pseudo_time(imp_df, start_cell, 30, waypoints, n_jobs)
            pseudo_time, waypoints_weight_matrix = self.palantir_compute_pseudo_time(
                ms_data,
                start_cell,
                self.bagel_config["phenotypic_manifold_config"]["palantir"][
                    "pseudotime_knn"
                ],
                waypoints,
                n_jobs,
            )

            # Markov chain
            wp_data = data.loc[waypoints, :]
            constructed_markov_chain = self.palantir_construct_markov_chain(
                wp_data,
                self.bagel_config["phenotypic_manifold_config"]["palantir"][
                    "pseudotime_knn"
                ],
                pseudo_time,
                n_jobs,
            )

            # Terminal states
            terminal_states = self.palantir_terminal_states_from_markov_chain(
                constructed_markov_chain, wp_data, pseudo_time
            )

            # Excluded diffusion map boundaries
            dm_boundaries = pd.Index(set(wp_data.idxmax()).union(wp_data.idxmin()))
            excluded_boundaries = dm_boundaries.difference(terminal_states).difference(
                [start_cell]
            )

            # tsen data of
            wp_data_TSNE_ROW = phenotypic_manifold_pca_projections.loc[terminal_states]
            # Dump data to increase performance time
            phenotypic_manifold_pca_projections.to_pickle(
                f"{self.result_folder}/phenotypic_manifold_pca_projections.pkl"
            )
            joblib.dump(
                pseudo_time, f"{self.result_folder}/pseudo_time.pkl", compress=3
            )
            joblib.dump(waypoints, f"{self.result_folder}/waypoints.pkl", compress=3)
            joblib.dump(wp_data, f"{self.result_folder}/wp_data.pkl", compress=3)
            joblib.dump(
                constructed_markov_chain,
                f"{self.result_folder}/constructed_markov_chain.pkl",
                compress=3,
            )
            joblib.dump(
                terminal_states, f"{self.result_folder}/terminal_states.pkl", compress=3
            )
            joblib.dump(
                excluded_boundaries,
                f"{self.result_folder}/excluded_boundaries.pkl",
                compress=3,
            )

            sizes = log_norm_bagel_df.sum(axis=1)  # Define the expressions per cell

            # TODO plot output for debug
            # plt.scatter(
            #     phenotypic_manifold_pca_projections["x"],
            #     phenotypic_manifold_pca_projections["y"],
            #     s=3,
            #     c=sizes,
            #     cmap=matplotlib.cm.Spectral_r,
            # )
            # plt.scatter(
            #     phenotypic_manifold_pca_projections["x"],
            #     phenotypic_manifold_pca_projections["y"],
            #     s=20,
            #     c="k",
            # )
            # plt.axis("off")
            # plt.title("Phenotypic manifold")
            # plt.colorbar()
            # plot_label = f"{self.result_folder}/test_own.png"
            # plt.savefig(
            #     plot_label,
            #     bbox_inches="tight",
            # )

            # plt.close()

            # Set FLAG to false
        else:
            phenotypic_manifold_pca_projections = pd.read_pickle(
                f"{self.result_folder}/phenotypic_manifold_pca_projections.pkl"
            )
            pseudo_time = joblib.load(f"{self.result_folder}/pseudo_time.pkl")
            waypoints = joblib.load(f"{self.result_folder}/waypoints.pkl")
            wp_data = joblib.load(f"{self.result_folder}/wp_data.pkl")
            constructed_markov_chain = joblib.load(
                f"{self.result_folder}/constructed_markov_chain.pkl"
            )
            terminal_states = joblib.load(f"{self.result_folder}/terminal_states.pkl")
            excluded_boundaries = joblib.load(
                f"{self.result_folder}/excluded_boundaries.pkl"
            )

        return (
            phenotypic_manifold_pca_projections,
            pseudo_time,
            waypoints,
            wp_data,
            constructed_markov_chain,
            terminal_states,
            excluded_boundaries,
        )

    def palantir_pseudo_time(self):
        """
        palantir_pseudo_time
        """
