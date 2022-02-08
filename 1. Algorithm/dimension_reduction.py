import pandas as pd
import numpy as np

# from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.manifold import TSNE

# import phenograph

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, find, issparse
from scipy.sparse.linalg import eigs

"""
DISCLAIMER
The following functions
1) run_pca,
2) run_diffusion_maps,
3) run_magic_imputation,
4) determine_multiscale_space,
5) run_tsne,
was used from the Palantir algorithm:
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


def run_pca(data, n_components=300):
    """Run PCA

    :param data: Dataframe of cells X genes. Typicaly multiscale space diffusion components
    :param n_components: Number of principal components
    :return: PCA projections of the data and the explained variance
    """
    pca = PCA(n_components=n_components, svd_solver="randomized")
    data = data.apply(lambda x: pd.to_numeric(x, errors="ignore"))
    pca_projections = pca.fit_transform(data)
    pca_projections = pd.DataFrame(pca_projections, index=data.index)
    return pca_projections  # , pca.explained_variance_ratio_


def run_diffusion_maps(data_df, n_components, knn=30, n_jobs=-1):
    """Run Diffusion maps using the adaptive anisotropic kernel

    :param data_df: PCA projections of the data or adjancency matrix
    :param n_components: Number of diffusion components
    :return: Diffusion components, corresponding eigen values and the diffusion operator
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
    D, V = eigs(T, n_components, tol=1e-4, maxiter=1000)
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


def run_magic_imputation(data, dm_res, n_steps=3):
    """Run MAGIC imputation

    :param dm_res: Diffusion map results from run_diffusion_maps
    :param n_steps: Number of steps in the diffusion operator
    :return: Imputed data matrix
    """
    T_steps = dm_res["T"] ** n_steps
    imputed_data = pd.DataFrame(
        np.dot(T_steps.todense(), data), index=data.index, columns=data.columns
    )

    return imputed_data


def determine_multiscale_space(dm_res, n_eigs=None):
    """Determine multi scale space of the data

    :param dm_res: Diffusion map results from run_diffusion_maps
    :param n_eigs: Number of eigen vectors to use. If None specified, the number
            of eigen vectors will be determined using eigen gap
    :return: Multi scale data matrix
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


def run_tsne(data, n_dim=2, perplexity=150, **kwargs):
    """Run tSNE

    :param data: Dataframe of cells X genes. Typicaly multiscale space diffusion components
    :param n_dim: Number of dimensions for tSNE embedding
    :return: tSNE embedding of the data
    """
    tsne = TSNE(n_components=n_dim, perplexity=perplexity, **kwargs).fit_transform(
        data.values
    )
    tsne = pd.DataFrame(tsne, index=data.index)
    tsne.columns = ["x", "y"]
    return tsne
