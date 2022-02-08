# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt


from sklearn.decomposition import PCA

# from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import math
import pandas as pd


from sklearn import preprocessing  # Normalise data [-1, 1]

min_max_scaler = preprocessing.MinMaxScaler()  # [0,1]


def Frenet(pseudo_data, pov_data, window_size, step_size):

    """
    -----------------------------------------------------------------
    Step size data
    -----------------------------------------------------------------
    """

    pseudo_data.reset_index(drop=True, inplace=True)
    pov_data.reset_index(drop=True, inplace=True)
    PCA_window_pseudo_data = pov_data.head(step_size)
    PCA_window_pseudo_data.reset_index(drop=True, inplace=True)

    """
    -----------------------------------------------------------------
    Pseudo-time bias
    -----------------------------------------------------------------
    """
    pt_bias = 10000  # Bias pt to ensure PCA is in the direction of PT
    x_scaled = (
        pt_bias * PCA_window_pseudo_data["Pseudo_Time_normal"].values
    )  # returns a numpy array
    d = {
        "Pseudo_Time_normal": x_scaled,
        "tsne_1": PCA_window_pseudo_data.tsne_1,
        "tsne_2": PCA_window_pseudo_data.tsne_2,
    }
    df = pd.DataFrame(d)
    df.reset_index(drop=True, inplace=True)

    """
    -----------------------------------------------------------------
    Compute PCA
    -----------------------------------------------------------------
    """

    from numpy.linalg import eig

    # https://machinelearningmastery.com/calculate-principal-component-analysis-scratch-python/
    # define a matrix
    A = np.array([[1, 2], [3, 4], [5, 6]])
    # calculate the mean of each column
    M = np.mean(df.iloc[:, 0:3].values.T, axis=1)
    # center columns by subtracting column means
    C = df.iloc[:, 0:3].values - M
    # calculate covariance matrix of centered matrix
    V = np.cov(C.T)
    # eigendecomposition of covariance matrix
    values, vectors = eig(V)
    P = vectors.T.dot(C.T)

    # Built in PCA
    pca = PCA(n_components=1)
    pca.fit(df.iloc[:, 0:3].values)

    # Undo pseudo-time bias
    nv = pca.components_[0]
    nv[0] = nv[0] / pt_bias
    a, b, c = nv
    normal_vector = (1 / (np.sqrt(a**2 + b**2 + c**2))) * nv
    v = nv * 3 * np.sqrt(pca.explained_variance_[0])
    pca.mean_[0] = pca.mean_[0] / pt_bias

    """
    -----------------------------------------------------------------
    Project data onto PC-bias
    -----------------------------------------------------------------
    """
    M = np.mean(PCA_window_pseudo_data.iloc[:, 0:3].values.T, axis=1)
    # center columns by subtracting column means
    C_2 = df.iloc[:, 0:3].values - M
    # calculate covariance matrix of centered matrix
    V = np.cov(C.T)
    # eigendecomposition of covariance matrix
    values, vectors = eig(V)
    # project data
    P = nv.T.dot(C_2.T)

    result_df = pd.DataFrame(P)
    result_df.columns = ["PC1"]
    result_df["cell_ID_number"] = PCA_window_pseudo_data["cell_ID_number"].values

    """
    -----------------------------------------------------------------
    Window size slize
    -----------------------------------------------------------------
    """
    result_df.reset_index(drop=True, inplace=True)
    if (300 * v[0]) > pca.mean_[0]:
        NEW_WINDOW_pseudo_time_data = result_df.sort_values("PC1", ascending=True)
        NEW_WINDOW_pseudo_time_data = NEW_WINDOW_pseudo_time_data.head(window_size)
    else:
        NEW_WINDOW_pseudo_time_data = result_df.sort_values("PC1", ascending=False)
        NEW_WINDOW_pseudo_time_data = NEW_WINDOW_pseudo_time_data.head(window_size)

    NEW_WINDOW_pseudo_time_data.reset_index(drop=True, inplace=True)

    # Usable data after PCA anlysis
    window_pseudo_data = pseudo_data[
        pseudo_data["cell_ID_number"].isin(
            NEW_WINDOW_pseudo_time_data["cell_ID_number"].values
        )
    ]
    not_window_pseudo_data = pseudo_data[
        ~pseudo_data["cell_ID_number"].isin(window_pseudo_data["cell_ID_number"].values)
    ]
    pov_return = pov_data[
        ~pov_data["cell_ID_number"].isin(window_pseudo_data["cell_ID_number"].values)
    ]
    window_pseudo_data.reset_index(drop=True, inplace=True)
    not_window_pseudo_data.reset_index(drop=True, inplace=True)

    """
    -----------------------------------------------------------------
    Mean
    -----------------------------------------------------------------
    """

    MEAN_window_pseudo_data = (window_pseudo_data.iloc[:, 0:3].mean(axis=0)).values

    # Future work
    plane_x = 1
    plane_z = 1
    plane_y = 1

    return (
        MEAN_window_pseudo_data,
        normal_vector,
        v,
        plane_x,
        plane_y,
        plane_z,
        window_pseudo_data,
        not_window_pseudo_data,
        pov_return,
    )  # , pseudo_time_correct, selection_data, window_distance


def pov_plane_slice(
    pseudo_data,
    pov_data,
    window_size,
    step_size,
    total_windows,
    final_window,
    window_itteration,
):

    print("Plane estiamte start.")
    if final_window == True:  # Incroperate data that does not make a complete widow
        final_window = True

        window_size = len(pov_data["Pseudo_Time_normal"])
        step_size = len(pov_data["Pseudo_Time_normal"])
        (
            MEAN_window_pseudo_data,
            normal_vector,
            v,
            plane_x,
            plane_y,
            plane_z,
            window_pseudo_data,
            not_window_pseudo_data,
            pov_return,
        ) = Frenet(pseudo_data, pov_data, window_size, step_size)

    else:
        (
            MEAN_window_pseudo_data,
            normal_vector,
            v,
            plane_x,
            plane_y,
            plane_z,
            window_pseudo_data,
            not_window_pseudo_data,
            pov_return,
        ) = Frenet(pseudo_data, pov_data, window_size, step_size)

    back_face = window_pseudo_data["Pseudo_Time_normal"].min()
    front_face = window_pseudo_data["Pseudo_Time_normal"].max()

    print("Plane estiamte end.")
    return (
        window_pseudo_data,
        window_pseudo_data,
        MEAN_window_pseudo_data,
        normal_vector,
        v,
        plane_x,
        plane_y,
        plane_z,
        window_pseudo_data,
        not_window_pseudo_data,
        front_face,
        back_face,
        final_window,
        pov_return,
    )
