"""
frenet frame
"""

import numpy as np
from numpy.linalg import eig
from sklearn.decomposition import PCA
import pandas as pd


def frenet_math(
    bagel_loop_data,
    window_removed_bagel_loop_data,
    window_interval_actual_samples,
    window_interval_extra_samples,
):
    """
    frenet_math
    """
    # """
    # -----------------------------------------------------------------
    # window_interval_extra_samples data
    # -----------------------------------------------------------------
    # """

    bagel_loop_data.reset_index(drop=True, inplace=True)
    window_removed_bagel_loop_data.reset_index(drop=True, inplace=True)
    pca_window_bagel_loop_data = window_removed_bagel_loop_data.head(
        window_interval_extra_samples
    )
    pca_window_bagel_loop_data.reset_index(drop=True, inplace=True)

    # """
    # -----------------------------------------------------------------
    # Pseudo-time bias
    # -----------------------------------------------------------------
    # """
    pt_bias = 10000  # Bias pt to ensure PCA is in the direction of PT
    x_scaled = (
        pt_bias * pca_window_bagel_loop_data["pseudo_time_normal"].values
    )  # returns a numpy array
    d = {
        "pseudo_time_normal": x_scaled,
        "pca_1": pca_window_bagel_loop_data["pca_1"],
        "pca_2": pca_window_bagel_loop_data["pca_2"],
    }
    df = pd.DataFrame(d)
    df.reset_index(drop=True, inplace=True)

    # """
    # -----------------------------------------------------------------
    # Compute PCA
    # -----------------------------------------------------------------
    # """

    # https://machinelearningmastery.com/calculate-principal-component-analysis-scratch-python/
    # define a matrix
    # A = np.array([[1, 2], [3, 4], [5, 6]])
    # calculate the mean of each column
    # M = np.mean(df.iloc[:, 0:3].values.T, axis=1)
    # # center columns by subtracting column means
    # C = df.iloc[:, 0:3].values - M
    # # calculate covariance matrix of centered matrix
    # V = np.cov(C.T)
    # # eigendecomposition of covariance matrix
    # values, vectors = eig(V)
    # P = vectors.T.dot(C.T)

    # Built in PCA
    pca = PCA(n_components=1)
    pca.fit(df.iloc[:, 0:3].values)

    # Undo pseudo-time bias
    nv = pca.components_[0]
    nv[0] = nv[0] / pt_bias
    a, b, c = nv
    unit_vector = (1 / (np.sqrt(a**2 + b**2 + c**2))) * nv
    variance_vector = nv * 3 * np.sqrt(pca.explained_variance_[0])
    # Taking the square root of the variance gives us the standard deviation.
    # By multiplying by 3, we're essentially capturing almost all (99.7%) of the variation along this component.

    pca.mean_[0] = pca.mean_[0] / pt_bias

    # """
    # -----------------------------------------------------------------
    # Project data onto PC-bias
    # -----------------------------------------------------------------
    # """
    M = np.mean(pca_window_bagel_loop_data.iloc[:, 0:3].values.T, axis=1)
    # center columns by subtracting column means
    C_2 = df.iloc[:, 0:3].values - M
    # calculate covariance matrix of centered matrix
    V = np.cov(C_2.T)
    # eigendecomposition of covariance matrix
    values, vectors = eig(V)
    # project data
    P = nv.T.dot(C_2.T)

    result_df = pd.DataFrame(P)
    result_df.columns = ["PC1"]
    result_df["cell_id_number"] = pca_window_bagel_loop_data["cell_id_number"].values

    # """
    # -----------------------------------------------------------------
    # Window interval slize
    # -----------------------------------------------------------------
    # """
    result_df.reset_index(drop=True, inplace=True)
    if (300 * variance_vector[0]) > pca.mean_[0]:
        new_window_bagel_loop_data = result_df.sort_values("PC1", ascending=True)
        new_window_bagel_loop_data = new_window_bagel_loop_data.head(
            window_interval_actual_samples
        )
    else:
        new_window_bagel_loop_data = result_df.sort_values("PC1", ascending=False)
        new_window_bagel_loop_data = new_window_bagel_loop_data.head(
            window_interval_actual_samples
        )

    new_window_bagel_loop_data.reset_index(drop=True, inplace=True)

    # Usable data after PCA anlysis
    window_bagel_loop_data = bagel_loop_data[
        bagel_loop_data["cell_id_number"].isin(
            new_window_bagel_loop_data["cell_id_number"].values
        )
    ]
    not_window_bagel_loop_data = bagel_loop_data[
        ~bagel_loop_data["cell_id_number"].isin(
            window_bagel_loop_data["cell_id_number"].values
        )
    ]
    return_window_removed_bagel_loop_data = window_removed_bagel_loop_data[
        ~window_removed_bagel_loop_data["cell_id_number"].isin(
            window_bagel_loop_data["cell_id_number"].values
        )
    ]
    window_bagel_loop_data.reset_index(drop=True, inplace=True)
    not_window_bagel_loop_data.reset_index(drop=True, inplace=True)

    # """
    # -----------------------------------------------------------------
    # Mean
    # -----------------------------------------------------------------
    # """

    mean_window_bagel_loop_data = (
        window_bagel_loop_data.iloc[:, 0:3].mean(axis=0)
    ).values

    # Create return df
    columns_df = df.columns.tolist()
    unit_vector = pd.DataFrame(
        [unit_vector.tolist()],
        columns=columns_df,
    )

    variance_vector = pd.DataFrame(
        [variance_vector.tolist()],
        columns=columns_df,
    )

    return (
        mean_window_bagel_loop_data,
        unit_vector,
        variance_vector,
        window_bagel_loop_data,
        not_window_bagel_loop_data,
        return_window_removed_bagel_loop_data,
    )


def frenet_frame_slice(
    bagel_loop_data,
    window_removed_bagel_loop_data,
    window_interval_actual_samples,
    window_interval_extra_samples,
    final_window,
):
    """
    frenet_frame_slice
    """

    print("Frenet frame estiamte start.")
    if final_window is True:  # Incroperate data that does not make a complete widow
        window_interval_actual_samples = len(
            window_removed_bagel_loop_data["pseudo_time_normal"]
        )
        window_interval_extra_samples = len(
            window_removed_bagel_loop_data["pseudo_time_normal"]
        )

    (
        mean_window_bagel_loop_data,
        unit_vector,
        variance_vector,
        window_bagel_loop_data,
        not_window_bagel_loop_data,
        return_window_removed_bagel_loop_data,
    ) = frenet_math(
        bagel_loop_data,
        window_removed_bagel_loop_data,
        window_interval_actual_samples,
        window_interval_extra_samples,
    )

    print("Frenet frame estiamte end.")
    return (
        mean_window_bagel_loop_data,
        unit_vector,
        variance_vector,
        window_bagel_loop_data,
        not_window_bagel_loop_data,
        return_window_removed_bagel_loop_data,
    )
