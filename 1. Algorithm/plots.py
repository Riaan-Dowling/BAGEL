import numpy as np
import pandas as pd


import joblib
import matplotlib
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d.art3d import Poly3DCollection


from scipy.stats import multivariate_normal, norm

import matplotlib.cm as cm


from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

import os
import joblib

import gp
import data_import


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def palantir_pseudo_time_plot(pseudo_data, c, palantir_pseudo_time_plot_FLAG):
    import matplotlib
    import matplotlib.pyplot as plt

    if palantir_pseudo_time_plot_FLAG == True:

        wp_data_TSNE_ROW = joblib.load("wp_data_TSNE_ROW.pkl")
        terminal_states = joblib.load("terminal_states.pkl")

        early_cell = joblib.load("early_cell.pkl")
        wp_data = joblib.load("wp_data.pkl")
        dm_boundaries = pd.Index(set(wp_data.idxmax()).union(wp_data.idxmin()))

        # Define early cell
        early_cell_data = pseudo_data.loc[early_cell]
        excluded_boundaries = dm_boundaries.difference(terminal_states).difference(
            [early_cell]
        )
        # start_cell = pseudo_data.loc[excluded_boundaries]
        if excluded_boundaries.empty == True:
            start_cell = early_cell_data
        else:
            start_cell_data = pseudo_data.loc[excluded_boundaries]
            previous_dist = 1000000
            for i in range(len(excluded_boundaries)):
                # Select one cell
                select_one = start_cell_data.head(1)
                # Remove selected cell from data
                start_cell_data = start_cell_data[
                    ~start_cell_data["Pseudo_Time_normal"].isin(
                        select_one["Pseudo_Time_normal"].values
                    )
                ]
                # Calculate euclidean distance
                Data_euclidean_distance = np.linalg.norm(select_one - early_cell_data)
                if Data_euclidean_distance < previous_dist:
                    start_cell = select_one
                previous_dist = Data_euclidean_distance

        fig = plt.figure()
        ax = fig.add_subplot(111)
        img = ax.scatter(
            pseudo_data["tsne_1"],
            pseudo_data["tsne_2"],
            s=5,
            marker="o",
            cmap=matplotlib.cm.plasma,
            c=c,
            label="Single cell",
        )
        ax.scatter(
            wp_data_TSNE_ROW["tsne_1"],
            wp_data_TSNE_ROW["tsne_2"],
            s=50,
            marker="X",
            c="m",
            label="Terminal cell data set",
        )
        ax.scatter(
            start_cell["tsne_1"],
            start_cell["tsne_2"],
            c="c",
            marker="h",
            s=50,
            label="Start cell data set",
        )
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis="both", which="both", length=0)
        ax.set_ylabel(r"PC$_{\mathrm{v}}$2")
        ax.set_xlabel(r"PC$_{\mathrm{v}}$1")
        plt.title("Phenotypic manifold")
        plt.legend()
        fig.colorbar(img)
        plt.show()

    else:
        return


def original_manifold_plot(pseudo_data, c, original_manifold_plot_FLAG):
    import matplotlib
    import matplotlib.pyplot as plt

    if original_manifold_plot_FLAG == True:

        wp_data_TSNE_ROW = joblib.load("wp_data_TSNE_ROW.pkl")

        terminal_states = joblib.load("terminal_states.pkl")

        early_cell = joblib.load("early_cell.pkl")
        wp_data = joblib.load("wp_data.pkl")
        dm_boundaries = pd.Index(set(wp_data.idxmax()).union(wp_data.idxmin()))

        # Define early cell
        early_cell_data = pseudo_data.loc[early_cell]
        excluded_boundaries = dm_boundaries.difference(terminal_states).difference(
            [early_cell]
        )
        # start_cell = pseudo_data.loc[excluded_boundaries]
        if excluded_boundaries.empty == True:
            start_cell = early_cell_data
        else:
            start_cell_data = pseudo_data.loc[excluded_boundaries]
            previous_dist = 1000000
            for i in range(len(excluded_boundaries)):
                # Select one cell
                select_one = start_cell_data.head(1)
                # Remove selected cell from data
                start_cell_data = start_cell_data[
                    ~start_cell_data["Pseudo_Time_normal"].isin(
                        select_one["Pseudo_Time_normal"].values
                    )
                ]
                # Calculate euclidean distance
                Data_euclidean_distance = np.linalg.norm(select_one - early_cell_data)
                if Data_euclidean_distance < previous_dist:
                    start_cell = select_one
                previous_dist = Data_euclidean_distance

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(elev=15, azim=-45)
        ax.scatter(
            pseudo_data["Pseudo_Time_normal"],
            pseudo_data["tsne_1"],
            pseudo_data["tsne_2"],
            cmap=matplotlib.cm.plasma,
            c=c,
            marker="o",
            s=5,
            label="Single cell",
        )
        ax.scatter(
            wp_data_TSNE_ROW["Pseudo_Time_normal"],
            wp_data_TSNE_ROW["tsne_1"],
            wp_data_TSNE_ROW["tsne_2"],
            c="k",
            marker="X",
            s=50,
            label="Terminal states",
        )
        ax.scatter(
            start_cell["Pseudo_Time_normal"],
            start_cell["tsne_1"],
            start_cell["tsne_2"],
            c="c",
            marker="h",
            s=50,
            label="Start cell data set",
        )

        plt.legend()
        ax.set_zlabel(r"PC$_{\mathrm{v}}$2")
        ax.set_ylabel(r"PC$_{\mathrm{v}}$1")
        ax.set_xlabel("Pseudo time")
        plt.legend()
        plt.show()

    else:
        return


def VIDEO_window_plot_FOLLOW(
    window_number,
    normal_vector,
    covariance_length,
    back_face,
    front_face,
    window_pseudo_data,
    not_window_pseudo_data,
    p1_rotate,
    plane_x,
    plane_y,
    plane_z,
    M1_map_mean,
    M1_map_cov,
    M2_map_mean_G1,
    M2_map_cov_G1,
    M2_map_mean_G2,
    M2_map_cov_G2,
    Model_1,
    Model_2,
    split_VIDEO,
    previouse_window_L1,
    previouse_window_L2,
    estimate_original_pt,
    plane_normal_window_FLAG,
    plane_window_FLAG,
    VIDEO_window_plot_FOLLOW_FLAG,
    data_Association_after_split_Flag,
):

    if VIDEO_window_plot_FOLLOW_FLAG == True:
        parent_dir = os.path.dirname(os.path.realpath(__file__))
        resultPath = os.path.join(parent_dir, "VideoResultPictures")

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        azim_start = -180
        azim_end = 180
        azim_total_steps = 3
        azim_step_size = (azim_end - azim_start) / azim_total_steps
        azim_step = 0
        for a in range(azim_total_steps):
            max_tsne_1 = 1
            min_tsne_1 = -1

            max_tsne_2 = 1
            min_tsne_2 = -1
            # Plot window

            original_box = np.array(
                [
                    [back_face, min_tsne_1, min_tsne_2],
                    [back_face, max_tsne_1, min_tsne_2],
                    [back_face, max_tsne_1, max_tsne_2],
                    [back_face, min_tsne_1, max_tsne_2],
                ]
            )

            # Project points onto plane

            V = original_box - p1_rotate  # Obtaine vectors
            # print(V)

            step1 = (V).dot(normal_vector)  # Vectors magnitude
            step1 = pd.DataFrame(list(zip(step1, step1, step1)))
            # print(step1)
            cp_n = pd.DataFrame([normal_vector])
            cp2 = pd.concat([cp_n] * 4, ignore_index=True)
            # print(cp2)
            step_2 = cp2.mul(step1)  # project onto normal vector
            step_2.columns = ["Pseudo_Time_normal", "tsne_1", "tsne_2"]
            # print(step_2)
            points = original_box - step_2  # Minimise distance vector
            points = pd.DataFrame(points)

            # Plot widow data and 'Not' window data
            fig = plt.figure(figsize=(14, 8))
            ax = fig.add_subplot(121, projection="3d")
            ax.view_init(elev=3, azim=azim_start + azim_step)

            ax.set_zlabel(r"PC$_{\mathrm{v}}$2")
            ax.set_ylabel(r"PC$_{\mathrm{v}}$1")
            ax.set_xlabel("Pseudo time")

            ax.scatter(
                not_window_pseudo_data["Pseudo_Time_normal"],
                not_window_pseudo_data["tsne_1"],
                not_window_pseudo_data["tsne_2"],
                c="k",
                marker="o",
                s=5,
                label="Cell X",
                alpha=0.05,
            )
            if split_VIDEO == False:
                ax.scatter(
                    previouse_window_L1["Pseudo_Time_normal"],
                    previouse_window_L1["tsne_1"],
                    previouse_window_L1["tsne_2"],
                    s=20,
                    marker="o",
                    color="b",
                    label="Cells in window",
                )
            else:
                ax.scatter(
                    previouse_window_L1["Pseudo_Time_normal"],
                    previouse_window_L1["tsne_1"],
                    previouse_window_L1["tsne_2"],
                    s=20,
                    marker="o",
                    color="b",
                    label="Cells in window",
                )
                ax.scatter(
                    previouse_window_L2["Pseudo_Time_normal"],
                    previouse_window_L2["tsne_1"],
                    previouse_window_L2["tsne_2"],
                    s=20,
                    marker="o",
                    color="g",
                    label="Cells in window",
                )

            MEAN_window_pseudo_data = (
                window_pseudo_data.iloc[:, 0:3].mean(axis=0)
            ).values
            if data_Association_after_split_Flag == False:
                if (covariance_length[0]) > MEAN_window_pseudo_data[0]:
                    test_1 = covariance_length + MEAN_window_pseudo_data
                    line = Arrow3D(
                        [MEAN_window_pseudo_data[0], test_1[0]],
                        [MEAN_window_pseudo_data[1], test_1[1]],
                        [MEAN_window_pseudo_data[2], test_1[2]],
                        mutation_scale=30,
                        lw=5,
                        arrowstyle="-|>",
                        color="r",
                    )
                    ax.add_artist(line)

                else:
                    test_1 = -covariance_length + MEAN_window_pseudo_data
                    line = Arrow3D(
                        [MEAN_window_pseudo_data[0], test_1[0]],
                        [MEAN_window_pseudo_data[1], test_1[1]],
                        [MEAN_window_pseudo_data[2], test_1[2]],
                        mutation_scale=30,
                        lw=5,
                        arrowstyle="-|>",
                        color="r",
                    )
                    ax.add_artist(line)

            # Plot widow data
            ax = fig.add_subplot(122, projection="3d")
            ax.view_init(elev=3, azim=azim_start + azim_step)
            ax.set_zlabel(r"PC$_{\mathrm{v}}$2")
            ax.set_ylabel(r"PC$_{\mathrm{v}}$1")
            ax.set_xlabel("Pseudo time")
            if split_VIDEO == False:
                ax.scatter(
                    previouse_window_L1["Pseudo_Time_normal"],
                    previouse_window_L1["tsne_1"],
                    previouse_window_L1["tsne_2"],
                    s=20,
                    marker="o",
                    color="b",
                    label="Cells in window",
                )
            else:
                ax.scatter(
                    previouse_window_L1["Pseudo_Time_normal"],
                    previouse_window_L1["tsne_1"],
                    previouse_window_L1["tsne_2"],
                    s=20,
                    marker="o",
                    color="b",
                    label="Cells in window",
                )
                ax.scatter(
                    previouse_window_L2["Pseudo_Time_normal"],
                    previouse_window_L2["tsne_1"],
                    previouse_window_L2["tsne_2"],
                    s=20,
                    marker="o",
                    color="g",
                    label="Cells in window",
                )

            azim_step = azim_step + azim_step_size

            # plt.show()
            picknm = str(window_number) + "." + str(a + 1) + ".png"
            plt.savefig(f"{resultPath}/{picknm}")
            plt.close()

        plt.scatter(
            not_window_pseudo_data.tsne_1,
            not_window_pseudo_data.tsne_2,
            c="r",
            marker="x",
            s=10,
            label="Pseudo data",
        )
        plt.scatter(
            window_pseudo_data.tsne_1,
            window_pseudo_data.tsne_2,
            c="b",
            marker="o",
            s=10,
            label="Window Pseudo data",
        )
        plt.xlabel(r"PC$_{\mathrm{v}}$1")
        plt.ylabel(r"PC$_{\mathrm{v}}$2")
        # picknm= str('two_dim_window') + str(window_number)+".png"
        picknm = str(window_number) + "." + str(0) + ".png"
        plt.savefig(f"{resultPath}/{picknm}")
        plt.close()

        # plt.show()
        if window_number >= 44:
            qwerty = 1

    else:
        return
