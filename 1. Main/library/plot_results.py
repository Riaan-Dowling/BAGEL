"""
plotting
"""

import pandas as pd
import numpy as np


import matplotlib
import matplotlib.pyplot as plt

import matplotlib.cm as cm
import joblib
from mpl_toolkits.mplot3d import proj3d
import matplotlib.patches as mpatches

import os

from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import shutil
import library.gaussian_process as gp_sup


plt.rcParams["font.family"] = "Times New Roman"


class Arrow3D(FancyArrowPatch):
    """
    Arrow3D
    """

    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        """
        do_3d_projection
        """
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


def frenet_frame_plot_helper(column, variance_vector, mean_window_bagel_loop_data):
    """
    frenet_frame_plot_helper"""
    # TODO check functionality
    if (500 * variance_vector[column].values[0]) > mean_window_bagel_loop_data[
        column
    ].values[0]:
        return_value = (
            mean_window_bagel_loop_data[column].values[0]
            + variance_vector[column].values[0]
        )

    else:
        return_value = (
            mean_window_bagel_loop_data[column].values[0]
            + variance_vector[column].values[0]
        )

    return return_value


def determine_start_cell(bagel_loop_data, terminal_states, early_cell, dm_boundaries):
    """
    determine_early_cell
    """

    early_cell_data = bagel_loop_data.loc[early_cell]
    excluded_boundaries = dm_boundaries.difference(terminal_states).difference(
        [early_cell]
    )

    # start_cell = bagel_loop_data.loc[excluded_boundaries]
    if excluded_boundaries.empty is True:
        start_cell = early_cell_data
    else:
        start_cell_data = bagel_loop_data.loc[excluded_boundaries]
        previous_dist = 1000000
        for _ in range(len(excluded_boundaries)):
            # Select one cell
            select_one = start_cell_data.head(1)
            # Remove selected cell from data
            start_cell_data = start_cell_data[
                ~start_cell_data["pseudo_time_normal"].isin(
                    select_one["pseudo_time_normal"].values
                )
            ]
            # Calculate euclidean distance
            data_euclidean_distance = np.linalg.norm(select_one - early_cell_data)
            if data_euclidean_distance < previous_dist:
                start_cell = select_one
            previous_dist = data_euclidean_distance
    return start_cell


def results(
    primary_label,
    secondary_label,
    output_prefix_label,
    genelist,
    two_dimension_manifold_plot,
    three_dimension_manifold_plot,
    gene_expression_plot,
    bifurcation_plot,
    one_lineage_plot,
    all_lineage_plot,
    frenet_frame_plot,
    gp_with_data_plot,
    gp_only_plot,
    gp_per_lineage_plot,
    result_folder,
):
    """
    two_dimension_manifold_plot = False #Two dimensional phenotypic manifold plot
    three_dimension_manifold_plot = False #Three dimensional phenotypic manifold plot
    gene_expression_plot = False #Gene expressions of two dimensional phenotypic manifold plot
    bifurcation_plot = False # Detected bifurcation points plot
    one_lineage_plot = False # Plot one detected lineage at a time
    all_lineage_plot = False # Plot all detected lineage
    frenet_frame_plot = False # Plot Frenet frame representation
    gp_with_data_plot = False # Gaussian process with data plot
    gp_only_plot = False # Gaussian process only plot
    gp_per_lineage_plot = True # Plot one detected lineage with Gaussian process at a time
    """
    # Colours of graphs
    colors = ["b", "g", "gold", "sienna", "silver"]

    # lOAD DATA
    log_norm_main_df = joblib.load(f"{result_folder}/log_norm_main_df.pkl")
    bagel_loop_data = joblib.load(f"{result_folder}/bagel_loop_data.pkl")
    bagel_loop_data_terminal_state = joblib.load(
        f"{result_folder}/bagel_loop_data_terminal_state.pkl"
    )

    # link terminal state to normalized manifold
    terminal_states = joblib.load(f"{result_folder}/terminal_states.pkl")
    final_lineage_df = joblib.load(
        f"{result_folder}/final_lineage_df.pkl"
    )  # lineage clustes
    final_lineage_counter = joblib.load(
        f"{result_folder}/final_lineage_counter.pkl"
    )  # Total lineageas
    bifurcation_data = joblib.load(f"{result_folder}/bifurcation_data.pkl")
    total_bifurcations = joblib.load(f"{result_folder}/total_bifurcations.pkl")

    # Start cell datapoint
    wp_data = joblib.load(f"{result_folder}/wp_data.pkl")
    dm_boundaries = pd.Index(set(wp_data.idxmax()).union(wp_data.idxmin()))
    early_cell = joblib.load(f"{result_folder}/early_cell.pkl")

    # Define start cell
    start_cell = determine_start_cell(
        bagel_loop_data, terminal_states, early_cell, dm_boundaries
    )

    two_data_set_flag = joblib.load(f"{result_folder}/two_data_set_flag.pkl")
    # """
    # Create result folder
    # """

    output_dir = os.path.join(result_folder, "BAGEl_results")
    # delete old video folder if possible
    try:
        shutil.rmtree(output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    except Exception as e:
        print(e)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    if two_data_set_flag is True:
        print("Two data sets.")
        # """
        # -----------------------------------------------------------------
        # Parameters
        # -----------------------------------------------------------------
        # """
        total_secondary_cells_used = joblib.load("total_secondary_cells_used.pkl")
        total_length = len(bagel_loop_data.index)
        main_length = total_length - total_secondary_cells_used
        main_data = bagel_loop_data.head(main_length)
        secondary_data = bagel_loop_data.tail(total_secondary_cells_used)

        main_bagel_loop_data = bagel_loop_data.head(main_length)
        secondary_bagel_loop_data = bagel_loop_data.tail(total_secondary_cells_used)

        if two_dimension_manifold_plot is True:
            # """
            # -----------------------------------------------------------------
            # 2d manifold
            # -----------------------------------------------------------------
            # """
            try:
                selected_cell_row = joblib.load("selected_cell_row.pkl")
                selected_cell_row = bagel_loop_data.loc[selected_cell_row.index]
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.scatter(
                    main_data["pca_1"],
                    main_data["pca_2"],
                    marker="o",
                    s=10,
                    c="r",
                    label=primary_label,
                )
                ax.scatter(
                    secondary_data["pca_1"],
                    secondary_data["pca_2"],
                    marker="D",
                    s=10,
                    c="k",
                    label=secondary_label,
                )
                ax.scatter(
                    selected_cell_row["pca_1"],
                    selected_cell_row["pca_2"],
                    c="g",
                    marker="s",
                    s=50,
                    label="Reference cell",
                )
                ax.scatter(
                    bagel_loop_data_terminal_state["pca_1"],
                    bagel_loop_data_terminal_state["pca_2"],
                    c="m",
                    marker="X",
                    s=50,
                    label="Terminal cell datapoint",
                )
                ax.scatter(
                    start_cell["pca_1"],
                    start_cell["pca_2"],
                    c="c",
                    marker="h",
                    s=50,
                    label="Start cell datapoint",
                )
                plt.setp(ax.get_xticklabels(), visible=False)
                plt.setp(ax.get_yticklabels(), visible=False)
                ax.tick_params(axis="both", which="both", length=0)
                ax.set_ylabel(r"PC$_{\mathrm{v}}$2")
                ax.set_xlabel(r"PC$_{\mathrm{v}}$1")
                plt.title("Phenotypic manifold")
                plt.legend()
                # plt.show()
                picknm = output_prefix_label + "_2d_manifold.png"
                plt.savefig(f"{output_dir}/{picknm}")
                plt.close()
            except Exception as e:
                print(f"No reference cell - {e}")

                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.scatter(
                    main_data["pca_1"],
                    main_data["pca_2"],
                    marker="o",
                    s=10,
                    c="r",
                    label=primary_label,
                )
                ax.scatter(
                    secondary_data["pca_1"],
                    secondary_data["pca_2"],
                    marker="D",
                    s=10,
                    c="k",
                    label=secondary_label,
                )

                ax.scatter(
                    bagel_loop_data_terminal_state["pca_1"],
                    bagel_loop_data_terminal_state["pca_2"],
                    c="m",
                    marker="X",
                    s=50,
                    label="Terminal cell datapoint",
                )
                ax.scatter(
                    start_cell["pca_1"],
                    start_cell["pca_2"],
                    c="c",
                    marker="h",
                    s=50,
                    label="Start cell datapoint",
                )
                plt.setp(ax.get_xticklabels(), visible=False)
                plt.setp(ax.get_yticklabels(), visible=False)
                ax.tick_params(axis="both", which="both", length=0)
                ax.set_ylabel(r"PC$_{\mathrm{v}}$2")
                ax.set_xlabel(r"PC$_{\mathrm{v}}$1")
                plt.title("Phenotypic manifold")
                plt.legend()
                # plt.show()
                picknm = output_prefix_label + "_2d_manifold.png"
                plt.savefig(f"{output_dir}/{picknm}")
                plt.close()

            # """
            # -----------------------------------------------------------------
            # 2d pseudo_time
            # -----------------------------------------------------------------
            # """

            fig = plt.figure()
            ax = fig.add_subplot(111)
            img = ax.scatter(
                main_data["pca_1"],
                main_data["pca_2"],
                s=5,
                marker="o",
                cmap=matplotlib.cm.plasma,
                c=bagel_loop_data["pseudo_time_normal"].head(main_length),
                label=primary_label,
            )
            ax.scatter(
                secondary_data["pca_1"],
                secondary_data["pca_2"],
                s=20,
                marker="D",
                cmap=matplotlib.cm.plasma,
                c=bagel_loop_data["pseudo_time_normal"].tail(
                    total_secondary_cells_used
                ),
                label=secondary_label,
            )

            ax.scatter(
                bagel_loop_data_terminal_state["pca_1"],
                bagel_loop_data_terminal_state["pca_2"],
                s=50,
                marker="X",
                c="m",
                label="Terminal cell datapoint",
            )
            ax.scatter(
                start_cell["pca_1"],
                start_cell["pca_2"],
                c="c",
                marker="h",
                s=50,
                label="Start cell datapoint",
            )

            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis="both", which="both", length=0)
            ax.set_ylabel(r"PC$_{\mathrm{v}}$2")
            ax.set_xlabel(r"PC$_{\mathrm{v}}$1")
            plt.title("Phenotypic manifold")
            plt.legend()
            fig.colorbar(img)
            # plt.show()
            picknm = output_prefix_label + "_2d_manifold_PSEUDO_TIME.png"
            plt.savefig(f"{output_dir}/{picknm}")
            plt.close()

        if three_dimension_manifold_plot is True:
            # """
            # -----------------------------------------------------------------
            # 3d manifold
            # -----------------------------------------------------------------
            # """

            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.view_init(elev=12, azim=-100)
            ax.scatter(
                bagel_loop_data_terminal_state["pseudo_time_normal"],
                bagel_loop_data_terminal_state["pca_1"],
                bagel_loop_data_terminal_state["pca_2"],
                c="m",
                marker="X",
                s=50,
                label="Terminal cell datapoint",
            )
            ax.scatter(
                start_cell["pseudo_time_normal"],
                start_cell["pca_1"],
                start_cell["pca_2"],
                c="c",
                marker="h",
                s=50,
                label="Start cell datapoint",
            )
            temp1 = ax.scatter(
                main_bagel_loop_data["pseudo_time_normal"],
                main_bagel_loop_data["pca_1"],
                main_bagel_loop_data["pca_2"],
                c="r",
                marker="o",
                s=5,
                label=primary_label,
                alpha=1,
            )
            temp2 = ax.scatter(
                secondary_bagel_loop_data["pseudo_time_normal"],
                secondary_bagel_loop_data["pca_1"],
                secondary_bagel_loop_data["pca_2"],
                c="k",
                marker="D",
                s=20,
                label=secondary_label,
                alpha=1,
            )

            ax.set_zlabel(r"PC$_{\mathrm{v}}$2")
            ax.set_ylabel(r"PC$_{\mathrm{v}}$1")
            ax.set_xlabel("pseudo-time")
            ax.set_xticks([0, 0.5, 1])
            ax.set_yticks([-1, -0.5, 0, 0.5, 1])
            ax.set_zticks([-1, -0.5, 0, 0.5, 1])
            f = lambda x, y, z: proj3d.proj_transform(x, y, z, ax.get_proj())[:2]
            ax.legend(
                loc="lower left",
                bbox_to_anchor=f(0.8, -1, 0.8),
                bbox_transform=ax.transData,
            )
            temp1.set_alpha(0.2)
            temp2.set_alpha(0.5)
            picknm = output_prefix_label + "_3d_manifold.png"
            plt.savefig(f"{output_dir}/{picknm}")
            plt.close()

        if bifurcation_plot is True:
            # """
            # -----------------------------------------------------------------
            # Bifurcation points
            # -----------------------------------------------------------------
            # """
            # azim = [-88, -178 , -180]
            # elev = [12, 12 , 70]
            azim = [-100, -173, -172]
            elev = [12, 12, 70]
            temp = bifurcation_data
            for a in range(3):
                bifurcation_data = temp
                fig = plt.figure()
                ax = fig.add_subplot(111, projection="3d")
                ax.view_init(elev=elev[a], azim=azim[a])
                # colors = iter(cm.rainbow(np.linspace(0.5, 1, final_lineage_counter)))

                ax.scatter(
                    bagel_loop_data_terminal_state["pseudo_time_normal"],
                    bagel_loop_data_terminal_state["pca_1"],
                    bagel_loop_data_terminal_state["pca_2"],
                    c="m",
                    marker="X",
                    s=50,
                    label="Terminal cell datapoint",
                )
                ax.scatter(
                    start_cell["pseudo_time_normal"],
                    start_cell["pca_1"],
                    start_cell["pca_2"],
                    c="c",
                    marker="h",
                    s=50,
                    label="Start cell datapoint",
                )
                temp1 = ax.scatter(
                    main_bagel_loop_data["pseudo_time_normal"],
                    main_bagel_loop_data["pca_1"],
                    main_bagel_loop_data["pca_2"],
                    c="r",
                    marker="o",
                    s=5,
                    label=primary_label,
                    alpha=1,
                )
                temp2 = ax.scatter(
                    secondary_bagel_loop_data["pseudo_time_normal"],
                    secondary_bagel_loop_data["pca_1"],
                    secondary_bagel_loop_data["pca_2"],
                    c="k",
                    marker="D",
                    s=20,
                    label=secondary_label,
                    alpha=1,
                )

                for e in range(total_bifurcations):
                    # Bifurcation
                    bifurcation_data.columns = range(
                        bifurcation_data.shape[1]
                    )  # Rename column names
                    bifurcation_point = bifurcation_data.iloc[
                        :, 0:6
                    ]  # Select first SIX columns as new input
                    # Data = {'g1_pseudo_time_normal':ORIGINAl_window_1['pseudo_time_normal'],'g1_pca_1':ORIGINAl_window_1['tsne_1'],'g1_pca_2':ORIGINAl_window_1['tsne_2'],'g2_pseudo_time_normal':ORIGINAl_window_2['pseudo_time_normal'], 'g2_pca_1':ORIGINAl_window_2['tsne_1'],'g2_pca_2':ORIGINAl_window_2['tsne_2']}
                    bifurcation_point = (
                        bifurcation_point.dropna()
                    )  # Drop possible NAN data

                    bifurcation_data.columns = range(
                        bifurcation_data.shape[1]
                    )  # Rename column names
                    bifurcation_data = bifurcation_data.drop(
                        [0, 1, 2, 3, 4, 5], axis=1
                    )  # Remove new pseudo data lineage from data

                    bifurcation_point.columns = [
                        "g1_pseudo_time_normal",
                        "g1_pca_1",
                        "g1_pca_2",
                        "g2_pseudo_time_normal",
                        "g2_pca_1",
                        "g2_pca_2",
                    ]
                    # Data in original 3d window
                    d = {
                        "pseudo_time_normal": bifurcation_point[
                            "g1_pseudo_time_normal"
                        ],
                        "pca_1": bifurcation_point["g1_pca_1"],
                        "pca_2": bifurcation_point["g1_pca_2"],
                    }
                    current_ORIGINAl_window_g1_3d = pd.DataFrame(d)
                    current_ORIGINAl_window_g1_3d = (
                        current_ORIGINAl_window_g1_3d.dropna()
                    )  # Drop NAN rows

                    d = {
                        "pseudo_time_normal": bifurcation_point[
                            "g2_pseudo_time_normal"
                        ],
                        "pca_1": bifurcation_point["g2_pca_1"],
                        "pca_2": bifurcation_point["g2_pca_2"],
                    }
                    current_ORIGINAl_window_g2_3d = pd.DataFrame(d)
                    current_ORIGINAl_window_g2_3d = (
                        current_ORIGINAl_window_g2_3d.dropna()
                    )  # Drop NAN rows

                    bifurcation_point_mean_1 = current_ORIGINAl_window_g1_3d.head(1)
                    bifurcation_point_mean_2 = current_ORIGINAl_window_g2_3d.head(1)

                    bifurcation_point_mean_1.reset_index(drop=True, inplace=True)
                    bifurcation_point_mean_2.reset_index(drop=True, inplace=True)
                    Frames = [bifurcation_point_mean_1, bifurcation_point_mean_2]
                    new_data_FRAME = pd.concat(Frames, axis=0, sort=False)

                    test = new_data_FRAME.mean(axis=0)

                    ax.scatter(
                        (test["pseudo_time_normal"]),
                        test["pca_1"],
                        test["pca_2"],
                        color=colors[e],
                        marker="o",
                        s=200,
                        alpha=1,
                        label="Bifurcation point: " + str(e + 1),
                    )
                    # ax.scatter( (bifurcation_point_mean_2['pseudo_time_normal']), bifurcation_point_mean_2['tsne_1'], bifurcation_point_mean_2['tsne_2'],color =next(colors), marker='o', s = 200, alpha=1, label='Bifurcation point')

                ax.set_zlabel(r"PC$_{\mathrm{v}}$2")
                ax.set_ylabel(r"PC$_{\mathrm{v}}$1")
                ax.set_xlabel("pseudo-time")
                ax.set_xticks([0, 0.5, 1])
                ax.set_yticks([-1, -0.5, 0, 0.5, 1])
                ax.set_zticks([-1, -0.5, 0, 0.5, 1])
                f = lambda x, y, z: proj3d.proj_transform(x, y, z, ax.get_proj())[:2]
                ax.legend(
                    loc="lower left",
                    bbox_to_anchor=f(0.8, -1, 0.8),
                    bbox_transform=ax.transData,
                )
                temp1.set_alpha(0.3)
                temp2.set_alpha(0.3)
                # plt.show()
                picknm = output_prefix_label + "_bifurcation." + str(a + 1) + ".png"
                plt.savefig(f"{output_dir}/{picknm}")
                plt.close()

        if one_lineage_plot is True:
            # """
            # -----------------------------------------------------------------
            # 1 lineage at a time
            # -----------------------------------------------------------------
            # """

            # colors = iter(cm.rainbow(np.linspace(0.7, 1, final_lineage_counter)))

            for a in range(final_lineage_counter):
                lineage = final_lineage_df.iloc[
                    :, 0 + a * 3 : 3 + a * 3
                ].dropna()  # Select first three columns as new input

                lineage_terminal_state = lineage[
                    lineage["pseudo_time_normal"].isin(
                        bagel_loop_data_terminal_state["pseudo_time_normal"].values
                    )
                ]
                lineage_mouse = lineage[
                    lineage["pseudo_time_normal"].isin(
                        main_bagel_loop_data["pseudo_time_normal"].values
                    )
                ]
                lineage_human = lineage[
                    lineage["pseudo_time_normal"].isin(
                        secondary_bagel_loop_data["pseudo_time_normal"].values
                    )
                ]

                fig = plt.figure()
                ax = fig.add_subplot(111, projection="3d")
                ax.view_init(elev=12, azim=-100)

                # ax.scatter( lineage.pseudo_time_normal, lineage["pca_1"],  lineage["pca_2"], color =next(colors),alpha = 0.1, marker='o', s = 5, label='PC-lineage-' + str(a + 1))

                temp1 = ax.scatter(
                    lineage_mouse["pseudo_time_normal"],
                    lineage_mouse["pca_1"],
                    lineage_mouse["pca_2"],
                    c="r",
                    marker="o",
                    s=5,
                    label=primary_label,
                    alpha=1,
                )
                temp2 = ax.scatter(
                    lineage_human["pseudo_time_normal"],
                    lineage_human["pca_1"],
                    lineage_human["pca_2"],
                    c="k",
                    marker="D",
                    s=20,
                    label=secondary_label,
                    alpha=1,
                )

                ax.scatter(
                    lineage_terminal_state["pseudo_time_normal"],
                    lineage_terminal_state["pca_1"],
                    lineage_terminal_state["pca_2"],
                    c="m",
                    marker="X",
                    s=50,
                    label="Terminal cell datapoint",
                )
                ax.scatter(
                    start_cell["pseudo_time_normal"],
                    start_cell["pca_1"],
                    start_cell["pca_2"],
                    c="c",
                    marker="h",
                    s=50,
                    label="Start cell datapoint",
                )
                ax.set_zlabel(r"PC$_{\mathrm{v}}$2")
                ax.set_ylabel(r"PC$_{\mathrm{v}}$1")
                ax.set_xlabel("pseudo-time")
                ax.set_xticks([0, 0.5, 1])
                ax.set_yticks([-1, -0.5, 0, 0.5, 1])
                ax.set_zticks([-1, -0.5, 0, 0.5, 1])
                f = lambda x, y, z: proj3d.proj_transform(x, y, z, ax.get_proj())[:2]
                ax.legend(
                    loc="lower left",
                    bbox_to_anchor=f(0.8, -1, 0.8),
                    bbox_transform=ax.transData,
                )
                temp1.set_alpha(0.3)
                temp2.set_alpha(0.3)
                # plt.show()
                picknm = output_prefix_label + "_lineage_" + str(a + 1) + ".png"
                plt.savefig(f"{output_dir}/{picknm}")
                plt.close()

        if all_lineage_plot is True:
            # """
            # -----------------------------------------------------------------
            # All lineages
            # -----------------------------------------------------------------
            # """
            # colors = iter(cm.rainbow(np.linspace(0.7, 1, final_lineage_counter)))
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.view_init(elev=12, azim=-100)
            for a in range(final_lineage_counter):

                lineage = final_lineage_df.iloc[
                    :, 0 + a * 3 : 3 + a * 3
                ].dropna()  # Select first three columns as new input

                lineage_terminal_state = lineage[
                    lineage["pseudo_time_normal"].isin(
                        bagel_loop_data_terminal_state["pseudo_time_normal"].values
                    )
                ]
                lineage_mouse = lineage[
                    lineage["pseudo_time_normal"].isin(
                        main_bagel_loop_data["pseudo_time_normal"].values
                    )
                ]
                lineage_human = lineage[
                    lineage["pseudo_time_normal"].isin(
                        secondary_bagel_loop_data["pseudo_time_normal"].values
                    )
                ]

                temp1 = ax.scatter(
                    lineage_mouse["pseudo_time_normal"],
                    lineage_mouse["pca_1"],
                    lineage_mouse["pca_2"],
                    c="r",
                    marker="o",
                    s=5,
                    label=primary_label,
                    alpha=0.3,
                )
                temp2 = ax.scatter(
                    lineage_human["pseudo_time_normal"],
                    lineage_human["pca_1"],
                    lineage_human["pca_2"],
                    c="k",
                    marker="D",
                    s=20,
                    label=secondary_label,
                    alpha=0.3,
                )
                ax.scatter(
                    lineage_terminal_state["pseudo_time_normal"],
                    lineage_terminal_state["pca_1"],
                    lineage_terminal_state["pca_2"],
                    c="m",
                    marker="X",
                    s=50,
                    label="Terminal cell datapoint",
                )
            ax.scatter(
                start_cell["pseudo_time_normal"],
                start_cell["pca_1"],
                start_cell["pca_2"],
                c="c",
                marker="h",
                s=50,
                label="Start cell datapoint",
            )
            ax.set_zlabel(r"PC$_{\mathrm{v}}$2")
            ax.set_ylabel(r"PC$_{\mathrm{v}}$1")
            ax.set_xlabel("pseudo-time")
            ax.set_xticks([0, 0.5, 1])
            ax.set_yticks([-1, -0.5, 0, 0.5, 1])
            ax.set_zticks([-1, -0.5, 0, 0.5, 1])
            f = lambda x, y, z: proj3d.proj_transform(x, y, z, ax.get_proj())[:2]
            ax.legend(
                loc="lower left",
                bbox_to_anchor=f(0.8, -1, 0.8),
                bbox_transform=ax.transData,
            )
            # temp1.set_alpha(0.3)
            # temp2.set_alpha(0.3)
            # plt.show()
            picknm = output_prefix_label + "_All_lINEAGE.png"
            plt.savefig(f"{output_dir}/{picknm}")
            plt.close()

        if frenet_frame_plot is True:
            # """
            # -----------------------------------------------------------------
            # Frenet frame
            # -----------------------------------------------------------------
            # """

            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.view_init(elev=12, azim=-100)
            for t in range(final_lineage_counter):

                frenet_frame_window_variance_vector = joblib.load(
                    f"{result_folder}/frenet_frame_window_variance_vector"
                    + str(t)
                    + ".pkl"
                )
                frenet_frame_window_mean = joblib.load(
                    f"{result_folder}/frenet_frame_window_mean" + str(t) + ".pkl"
                )
                frenet_frame_window_counter = joblib.load(
                    f"{result_folder}/frenet_frame_window_counter" + str(t) + ".pkl"
                )
                # Frenet Frame
                # colors = iter(cm.rainbow(np.linspace(0.5, 1, final_lineage_counter)))
                # for variance_vector,mean_window_bagel_loop_data in range(frenet_frame_window_counter):
                for (_, variance_vector), (_, mean_window_bagel_loop_data) in zip(
                    frenet_frame_window_variance_vector.iterrows(),
                    frenet_frame_window_mean.iterrows(),
                ):
                    # load rows as dataframes
                    mean_window_bagel_loop_data = (
                        mean_window_bagel_loop_data.to_frame().T
                    )
                    variance_vector = variance_vector.to_frame().T

                    # Determine arrow directions
                    frenet_frame_arrow_end = mean_window_bagel_loop_data.copy()

                    frenet_frame_arrow_end["pseudo_time_normal"] = (
                        frenet_frame_plot_helper(
                            "pseudo_time_normal",
                            variance_vector,
                            mean_window_bagel_loop_data,
                        )
                    )
                    frenet_frame_arrow_end["pca_1"] = frenet_frame_plot_helper(
                        "pca_1",
                        variance_vector,
                        mean_window_bagel_loop_data,
                    )
                    frenet_frame_arrow_end["pca_2"] = frenet_frame_plot_helper(
                        "pca_2",
                        variance_vector,
                        mean_window_bagel_loop_data,
                    )

                    # Plot arrow
                    line = Arrow3D(
                        [
                            mean_window_bagel_loop_data["pseudo_time_normal"].values[0],
                            frenet_frame_arrow_end["pseudo_time_normal"].values[0],
                        ],
                        [
                            mean_window_bagel_loop_data["pca_1"].values[0],
                            frenet_frame_arrow_end["pca_1"].values[0],
                        ],
                        [
                            mean_window_bagel_loop_data["pca_2"].values[0],
                            frenet_frame_arrow_end["pca_2"].values[0],
                        ],
                        mutation_scale=10,
                        lw=2,
                        arrowstyle="-|>",
                        color="g",
                    )
                    ax.add_artist(line)

                    # Remove used arrows
                    frenet_frame_window_variance_vector.columns = range(
                        frenet_frame_window_variance_vector.shape[1]
                    )  # Rename column names
                    frenet_frame_window_variance_vector = (
                        frenet_frame_window_variance_vector.drop([0], axis=1)
                    )  # Remove new pseudo data lineage from data

                    frenet_frame_window_mean.columns = range(
                        frenet_frame_window_mean.shape[1]
                    )  # Rename column names
                    frenet_frame_window_mean = frenet_frame_window_mean.drop(
                        [0], axis=1
                    )  # Remove new pseudo data lineage from data
            ax.scatter(
                bagel_loop_data_terminal_state["pseudo_time_normal"],
                bagel_loop_data_terminal_state["pca_1"],
                bagel_loop_data_terminal_state["pca_2"],
                c="m",
                marker="X",
                s=50,
                label="Terminal cell datapoint",
            )
            ax.scatter(
                start_cell["pseudo_time_normal"],
                start_cell["pca_1"],
                start_cell["pca_2"],
                c="c",
                marker="h",
                s=50,
                label="Start cell datapoint",
            )
            temp1 = ax.scatter(
                main_bagel_loop_data["pseudo_time_normal"],
                main_bagel_loop_data["pca_1"],
                main_bagel_loop_data["pca_2"],
                c="r",
                marker="o",
                s=5,
                label=primary_label,
                alpha=1,
            )
            temp2 = ax.scatter(
                secondary_bagel_loop_data["pseudo_time_normal"],
                secondary_bagel_loop_data["pca_1"],
                secondary_bagel_loop_data["pca_2"],
                c="k",
                marker="D",
                s=20,
                label=secondary_label,
                alpha=1,
            )

            ax.set_zlabel(r"PC$_{\mathrm{v}}$2")
            ax.set_ylabel(r"PC$_{\mathrm{v}}$1")
            ax.set_xlabel("pseudo-time")
            ax.set_xticks([0, 0.5, 1])
            ax.set_yticks([-1, -0.5, 0, 0.5, 1])
            ax.set_zticks([-1, -0.5, 0, 0.5, 1])
            f = lambda x, y, z: proj3d.proj_transform(x, y, z, ax.get_proj())[:2]
            patch = mpatches.Patch(color="green", label=r"PC$_w$ vector")
            handles, labels = ax.get_legend_handles_labels()
            handles.append(patch)
            ax.legend(
                handles=handles,
                loc="lower left",
                bbox_to_anchor=f(0.8, -1, 0.8),
                bbox_transform=ax.transData,
            )
            temp1.set_alpha(0.01)
            temp2.set_alpha(0.05)
            # plt.show()
            picknm = output_prefix_label + "_frenet_frame.png"
            plt.savefig(f"{output_dir}/{picknm}")
            plt.close()

        if gene_expression_plot is True:
            # """
            # -----------------------------------------------------------------
            # Gene expressions
            # -----------------------------------------------------------------
            # """
            f, ax = plt.subplots(
                1, len(genelist), figsize=(10, 5), sharex=True, sharey=True
            )
            ax = ax.ravel()

            for ig, g in enumerate(genelist):
                img = ax[ig].scatter(
                    main_bagel_loop_data["pca_1"],
                    main_bagel_loop_data["pca_2"],
                    cmap=matplotlib.cm.Spectral_r,
                    c=log_norm_main_df.loc[main_bagel_loop_data.index, genelist[ig]],
                    marker="o",
                    s=5,
                    label=primary_label,
                )
                ax[ig].scatter(
                    secondary_bagel_loop_data["pca_1"],
                    secondary_bagel_loop_data["pca_2"],
                    cmap=matplotlib.cm.Spectral_r,
                    c=log_norm_main_df.loc[
                        secondary_bagel_loop_data.index, genelist[ig]
                    ],
                    marker="D",
                    s=20,
                    label=secondary_label,
                )
                ax[ig].scatter(
                    bagel_loop_data_terminal_state["pca_1"],
                    bagel_loop_data_terminal_state["pca_2"],
                    c="m",
                    marker="X",
                    s=50,
                    label="Terminal cell datapoint",
                )
                ax[ig].scatter(
                    start_cell["pca_1"],
                    start_cell["pca_2"],
                    c="c",
                    marker="h",
                    s=50,
                    label="Start cell datapoint",
                )
                ax[ig].set_title(g)
                plt.setp(ax[ig].get_xticklabels(), visible=False)
                plt.setp(ax[ig].get_yticklabels(), visible=False)
                ax[ig].tick_params(axis="both", which="both", length=0)
                ax[ig].set_ylabel(r"PC$_{\mathrm{v}}$2")
                ax[ig].set_xlabel(r"PC$_{\mathrm{v}}$1")
                f.colorbar(img, ax=ax[ig])
            ax[0].legend(loc="upper left")
            picknm = output_prefix_label + "_gene_expression.png"
            plt.savefig(f"{output_dir}/{picknm}")
            plt.close()

        if gp_with_data_plot is True:
            # """
            # -----------------------------------------------------------------
            # Gaussian procees with data
            # -----------------------------------------------------------------
            # """
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.view_init(elev=12, azim=-100)
            # colors = iter(cm.rainbow(np.linspace(0.7, 1, final_lineage_counter)))
            # Plot final lineages plot
            temp = final_lineage_df
            for a in range(final_lineage_counter):
                lineage = final_lineage_df.iloc[
                    :, 0 + a * 3 : 3 + a * 3
                ].dropna()  # Select first three columns as new input

                # Dataframe to array data
                x_train = np.array(lineage["pseudo_time_normal"].values.tolist())
                y_train = lineage["pca_1"].to_numpy()
                f_star_1, sigma_1, x_star = gp_sup.Gaussian_process_algorithm(
                    x_train, y_train
                )

                y_train = lineage["pca_2"].to_numpy()
                f_star_2, sigma_2, x_star = gp_sup.Gaussian_process_algorithm(
                    x_train, y_train
                )

                ax.plot(
                    x_star,
                    f_star_1,
                    f_star_2,
                    color=colors[a],
                    linewidth=5,
                    label="PC-lineage-" + str(a + 1),
                )
            ax.scatter(
                bagel_loop_data_terminal_state["pseudo_time_normal"],
                bagel_loop_data_terminal_state["pca_1"],
                bagel_loop_data_terminal_state["pca_2"],
                c="m",
                marker="X",
                s=50,
                label="Terminal cell datapoint",
            )
            ax.scatter(
                start_cell["pseudo_time_normal"],
                start_cell["pca_1"],
                start_cell["pca_2"],
                c="c",
                marker="h",
                s=50,
                label="Start cell datapoint",
            )
            temp1 = ax.scatter(
                main_bagel_loop_data["pseudo_time_normal"],
                main_bagel_loop_data["pca_1"],
                main_bagel_loop_data["pca_2"],
                c="r",
                marker="o",
                s=5,
                label=primary_label,
                alpha=1,
            )
            temp2 = ax.scatter(
                secondary_bagel_loop_data["pseudo_time_normal"],
                secondary_bagel_loop_data["pca_1"],
                secondary_bagel_loop_data["pca_2"],
                c="k",
                marker="D",
                s=20,
                label="Human UCB",
                alpha=1,
            )

            ax.set_zlabel(r"PC$_{\mathrm{v}}$2")
            ax.set_ylabel(r"PC$_{\mathrm{v}}$1")
            ax.set_xlabel("pseudo-time")

            ax.set_xticks([0, 0.5, 1])
            ax.set_yticks([-1, -0.5, 0, 0.5, 1])
            ax.set_zticks([-1, -0.5, 0, 0.5, 1])
            f = lambda x, y, z: proj3d.proj_transform(x, y, z, ax.get_proj())[:2]
            ax.legend(
                loc="lower left",
                bbox_to_anchor=f(0.8, -1, 0.8),
                bbox_transform=ax.transData,
            )
            temp1.set_alpha(0.05)
            temp2.set_alpha(0.5)
            # plt.show()
            picknm = output_prefix_label + "_Gaussian_process_and_data.png"
            plt.savefig(f"{output_dir}/{picknm}")
            plt.close()

        if gp_per_lineage_plot is True:
            # """
            # -----------------------------------------------------------------
            # Gaussian per lineage
            # -----------------------------------------------------------------
            # """
            for a in range(final_lineage_counter):
                lineage = final_lineage_df.iloc[
                    :, 0 + a * 3 : 3 + a * 3
                ].dropna()  # Select first three columns as new input
                # lineage = lineage.dropna()  # Drop possible NAN data

                # temp.columns = range(temp.shape[1])  # Rename column names
                # temp = temp.drop(
                #     [0, 1, 2], axis=1
                # )  # Remove new pseudo data lineage from data

                # Dataframe to array data
                x_train = np.array(lineage["pseudo_time_normal"].values.tolist())
                y_train = lineage["pca_1"].to_numpy()
                f_star_1, sigma_1, x_star = gp_sup.Gaussian_process_algorithm(
                    x_train, y_train
                )

                y_train = lineage["pca_2"].to_numpy()
                f_star_2, sigma_2, x_star = gp_sup.Gaussian_process_algorithm(
                    x_train, y_train
                )

                lineage_terminal_state = lineage[
                    lineage["pseudo_time_normal"].isin(
                        bagel_loop_data_terminal_state["pseudo_time_normal"].values
                    )
                ]

                lineage_mouse = lineage[
                    lineage["pseudo_time_normal"].isin(
                        main_bagel_loop_data["pseudo_time_normal"].values
                    )
                ]
                lineage_human = lineage[
                    lineage["pseudo_time_normal"].isin(
                        secondary_bagel_loop_data["pseudo_time_normal"].values
                    )
                ]

                fig = plt.figure()

                ax = fig.add_subplot(111, projection="3d")
                ax.view_init(elev=12, azim=-100)

                ax.plot(
                    x_star,
                    f_star_1,
                    f_star_2,
                    color=colors[a],
                    linewidth=5,
                    label="PC-lineage-" + str(a + 1),
                )
                temp1 = ax.scatter(
                    lineage_mouse["pseudo_time_normal"],
                    lineage_mouse["pca_1"],
                    lineage_mouse["pca_2"],
                    c="r",
                    marker="o",
                    s=5,
                    label=primary_label,
                    alpha=1,
                )
                temp2 = ax.scatter(
                    lineage_human["pseudo_time_normal"],
                    lineage_human["pca_1"],
                    lineage_human["pca_2"],
                    c="k",
                    marker="D",
                    s=20,
                    label="Human UCB",
                    alpha=1,
                )

                ax.scatter(
                    lineage_terminal_state["pseudo_time_normal"],
                    lineage_terminal_state["pca_1"],
                    lineage_terminal_state["pca_2"],
                    c="m",
                    marker="X",
                    s=50,
                    label="Terminal cell",
                )
                ax.scatter(
                    start_cell["pseudo_time_normal"],
                    start_cell["pca_1"],
                    start_cell["pca_2"],
                    c="c",
                    marker="h",
                    s=50,
                    label="Start cell",
                )

                ax.legend()
                ax.set_zlabel("t-SNE z")
                ax.set_ylabel("t-SNE y")
                ax.set_xlabel("pseudo-time")
                ax.set_zlabel(r"PC$_{\mathrm{v}}$2")
                ax.set_ylabel(r"PC$_{\mathrm{v}}$1")
                ax.set_xlabel("pseudo-time")
                ax.set_xticks([0, 0.5, 1])
                ax.set_yticks([-1, -0.5, 0, 0.5, 1])
                ax.set_zticks([-1, -0.5, 0, 0.5, 1])
                f = lambda x, y, z: proj3d.proj_transform(x, y, z, ax.get_proj())[:2]
                ax.legend(
                    loc="lower left",
                    bbox_to_anchor=f(0.8, -1, 0.8),
                    bbox_transform=ax.transData,
                )
                temp1.set_alpha(0.2)
                temp2.set_alpha(0.5)
                picknm = (
                    output_prefix_label
                    + "_Gaussian_process_PER_lineage_"
                    + str(a + 1)
                    + ".png"
                )
                plt.savefig(f"{output_dir}/{picknm}")
                plt.close()

        if gp_only_plot:
            # """
            # -----------------------------------------------------------------
            # Gaussian procees only
            # -----------------------------------------------------------------
            # """
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.view_init(elev=12, azim=-100)
            # colors = iter(cm.rainbow(np.linspace(0.7, 1, final_lineage_counter)))
            # Plot final lineages plot
            for a in range(final_lineage_counter):
                lineage = final_lineage_df.iloc[
                    :, 0 + a * 3 : 3 + a * 3
                ].dropna()  # Select first three columns as new input

                # Dataframe to array data
                x_train = np.array(lineage["pseudo_time_normal"].values.tolist())
                y_train = lineage["pca_1"].to_numpy()
                f_star_1, sigma_1, x_star = gp_sup.Gaussian_process_algorithm(
                    x_train, y_train
                )

                y_train = lineage["pca_2"].to_numpy()
                f_star_2, sigma_2, x_star = gp_sup.Gaussian_process_algorithm(
                    x_train, y_train
                )

                ax.plot(
                    x_star,
                    f_star_1,
                    f_star_2,
                    color=colors[a],
                    linewidth=5,
                    label="PC-lineage-" + str(a + 1),
                )
            ax.scatter(
                bagel_loop_data_terminal_state["pseudo_time_normal"],
                bagel_loop_data_terminal_state["pca_1"],
                bagel_loop_data_terminal_state["pca_2"],
                c="m",
                marker="X",
                s=50,
                label="Terminal cell",
            )
            ax.scatter(
                start_cell["pseudo_time_normal"],
                start_cell["pca_1"],
                start_cell["pca_2"],
                c="c",
                marker="h",
                s=50,
                label="Start cell",
            )
            plt.legend()
            ax.set_zlabel("t-SNE z")
            ax.set_ylabel("t-SNE y")
            ax.set_xlabel("pseudo-time")
            ax.set_zlabel(r"PC$_{\mathrm{v}}$2")
            ax.set_ylabel(r"PC$_{\mathrm{v}}$1")
            ax.set_xlabel("pseudo-time")
            ax.set_xticks([0, 0.5, 1])
            ax.set_yticks([-1, -0.5, 0, 0.5, 1])
            ax.set_zticks([-1, -0.5, 0, 0.5, 1])
            f = lambda x, y, z: proj3d.proj_transform(x, y, z, ax.get_proj())[:2]
            ax.legend(
                loc="lower left",
                bbox_to_anchor=f(0.8, -1, 0.8),
                bbox_transform=ax.transData,
            )
            picknm = output_prefix_label + "_Gaussian_process_only.png"
            plt.savefig(f"{output_dir}/{picknm}")
            plt.close()

    else:
        print("One data set.")
        if gene_expression_plot is True:
            # """
            # -----------------------------------------------------------------
            # Gene expressions
            # -----------------------------------------------------------------
            # """
            f, ax = plt.subplots(
                1, len(genelist), figsize=(10, 5), sharex=True, sharey=True
            )
            ax = ax.ravel()

            for ig, g in enumerate(genelist):
                img = ax[ig].scatter(
                    bagel_loop_data["pca_1"],
                    bagel_loop_data["pca_2"],
                    c=log_norm_main_df.loc[bagel_loop_data.index, genelist[ig]],
                    s=3,
                    cmap=matplotlib.cm.Spectral_r,
                    label=primary_label,
                )
                ax[ig].scatter(
                    bagel_loop_data_terminal_state["pca_1"],
                    bagel_loop_data_terminal_state["pca_2"],
                    c="m",
                    marker="X",
                    s=50,
                    label="Terminal cell datapoint",
                )
                ax[ig].scatter(
                    start_cell["pca_1"],
                    start_cell["pca_2"],
                    c="c",
                    marker="h",
                    s=50,
                    label="Start cell datapoint",
                )
                ax[ig].set_title(g)
                # ax[ig].set_axis_off()
                plt.setp(ax[ig].get_xticklabels(), visible=False)
                plt.setp(ax[ig].get_yticklabels(), visible=False)
                ax[ig].tick_params(axis="both", which="both", length=0)
                ax[ig].set_ylabel(r"PC$_{\mathrm{v}}$2")
                ax[ig].set_xlabel(r"PC$_{\mathrm{v}}$1")
                f.colorbar(img, ax=ax[ig])  # , label = 'Gene expression level')
            ax[0].legend(loc="upper left")
            # plt.show()
            picknm = output_prefix_label + "_gene_expression.png"
            plt.savefig(f"{output_dir}/{picknm}")
            plt.close()

        if two_dimension_manifold_plot is True:
            # """
            # -----------------------------------------------------------------
            # 2d manifold
            # -----------------------------------------------------------------
            # """

            try:
                selected_cell_row = joblib.load("selected_cell_row.pkl")
                selected_cell_row = bagel_loop_data.loc[selected_cell_row.index]
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.scatter(
                    bagel_loop_data["pca_1"],
                    bagel_loop_data["pca_2"],
                    marker="o",
                    s=10,
                    c="r",
                    label=primary_label,
                )
                ax.scatter(
                    selected_cell_row["pca_1"],
                    selected_cell_row["pca_2"],
                    c="g",
                    marker="s",
                    s=50,
                    label="Reference cell",
                )
                ax.scatter(
                    bagel_loop_data_terminal_state["pca_1"],
                    bagel_loop_data_terminal_state["pca_2"],
                    c="m",
                    marker="X",
                    s=50,
                    label="Terminal cell datapoint",
                )
                ax.scatter(
                    start_cell["pca_1"],
                    start_cell["pca_2"],
                    c="c",
                    marker="h",
                    s=50,
                    label="Start cell datapoint",
                )
                plt.setp(ax.get_xticklabels(), visible=False)
                plt.setp(ax.get_yticklabels(), visible=False)
                ax.tick_params(axis="both", which="both", length=0)
                ax.set_ylabel(r"PC$_{\mathrm{v}}$2")
                ax.set_xlabel(r"PC$_{\mathrm{v}}$1")
                plt.title("Phenotypic manifold")
                plt.legend()
                # plt.show()
                picknm = output_prefix_label + "_2d_manifold.png"
                plt.savefig(f"{output_dir}/{picknm}")
                plt.close()
            except Exception as e:
                print(e)
                matplotlib.pyplot.hsv()
                print("No reference cell")
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.scatter(
                    bagel_loop_data["pca_1"],
                    bagel_loop_data["pca_2"],
                    marker="o",
                    s=10,
                    c="r",
                    label=primary_label,
                )
                ax.scatter(
                    bagel_loop_data_terminal_state["pca_1"],
                    bagel_loop_data_terminal_state["pca_2"],
                    c="m",
                    marker="X",
                    s=50,
                    label="Terminal cell datapoint",
                )
                ax.scatter(
                    start_cell["pca_1"],
                    start_cell["pca_2"],
                    c="c",
                    marker="h",
                    s=50,
                    label="Start cell datapoint",
                )
                plt.setp(ax.get_xticklabels(), visible=False)
                plt.setp(ax.get_yticklabels(), visible=False)
                ax.tick_params(axis="both", which="both", length=0)
                ax.set_ylabel(r"PC$_{\mathrm{v}}$2")
                ax.set_xlabel(r"PC$_{\mathrm{v}}$1")
                plt.title("Phenotypic manifold")
                plt.legend()
                # plt.show()
                picknm = output_prefix_label + "_2d_manifold.png"
                plt.savefig(f"{output_dir}/{picknm}")
                plt.close()

            # """
            # -----------------------------------------------------------------
            # 2d pseudo_time
            # -----------------------------------------------------------------
            # """

            fig = plt.figure()
            ax = fig.add_subplot(111)
            img = ax.scatter(
                bagel_loop_data["pca_1"],
                bagel_loop_data["pca_2"],
                s=5,
                marker="o",
                cmap=matplotlib.cm.plasma,
                c=bagel_loop_data["pseudo_time_normal"].values.tolist(),
                label=primary_label,
            )
            ax.scatter(
                bagel_loop_data_terminal_state["pca_1"],
                bagel_loop_data_terminal_state["pca_2"],
                s=50,
                marker="X",
                c="m",
                label="Terminal cell datapoint",
            )
            ax.scatter(
                start_cell["pca_1"],
                start_cell["pca_2"],
                c="c",
                marker="h",
                s=50,
                label="Start cell datapoint",
            )
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis="both", which="both", length=0)
            ax.set_ylabel(r"PC$_{\mathrm{v}}$2")
            ax.set_xlabel(r"PC$_{\mathrm{v}}$1")
            plt.title("Phenotypic manifold")
            plt.legend()
            fig.colorbar(img)
            # plt.show()
            picknm = output_prefix_label + "_2d_manifold_PSEUDO_TIME.png"
            plt.savefig(f"{output_dir}/{picknm}")
            plt.close()

        if three_dimension_manifold_plot is True:
            # """
            # -----------------------------------------------------------------
            # 3d manifold
            # -----------------------------------------------------------------
            # """

            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.view_init(elev=12, azim=-100)
            ax.scatter(
                bagel_loop_data_terminal_state["pseudo_time_normal"],
                bagel_loop_data_terminal_state["pca_1"],
                bagel_loop_data_terminal_state["pca_2"],
                c="m",
                marker="X",
                s=50,
                label="Terminal cell datapoint",
            )
            ax.scatter(
                start_cell["pseudo_time_normal"],
                start_cell["pca_1"],
                start_cell["pca_2"],
                c="c",
                marker="h",
                s=50,
                label="Start cell datapoint",
            )
            temp1 = ax.scatter(
                bagel_loop_data["pseudo_time_normal"],
                bagel_loop_data["pca_1"],
                bagel_loop_data["pca_2"],
                c="r",
                marker="o",
                s=5,
                label=primary_label,
                alpha=1,
            )
            ax.set_zlabel(r"PC$_{\mathrm{v}}$2")
            ax.set_ylabel(r"PC$_{\mathrm{v}}$1")
            ax.set_xlabel("pseudo-time")
            ax.set_xticks([0, 0.5, 1])
            ax.set_yticks([-1, -0.5, 0, 0.5, 1])
            ax.set_zticks([-1, -0.5, 0, 0.5, 1])
            f = lambda x, y, z: proj3d.proj_transform(x, y, z, ax.get_proj())[:2]
            ax.legend(
                loc="lower left",
                bbox_to_anchor=f(0.8, -1, 0.8),
                bbox_transform=ax.transData,
            )
            temp1.set_alpha(0.2)
            # plt.show()
            picknm = output_prefix_label + "_3d_manifold.png"
            plt.savefig(f"{output_dir}/{picknm}")
            plt.close()

        if bifurcation_plot is True:
            # """
            # -----------------------------------------------------------------
            # Bifurcation points
            # -----------------------------------------------------------------
            # """
            azim = [-100, -173, -172]
            elev = [12, 12, 70]
            temp = bifurcation_data
            for a in range(3):
                fig = plt.figure()
                ax = fig.add_subplot(111, projection="3d")
                bifurcation_data = temp
                ax.view_init(elev=elev[a], azim=azim[a])
                # colors = iter(cm.rainbow(np.linspace(0.5, 1, final_lineage_counter)))
                # colors = ['b','g', 'gold', 'sienna', 'silver']

                temp1 = ax.scatter(
                    bagel_loop_data["pseudo_time_normal"],
                    bagel_loop_data["pca_1"],
                    bagel_loop_data["pca_2"],
                    c="r",
                    marker="o",
                    s=5,
                    label=primary_label,
                    alpha=1,
                )
                ax.scatter(
                    bagel_loop_data_terminal_state["pseudo_time_normal"],
                    bagel_loop_data_terminal_state["pca_1"],
                    bagel_loop_data_terminal_state["pca_2"],
                    c="m",
                    marker="X",
                    s=50,
                    label="Terminal cell datapoint",
                )
                ax.scatter(
                    start_cell["pseudo_time_normal"],
                    start_cell["pca_1"],
                    start_cell["pca_2"],
                    c="c",
                    marker="h",
                    s=50,
                    label="Start cell datapoint",
                )
                for e in range(total_bifurcations):
                    # Bifurcation
                    bifurcation_data.columns = range(
                        bifurcation_data.shape[1]
                    )  # Rename column names
                    bifurcation_point = bifurcation_data.iloc[
                        :, 0:6
                    ]  # Select first SIX columns as new input
                    # Data = {'g1_pseudo_time_normal':ORIGINAl_window_1['pseudo_time_normal'],'g1_pca_1':ORIGINAl_window_1['tsne_1'],'g1_pca_2':ORIGINAl_window_1['tsne_2'],'g2_pseudo_time_normal':ORIGINAl_window_2['pseudo_time_normal'], 'g2_pca_1':ORIGINAl_window_2['tsne_1'],'g2_pca_2':ORIGINAl_window_2['tsne_2']}
                    bifurcation_point = (
                        bifurcation_point.dropna()
                    )  # Drop possible NAN data

                    bifurcation_data.columns = range(
                        bifurcation_data.shape[1]
                    )  # Rename column names
                    bifurcation_data = bifurcation_data.drop(
                        [0, 1, 2, 3, 4, 5], axis=1
                    )  # Remove new pseudo data lineage from data

                    bifurcation_point.columns = [
                        "g1_pseudo_time_normal",
                        "g1_pca_1",
                        "g1_pca_2",
                        "g2_pseudo_time_normal",
                        "g2_pca_1",
                        "g2_pca_2",
                    ]
                    # Data in original 3d window
                    d = {
                        "pseudo_time_normal": bifurcation_point[
                            "g1_pseudo_time_normal"
                        ],
                        "pca_1": bifurcation_point["g1_pca_1"],
                        "pca_2": bifurcation_point["g1_pca_2"],
                    }
                    current_ORIGINAl_window_g1_3d = pd.DataFrame(d)
                    current_ORIGINAl_window_g1_3d = (
                        current_ORIGINAl_window_g1_3d.dropna()
                    )  # Drop NAN rows

                    d = {
                        "pseudo_time_normal": bifurcation_point[
                            "g2_pseudo_time_normal"
                        ],
                        "pca_1": bifurcation_point["g2_pca_1"],
                        "pca_2": bifurcation_point["g2_pca_2"],
                    }
                    current_ORIGINAl_window_g2_3d = pd.DataFrame(d)
                    current_ORIGINAl_window_g2_3d = (
                        current_ORIGINAl_window_g2_3d.dropna()
                    )  # Drop NAN rows

                    bifurcation_point_mean_1 = current_ORIGINAl_window_g1_3d.head(1)
                    bifurcation_point_mean_2 = current_ORIGINAl_window_g2_3d.head(1)

                    bifurcation_point_mean_1.reset_index(drop=True, inplace=True)
                    bifurcation_point_mean_2.reset_index(drop=True, inplace=True)
                    Frames = [bifurcation_point_mean_1, bifurcation_point_mean_2]
                    new_data_FRAME = pd.concat(Frames, axis=0, sort=False)

                    test = new_data_FRAME.mean(axis=0)

                    ax.scatter(
                        (test["pseudo_time_normal"]),
                        test["pca_1"],
                        test["pca_2"],
                        color=colors[e],
                        marker="o",
                        s=200,
                        alpha=1,
                        label="Bifurcation point: " + str(e + 1),
                    )
                    # ax.scatter( (bifurcation_point_mean_2['pseudo_time_normal']), bifurcation_point_mean_2['tsne_1'], bifurcation_point_mean_2['tsne_2'],color =next(colors), marker='o', s = 200, alpha=1, label='Bifurcation point')

                ax.set_zlabel(r"PC$_{\mathrm{v}}$2")
                ax.set_ylabel(r"PC$_{\mathrm{v}}$1")
                ax.set_xlabel("pseudo-time")
                ax.set_xticks([0, 0.5, 1])
                ax.set_yticks([-1, -0.5, 0, 0.5, 1])
                ax.set_zticks([-1, -0.5, 0, 0.5, 1])
                f = lambda x, y, z: proj3d.proj_transform(x, y, z, ax.get_proj())[:2]
                ax.legend(
                    loc="lower left",
                    bbox_to_anchor=f(0.8, -1, 0.8),
                    bbox_transform=ax.transData,
                )
                temp1.set_alpha(0.1)
                # plt.show()
                picknm = output_prefix_label + "_bifurcation." + str(a + 1) + ".png"
                plt.savefig(f"{output_dir}/{picknm}")
                plt.close()

        if one_lineage_plot is True:
            # """
            # -----------------------------------------------------------------
            # 1 lineage at a time
            # -----------------------------------------------------------------
            # """

            # colors = iter(cm.rainbow(np.linspace(0.7, 1, final_lineage_counter)))

            temp = final_lineage_df.copy()

            for a in range(final_lineage_counter):
                lineage = final_lineage_df.iloc[
                    :, 0 + a * 3 : 3 + a * 3
                ].dropna()  # Select first three columns as new input

                # Terminal states of the lineage
                lineage_terminal_state = lineage[
                    lineage["pseudo_time_normal"].isin(
                        bagel_loop_data_terminal_state["pseudo_time_normal"].values
                    )
                ]

                fig = plt.figure()
                ax = fig.add_subplot(111, projection="3d")
                ax.view_init(elev=12, azim=-100)

                ax.scatter(
                    lineage_terminal_state["pseudo_time_normal"],
                    lineage_terminal_state["pca_1"],
                    lineage_terminal_state["pca_2"],
                    c="m",
                    marker="X",
                    s=50,
                    label="Terminal cell datapoint",
                )
                ax.scatter(
                    start_cell["pseudo_time_normal"],
                    start_cell["pca_1"],
                    start_cell["pca_2"],
                    c="c",
                    marker="h",
                    s=50,
                    label="Start cell datapoint",
                )

                ax.scatter(
                    lineage.pseudo_time_normal,
                    lineage["pca_1"],
                    lineage["pca_2"],
                    color=colors[a],
                    marker="o",
                    s=5,
                    label="PC-lineage-" + str(a + 1),
                )
                ax.set_zlabel(r"PC$_{\mathrm{v}}$2")
                ax.set_ylabel(r"PC$_{\mathrm{v}}$1")
                ax.set_xlabel("pseudo-time")
                ax.set_xticks([0, 0.5, 1])
                ax.set_yticks([-1, -0.5, 0, 0.5, 1])
                ax.set_zticks([-1, -0.5, 0, 0.5, 1])
                f = lambda x, y, z: proj3d.proj_transform(x, y, z, ax.get_proj())[:2]
                ax.legend(
                    loc="lower left",
                    bbox_to_anchor=f(0.8, -1, 0.8),
                    bbox_transform=ax.transData,
                )
                # plt.show()
                picknm = output_prefix_label + "_lineage_" + str(a + 1) + ".png"
                # plt.show()
                plt.savefig(f"{output_dir}/{picknm}")
                plt.close()
        if all_lineage_plot is True:
            # """
            # -----------------------------------------------------------------
            # All lineages
            # -----------------------------------------------------------------
            # """
            # colors = iter(cm.rainbow(np.linspace(0.7, 1, final_lineage_counter)))
            temp = final_lineage_df
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.view_init(elev=12, azim=-100)
            for a in range(final_lineage_counter):
                lineage = final_lineage_df.iloc[
                    :, 0 + a * 3 : 3 + a * 3
                ].dropna()  # Select first three columns as new input
                # lineage = lineage.dropna()  # Drop possible NAN data
                # temp.columns = range(temp.shape[1])  # Rename column names
                # temp = temp.drop(
                #     [0, 1, 2], axis=1
                # )  # Remove new pseudo data lineage from data
                ax.scatter(
                    lineage.pseudo_time_normal,
                    lineage["pca_1"],
                    lineage["pca_2"],
                    color=colors[a],
                    alpha=1,
                    marker="o",
                    s=5,
                    label="PC-lineage-" + str(a + 1),
                )

            ax.scatter(
                bagel_loop_data_terminal_state["pseudo_time_normal"],
                bagel_loop_data_terminal_state["pca_1"],
                bagel_loop_data_terminal_state["pca_2"],
                c="m",
                marker="X",
                s=50,
                label="Terminal cell datapoint",
            )
            ax.scatter(
                start_cell["pseudo_time_normal"],
                start_cell["pca_1"],
                start_cell["pca_2"],
                c="c",
                marker="h",
                s=50,
                label="Start cell datapoint",
            )

            ax.set_zlabel(r"PC$_{\mathrm{v}}$2")
            ax.set_ylabel(r"PC$_{\mathrm{v}}$1")
            ax.set_xlabel("pseudo-time")
            ax.set_xticks([0, 0.5, 1])
            ax.set_yticks([-1, -0.5, 0, 0.5, 1])
            ax.set_zticks([-1, -0.5, 0, 0.5, 1])
            f = lambda x, y, z: proj3d.proj_transform(x, y, z, ax.get_proj())[:2]
            ax.legend(
                loc="lower left",
                bbox_to_anchor=f(0.8, -1, 0.8),
                bbox_transform=ax.transData,
            )
            temp1.set_alpha(0.3)
            # plt.show()
            picknm = output_prefix_label + "_All_lINEAGE.png"
            plt.savefig(f"{output_dir}/{picknm}")
            plt.close()

        if frenet_frame_plot is True:
            # """
            # -----------------------------------------------------------------
            # Frenet frame
            # -----------------------------------------------------------------
            # """

            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.view_init(elev=12, azim=-100)
            for t in range(final_lineage_counter):

                frenet_frame_window_variance_vector = joblib.load(
                    f"{result_folder}/frenet_frame_window_variance_vector"
                    + str(t)
                    + ".pkl"
                )
                frenet_frame_window_mean = joblib.load(
                    f"{result_folder}/frenet_frame_window_mean" + str(t) + ".pkl"
                )
                frenet_frame_window_counter = joblib.load(
                    f"{result_folder}/frenet_frame_window_counter" + str(t) + ".pkl"
                )
                # Frenet Frame
                # colors = iter(cm.rainbow(np.linspace(0.5, 1, final_lineage_counter)))
                # for variance_vector,mean_window_bagel_loop_data in range(frenet_frame_window_counter):
                for (_, variance_vector), (_, mean_window_bagel_loop_data) in zip(
                    frenet_frame_window_variance_vector.iterrows(),
                    frenet_frame_window_mean.iterrows(),
                ):
                    # load rows as dataframes
                    mean_window_bagel_loop_data = (
                        mean_window_bagel_loop_data.to_frame().T
                    )
                    variance_vector = variance_vector.to_frame().T

                    # Determine arrow directions
                    frenet_frame_arrow_end = mean_window_bagel_loop_data.copy()

                    frenet_frame_arrow_end["pseudo_time_normal"] = (
                        frenet_frame_plot_helper(
                            "pseudo_time_normal",
                            variance_vector,
                            mean_window_bagel_loop_data,
                        )
                    )
                    frenet_frame_arrow_end["pca_1"] = frenet_frame_plot_helper(
                        "pca_1",
                        variance_vector,
                        mean_window_bagel_loop_data,
                    )
                    frenet_frame_arrow_end["pca_2"] = frenet_frame_plot_helper(
                        "pca_2",
                        variance_vector,
                        mean_window_bagel_loop_data,
                    )

                    # Plot arrow
                    line = Arrow3D(
                        [
                            mean_window_bagel_loop_data["pseudo_time_normal"].values[0],
                            frenet_frame_arrow_end["pseudo_time_normal"].values[0],
                        ],
                        [
                            mean_window_bagel_loop_data["pca_1"].values[0],
                            frenet_frame_arrow_end["pca_1"].values[0],
                        ],
                        [
                            mean_window_bagel_loop_data["pca_2"].values[0],
                            frenet_frame_arrow_end["pca_2"].values[0],
                        ],
                        mutation_scale=10,
                        lw=2,
                        arrowstyle="-|>",
                        color="g",
                    )
                    ax.add_artist(line)
            temp1 = ax.scatter(
                bagel_loop_data["pseudo_time_normal"],
                bagel_loop_data["pca_1"],
                bagel_loop_data["pca_2"],
                c="r",
                marker="o",
                s=10,
                alpha=1,
                label=primary_label,
            )
            ax.scatter(
                bagel_loop_data_terminal_state["pseudo_time_normal"],
                bagel_loop_data_terminal_state["pca_1"],
                bagel_loop_data_terminal_state["pca_2"],
                c="m",
                marker="X",
                s=50,
                label="Terminal cell datapoint",
            )
            ax.scatter(
                start_cell["pseudo_time_normal"],
                start_cell["pca_1"],
                start_cell["pca_2"],
                c="c",
                marker="h",
                s=50,
                label="Start cell datapoint",
            )
            ax.set_zlabel(r"PC$_{\mathrm{v}}$2")
            ax.set_ylabel(r"PC$_{\mathrm{v}}$1")
            ax.set_xlabel("pseudo-time")
            ax.set_xticks([0, 0.5, 1])
            ax.set_yticks([-1, -0.5, 0, 0.5, 1])
            ax.set_zticks([-1, -0.5, 0, 0.5, 1])
            f = lambda x, y, z: proj3d.proj_transform(x, y, z, ax.get_proj())[:2]
            patch = mpatches.Patch(color="green", label=r"PC$_w$ vector")
            handles, labels = ax.get_legend_handles_labels()
            handles.append(patch)
            ax.legend(
                handles=handles,
                loc="lower left",
                bbox_to_anchor=f(0.8, -1, 0.8),
                bbox_transform=ax.transData,
            )
            temp1.set_alpha(0.01)
            picknm = output_prefix_label + "_frenet_frame.png"
            # plt.show()
            plt.savefig(f"{output_dir}/{picknm}")
            plt.close()

        if gp_with_data_plot is True:
            # """
            # -----------------------------------------------------------------
            # Gaussian procees with data
            # -----------------------------------------------------------------
            # """

            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.view_init(elev=12, azim=-100)
            # colors = iter(cm.rainbow(np.linspace(0.7, 1, final_lineage_counter)))
            # temp = final_lineage_df
            # Plot final lineages plot
            for a in range(final_lineage_counter):
                lineage = final_lineage_df.iloc[
                    :, 0 + a * 3 : 3 + a * 3
                ].dropna()  # Select first three columns as new input
                # lineage = lineage.dropna()  # Drop possible NAN data

                # temp.columns = range(temp.shape[1])  # Rename column names
                # temp = temp.drop(
                #     [0, 1, 2], axis=1
                # )  # Remove new pseudo data lineage from data

                # Dataframe to array data
                x_train = np.array(lineage["pseudo_time_normal"].values.tolist())
                y_train = lineage["pca_1"].to_numpy()
                f_star_1, sigma_1, x_star = gp_sup.Gaussian_process_algorithm(
                    x_train, y_train
                )

                y_train = lineage["pca_2"].to_numpy()
                f_star_2, sigma_2, x_star = gp_sup.Gaussian_process_algorithm(
                    x_train, y_train
                )
                ax.plot(
                    x_star,
                    f_star_1,
                    f_star_2,
                    color=colors[a],
                    linewidth=5,
                    label="PC-lineage-" + str(a + 1),
                )
            ax.scatter(
                bagel_loop_data_terminal_state["pseudo_time_normal"],
                bagel_loop_data_terminal_state["pca_1"],
                bagel_loop_data_terminal_state["pca_2"],
                c="m",
                marker="X",
                s=50,
                label="Terminal cell",
            )
            ax.scatter(
                start_cell["pseudo_time_normal"],
                start_cell["pca_1"],
                start_cell["pca_2"],
                c="c",
                marker="h",
                s=50,
                label="Start cell",
            )
            temp1 = ax.scatter(
                bagel_loop_data["pseudo_time_normal"],
                bagel_loop_data["pca_1"],
                bagel_loop_data["pca_2"],
                c="k",
                marker="o",
                s=10,
                alpha=1,
                label=primary_label,
            )

            ax.set_zlabel(r"PC$_{\mathrm{v}}$2")
            ax.set_ylabel(r"PC$_{\mathrm{v}}$1")
            ax.set_xlabel("pseudo-time")
            # plt.show()
            ax.set_xticks([0, 0.5, 1])
            ax.set_yticks([-1, -0.5, 0, 0.5, 1])
            ax.set_zticks([-1, -0.5, 0, 0.5, 1])
            f = lambda x, y, z: proj3d.proj_transform(x, y, z, ax.get_proj())[:2]
            ax.legend(
                loc="lower left",
                bbox_to_anchor=f(0.8, -1, 0.8),
                bbox_transform=ax.transData,
            )
            temp1.set_alpha(0.005)
            picknm = output_prefix_label + "_Gaussian_process_and_data.png"
            plt.savefig(f"{output_dir}/{picknm}")
            plt.close()

        if gp_only_plot is True:
            # """
            # -----------------------------------------------------------------
            # Gaussian procees only
            # -----------------------------------------------------------------
            # """

            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.view_init(elev=12, azim=-100)
            # colors = iter(cm.rainbow(np.linspace(0.7, 1, final_lineage_counter)))
            # Plot final lineages plot
            for a in range(final_lineage_counter):
                lineage = final_lineage_df.iloc[
                    :, 0 + a * 3 : 3 + a * 3
                ].dropna()  # Select first three columns as new input
                # lineage = lineage.dropna()  # Drop possible NAN data

                # temp.columns = range(temp.shape[1])  # Rename column names
                # temp = temp.drop(
                #     [0, 1, 2], axis=1
                # )  # Remove new pseudo data lineage from data

                # Dataframe to array data
                x_train = np.array(lineage["pseudo_time_normal"].values.tolist())
                y_train = lineage["pca_1"].to_numpy()
                f_star_1, sigma_1, x_star = gp_sup.Gaussian_process_algorithm(
                    x_train, y_train
                )

                y_train = lineage["pca_2"].to_numpy()
                f_star_2, sigma_2, x_star = gp_sup.Gaussian_process_algorithm(
                    x_train, y_train
                )

                ax.plot(
                    x_star,
                    f_star_1,
                    f_star_2,
                    color=colors[a],
                    linewidth=5,
                    label="PC-lineage-" + str(a + 1),
                )
            ax.scatter(
                bagel_loop_data_terminal_state["pseudo_time_normal"],
                bagel_loop_data_terminal_state["pca_1"],
                bagel_loop_data_terminal_state["pca_2"],
                c="m",
                marker="X",
                s=50,
                label="Terminal cell",
            )
            ax.scatter(
                start_cell["pseudo_time_normal"],
                start_cell["pca_1"],
                start_cell["pca_2"],
                c="c",
                marker="h",
                s=50,
                label="Start cell",
            )
            ax.set_zlabel(r"PC$_{\mathrm{v}}$2")
            ax.set_ylabel(r"PC$_{\mathrm{v}}$1")
            ax.set_xlabel("pseudo-time")
            ax.set_xticks([0, 0.5, 1])
            ax.set_yticks([-1, -0.5, 0, 0.5, 1])
            ax.set_zticks([-1, -0.5, 0, 0.5, 1])
            f = lambda x, y, z: proj3d.proj_transform(x, y, z, ax.get_proj())[:2]
            ax.legend(
                loc="lower left",
                bbox_to_anchor=f(0.8, -1, 0.8),
                bbox_transform=ax.transData,
            )
            picknm = output_prefix_label + "_Gaussian_process_only.png"
            plt.savefig(f"{output_dir}/{picknm}")
            plt.close()

        if gp_per_lineage_plot is True:
            # """
            # -----------------------------------------------------------------
            # Gaussian per lineage
            # -----------------------------------------------------------------
            # """

            for a in range(final_lineage_counter):
                lineage = final_lineage_df.iloc[
                    :, 0 + a * 3 : 3 + a * 3
                ].dropna()  # Select first three columns as new input

                # Dataframe to array data
                x_train = np.array(lineage["pseudo_time_normal"].values.tolist())
                y_train = lineage["pca_1"].to_numpy()
                f_star_1, sigma_1, x_star = gp_sup.Gaussian_process_algorithm(
                    x_train, y_train
                )

                y_train = lineage["pca_2"].to_numpy()
                f_star_2, sigma_2, x_star = gp_sup.Gaussian_process_algorithm(
                    x_train, y_train
                )

                lineage_terminal_state = lineage[
                    lineage["pseudo_time_normal"].isin(
                        bagel_loop_data_terminal_state["pseudo_time_normal"].values
                    )
                ]

                fig = plt.figure()

                ax = fig.add_subplot(111, projection="3d")
                ax.view_init(elev=12, azim=-100)

                ax.plot(
                    x_star,
                    f_star_1,
                    f_star_2,
                    color=colors[a],
                    linewidth=5,
                    label="PC-lineage-" + str(a + 1),
                )
                temp1 = ax.scatter(
                    lineage["pseudo_time_normal"],
                    lineage["pca_1"],
                    lineage["pca_2"],
                    c="k",
                    marker="x",
                    s=2,
                    alpha=1,
                    label=primary_label,
                )
                ax.scatter(
                    lineage_terminal_state["pseudo_time_normal"],
                    lineage_terminal_state["pca_1"],
                    lineage_terminal_state["pca_2"],
                    c="m",
                    marker="X",
                    s=50,
                    label="Terminal cell",
                )
                ax.scatter(
                    start_cell["pseudo_time_normal"],
                    start_cell["pca_1"],
                    start_cell["pca_2"],
                    c="c",
                    marker="h",
                    s=50,
                    label="Start cell",
                )

                ax.set_zlabel(r"PC$_{\mathrm{v}}$2")
                ax.set_ylabel(r"PC$_{\mathrm{v}}$1")
                ax.set_xlabel("pseudo-time")
                ax.set_xticks([0, 0.5, 1])
                ax.set_yticks([-1, -0.5, 0, 0.5, 1])
                ax.set_zticks([-1, -0.5, 0, 0.5, 1])
                f = lambda x, y, z: proj3d.proj_transform(x, y, z, ax.get_proj())[:2]
                ax.legend(
                    loc="lower left",
                    bbox_to_anchor=f(0.8, -1, 0.8),
                    bbox_transform=ax.transData,
                )
                temp1.set_alpha(0.1)
                picknm = (
                    output_prefix_label
                    + "_Gaussian_process_PER_lineage_"
                    + str(a + 1)
                    + ".png"
                )
                plt.savefig(f"{output_dir}/{picknm}")
                plt.close()
