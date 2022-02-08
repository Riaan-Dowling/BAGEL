import pandas as pd
import numpy as np


import matplotlib
import matplotlib.pyplot as plt

import matplotlib.cm as cm
import joblib
from mpl_toolkits.mplot3d import proj3d
import matplotlib.patches as mpatches

from sklearn import preprocessing  # Normalise data [-1, 1]

import os
import plots

from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import shutil
import gp


plt.rcParams["font.family"] = "Times New Roman"


def results(
    Primary_label,
    Secondary_label,
    Picture_name,
    genelist,
    two_dimension_manifold_plot,
    three_dimension_manifold_plot,
    gene_expression_plot,
    bifurcation_plot,
    one_lineage_plot,
    all_lineage_plot,
    frenet_frame_plot,
    GP_with_data_plot,
    GP_only_plot,
    GP_per_lineage_plot,
):

    """
    two_dimension_manifold_plot = False #Two dimensional phenotypic manifold plot
    three_dimension_manifold_plot = False #Three dimensional phenotypic manifold plot
    gene_expression_plot = False #Gene expressions of two dimensional phenotypic manifold plot
    bifurcation_plot = False # Detected bifurcation points plot
    one_lineage_plot = False # Plot one detected lineage at a time
    all_lineage_plot = False # Plot all detected lineage
    frenet_frame_plot = False # Plot Frenet frame representation
    GP_with_data_plot = False # Gaussian process with data plot
    GP_only_plot = False # Gaussian process only plot
    GP_per_lineage_plot = True # Plot one detected lineage with Gaussian process at a time
    """
    # Colours of graphs
    colors = ["b", "g", "gold", "sienna", "silver"]

    class Arrow3D(FancyArrowPatch):
        def __init__(self, xs, ys, zs, *args, **kwargs):
            FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
            self._verts3d = xs, ys, zs

        def draw(self, renderer):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
            FancyArrowPatch.draw(self, renderer)

    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    tsnedata = os.path.join(THIS_FOLDER, "sample_tsne.p")
    tsne = pd.read_pickle(tsnedata)
    pseudo_time = joblib.load("pseudo_time.pkl")  # Calculated pseudo_time
    waypoints = joblib.load("waypoints.pkl")  # Calculated pseudo_time
    c = pseudo_time[tsne.index]

    norm_df = joblib.load("norm_df.pkl")  # Calculated pseudo_time

    # two_data_set_FLAG = False
    # joblib.dump(two_data_set_FLAG, "two_data_set_FLAG.pkl", compress=3)

    # pseudo-time dataframe

    # [0,1] normalise
    min_max_scaler = preprocessing.MinMaxScaler()  # [0,1]
    pseudo_time = min_max_scaler.fit_transform(pseudo_time.values.reshape(-1, 1))

    pseudo_time = pseudo_time.tolist()
    pseudo_time = [j for i in pseudo_time for j in i]

    # [-1,1] noramles
    min_max_scaler = preprocessing.MaxAbsScaler()
    tsne_1 = min_max_scaler.fit_transform(tsne["x"].values.reshape(-1, 1))
    tsne_2 = min_max_scaler.fit_transform(tsne["y"].values.reshape(-1, 1))

    tsne_1 = tsne_1.tolist()
    tsne_1 = [j for i in tsne_1 for j in i]

    tsne_2 = tsne_2.tolist()
    tsne_2 = [j for i in tsne_2 for j in i]

    d = {"Pseudo_Time_normal": pseudo_time, "tsne_1": tsne_1, "tsne_2": tsne_2}
    pseudo_data = pd.DataFrame(d, index=tsne.index)

    # Link terminal state to normalized manifold
    terminal_states = joblib.load("terminal_states.pkl")
    wp_data_TSNE_ROW = pseudo_data.loc[terminal_states]
    joblib.dump(wp_data_TSNE_ROW, "wp_data_TSNE_ROW.pkl", compress=3)

    Final_lineage_df = joblib.load("Final_lineage_df.pkl")  # Lineage clustes
    Final_lineage_counter = joblib.load("Final_lineage_counter.pkl")  # Total lineageas
    bifurcation_data = joblib.load("bifurcation_data.pkl")
    total_bifurcations = joblib.load("total_bifurcations.pkl")

    # early_cell = 'W30258'
    # joblib.dump(early_cell, "early_cell.pkl", compress=3)
    # Start cell datapoint
    wp_data = joblib.load("wp_data.pkl")
    dm_boundaries = pd.Index(set(wp_data.idxmax()).union(wp_data.idxmin()))
    early_cell = joblib.load("early_cell.pkl")
    # early_cell = 'Run5_164698952452459'

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

    two_data_set_FLAG = joblib.load("two_data_set_FLAG.pkl")

    """
    Create result folder
    """
    # Delete video folder
    parent_dir = os.path.dirname(os.path.realpath(__file__))
    resultPath = os.path.join(parent_dir, "BAGEL_results")
    # delete old video folder if possible
    try:
        shutil.rmtree(resultPath)
        if not os.path.exists(resultPath):
            os.makedirs(resultPath)
    except:
        if not os.path.exists(resultPath):
            os.makedirs(resultPath)

    if two_data_set_FLAG == True:
        print("Two data sets.")
        """
        -----------------------------------------------------------------
        Parameters
        -----------------------------------------------------------------
        """
        total_Secondary_cells_used = joblib.load("total_Secondary_cells_used.pkl")
        total_length = len(pseudo_data.index)
        main_length = total_length - total_Secondary_cells_used
        main_data = pseudo_data.head(main_length)
        secondary_Data = pseudo_data.tail(total_Secondary_cells_used)

        Main_pseudo_data = pseudo_data.head(main_length)
        Secondary_pseudo_data = pseudo_data.tail(total_Secondary_cells_used)

        sizes = norm_df.sum(axis=1)  # Define the expressions per cell

        if two_dimension_manifold_plot == True:
            """
            -----------------------------------------------------------------
            2d manifold
            -----------------------------------------------------------------
            """
            try:
                Selected_cell_row = joblib.load("Selected_cell_row.pkl")
                Selected_cell_row = pseudo_data.loc[Selected_cell_row.index]
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.scatter(
                    main_data["tsne_1"],
                    main_data["tsne_2"],
                    marker="o",
                    s=10,
                    c="r",
                    label=Primary_label,
                )
                ax.scatter(
                    secondary_Data["tsne_1"],
                    secondary_Data["tsne_2"],
                    marker="D",
                    s=10,
                    c="k",
                    label=Secondary_label,
                )
                ax.scatter(
                    Selected_cell_row["tsne_1"],
                    Selected_cell_row["tsne_2"],
                    c="g",
                    marker="s",
                    s=50,
                    label="Reference cell",
                )
                ax.scatter(
                    wp_data_TSNE_ROW["tsne_1"],
                    wp_data_TSNE_ROW["tsne_2"],
                    c="m",
                    marker="X",
                    s=50,
                    label="Terminal cell datapoint",
                )
                ax.scatter(
                    start_cell["tsne_1"],
                    start_cell["tsne_2"],
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
                picknm = Picture_name + "_2d_manifold.png"
                plt.savefig(f"{resultPath}/{picknm}")
                plt.close()
            except:
                print("No reference cell")

                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.scatter(
                    main_data["tsne_1"],
                    main_data["tsne_2"],
                    marker="o",
                    s=10,
                    c="r",
                    label=Primary_label,
                )
                ax.scatter(
                    secondary_Data["tsne_1"],
                    secondary_Data["tsne_2"],
                    marker="D",
                    s=10,
                    c="k",
                    label=Secondary_label,
                )

                ax.scatter(
                    wp_data_TSNE_ROW["tsne_1"],
                    wp_data_TSNE_ROW["tsne_2"],
                    c="m",
                    marker="X",
                    s=50,
                    label="Terminal cell datapoint",
                )
                ax.scatter(
                    start_cell["tsne_1"],
                    start_cell["tsne_2"],
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
                picknm = Picture_name + "_2d_manifold.png"
                plt.savefig(f"{resultPath}/{picknm}")
                plt.close()

            """
            -----------------------------------------------------------------
            2d pseudo_time
            -----------------------------------------------------------------
            """

            fig = plt.figure()
            ax = fig.add_subplot(111)
            img = ax.scatter(
                main_data["tsne_1"],
                main_data["tsne_2"],
                s=5,
                marker="o",
                cmap=matplotlib.cm.plasma,
                c=c.head(main_length),
                label=Primary_label,
            )
            ax.scatter(
                secondary_Data["tsne_1"],
                secondary_Data["tsne_2"],
                s=20,
                marker="D",
                cmap=matplotlib.cm.plasma,
                c=c.tail(total_Secondary_cells_used),
                label=Secondary_label,
            )

            ax.scatter(
                wp_data_TSNE_ROW["tsne_1"],
                wp_data_TSNE_ROW["tsne_2"],
                s=50,
                marker="X",
                c="m",
                label="Terminal cell datapoint",
            )
            ax.scatter(
                start_cell["tsne_1"],
                start_cell["tsne_2"],
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
            picknm = Picture_name + "_2d_manifold_PSEUDO_TIME.png"
            plt.savefig(f"{resultPath}/{picknm}")
            plt.close()

        if three_dimension_manifold_plot == True:
            """
            -----------------------------------------------------------------
            3d manifold
            -----------------------------------------------------------------
            """

            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.view_init(elev=12, azim=-100)
            ax.scatter(
                wp_data_TSNE_ROW["Pseudo_Time_normal"],
                wp_data_TSNE_ROW["tsne_1"],
                wp_data_TSNE_ROW["tsne_2"],
                c="m",
                marker="X",
                s=50,
                label="Terminal cell datapoint",
            )
            ax.scatter(
                start_cell["Pseudo_Time_normal"],
                start_cell["tsne_1"],
                start_cell["tsne_2"],
                c="c",
                marker="h",
                s=50,
                label="Start cell datapoint",
            )
            temp1 = ax.scatter(
                Main_pseudo_data["Pseudo_Time_normal"],
                Main_pseudo_data["tsne_1"],
                Main_pseudo_data["tsne_2"],
                c="r",
                marker="o",
                s=5,
                label=Primary_label,
                alpha=1,
            )
            temp2 = ax.scatter(
                Secondary_pseudo_data["Pseudo_Time_normal"],
                Secondary_pseudo_data["tsne_1"],
                Secondary_pseudo_data["tsne_2"],
                c="k",
                marker="D",
                s=20,
                label=Secondary_label,
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
            picknm = Picture_name + "_3d_manifold.png"
            plt.savefig(f"{resultPath}/{picknm}")
            plt.close()

        if bifurcation_plot == True:
            """
            -----------------------------------------------------------------
            Bifurcation points
            -----------------------------------------------------------------
            """
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
                # colors = iter(cm.rainbow(np.linspace(0.5, 1, Final_lineage_counter)))

                ax.scatter(
                    wp_data_TSNE_ROW["Pseudo_Time_normal"],
                    wp_data_TSNE_ROW["tsne_1"],
                    wp_data_TSNE_ROW["tsne_2"],
                    c="m",
                    marker="X",
                    s=50,
                    label="Terminal cell datapoint",
                )
                ax.scatter(
                    start_cell["Pseudo_Time_normal"],
                    start_cell["tsne_1"],
                    start_cell["tsne_2"],
                    c="c",
                    marker="h",
                    s=50,
                    label="Start cell datapoint",
                )
                temp1 = ax.scatter(
                    Main_pseudo_data["Pseudo_Time_normal"],
                    Main_pseudo_data["tsne_1"],
                    Main_pseudo_data["tsne_2"],
                    c="r",
                    marker="o",
                    s=5,
                    label=Primary_label,
                    alpha=1,
                )
                temp2 = ax.scatter(
                    Secondary_pseudo_data["Pseudo_Time_normal"],
                    Secondary_pseudo_data["tsne_1"],
                    Secondary_pseudo_data["tsne_2"],
                    c="k",
                    marker="D",
                    s=20,
                    label=Secondary_label,
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
                    # Data = {'G1_Pseudo_Time_normal':ORIGINAL_window_1['Pseudo_Time_normal'],'G1_tsne_1':ORIGINAL_window_1['tsne_1'],'G1_tsne_2':ORIGINAL_window_1['tsne_2'],'G2_Pseudo_Time_normal':ORIGINAL_window_2['Pseudo_Time_normal'], 'G2_tsne_1':ORIGINAL_window_2['tsne_1'],'G2_tsne_2':ORIGINAL_window_2['tsne_2']}
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
                        "G1_Pseudo_Time_normal",
                        "G1_tsne_1",
                        "G1_tsne_2",
                        "G2_Pseudo_Time_normal",
                        "G2_tsne_1",
                        "G2_tsne_2",
                    ]
                    # Data in original 3d window
                    d = {
                        "Pseudo_Time_normal": bifurcation_point[
                            "G1_Pseudo_Time_normal"
                        ],
                        "tsne_1": bifurcation_point["G1_tsne_1"],
                        "tsne_2": bifurcation_point["G1_tsne_2"],
                    }
                    current_ORIGINAL_window_G1_3d = pd.DataFrame(d)
                    current_ORIGINAL_window_G1_3d = (
                        current_ORIGINAL_window_G1_3d.dropna()
                    )  # Drop NAN rows

                    d = {
                        "Pseudo_Time_normal": bifurcation_point[
                            "G2_Pseudo_Time_normal"
                        ],
                        "tsne_1": bifurcation_point["G2_tsne_1"],
                        "tsne_2": bifurcation_point["G2_tsne_2"],
                    }
                    current_ORIGINAL_window_G2_3d = pd.DataFrame(d)
                    current_ORIGINAL_window_G2_3d = (
                        current_ORIGINAL_window_G2_3d.dropna()
                    )  # Drop NAN rows

                    bifurcation_point_mean_1 = current_ORIGINAL_window_G1_3d.head(1)
                    bifurcation_point_mean_2 = current_ORIGINAL_window_G2_3d.head(1)

                    bifurcation_point_mean_1.reset_index(drop=True, inplace=True)
                    bifurcation_point_mean_2.reset_index(drop=True, inplace=True)
                    Frames = [bifurcation_point_mean_1, bifurcation_point_mean_2]
                    new_data_FRAME = pd.concat(Frames, axis=0, sort=False)

                    test = new_data_FRAME.mean(axis=0)
                    test.columns = ["Pseudo_Time_normal", "tsne_1", "tsne_2"]
                    ax.scatter(
                        (test["Pseudo_Time_normal"]),
                        test["tsne_1"],
                        test["tsne_2"],
                        color=colors[e],
                        marker="o",
                        s=200,
                        alpha=1,
                        label="Bifurcation point: " + str(e + 1),
                    )
                    # ax.scatter( (bifurcation_point_mean_2['Pseudo_Time_normal']), bifurcation_point_mean_2['tsne_1'], bifurcation_point_mean_2['tsne_2'],color =next(colors), marker='o', s = 200, alpha=1, label='Bifurcation point')

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
                picknm = Picture_name + "_bifurcation." + str(a + 1) + ".png"
                plt.savefig(f"{resultPath}/{picknm}")
                plt.close()

        if one_lineage_plot == True:
            """
            -----------------------------------------------------------------
            1 Lineage at a time
            -----------------------------------------------------------------
            """

            # colors = iter(cm.rainbow(np.linspace(0.7, 1, Final_lineage_counter)))
            temp = Final_lineage_df

            for z in range(Final_lineage_counter):
                Lineage = temp.iloc[:, 0:3]  # Select first three columns as new input
                Lineage = Lineage.dropna()  # Drop possible NAN data

                temp.columns = range(temp.shape[1])  # Rename column names
                temp = temp.drop(
                    [0, 1, 2], axis=1
                )  # Remove new pseudo data lineage from data

                Lineage.columns = ["Pseudo_Time_normal", "tsne_1", "tsne_2"]

                lineage_TERMINAL_STATE = Lineage[
                    Lineage["Pseudo_Time_normal"].isin(
                        wp_data_TSNE_ROW["Pseudo_Time_normal"].values
                    )
                ]
                Lineage_mouse = Lineage[
                    Lineage["Pseudo_Time_normal"].isin(
                        Main_pseudo_data["Pseudo_Time_normal"].values
                    )
                ]
                Lineage_human = Lineage[
                    Lineage["Pseudo_Time_normal"].isin(
                        Secondary_pseudo_data["Pseudo_Time_normal"].values
                    )
                ]

                fig = plt.figure()
                ax = fig.add_subplot(111, projection="3d")
                ax.view_init(elev=12, azim=-100)

                # ax.scatter( Lineage.Pseudo_Time_normal, Lineage.tsne_1,  Lineage.tsne_2, color =next(colors),alpha = 0.1, marker='o', s = 5, label='PC-lineage-' + str(z + 1))

                temp1 = ax.scatter(
                    Lineage_mouse["Pseudo_Time_normal"],
                    Lineage_mouse["tsne_1"],
                    Lineage_mouse["tsne_2"],
                    c="r",
                    marker="o",
                    s=5,
                    label=Primary_label,
                    alpha=1,
                )
                temp2 = ax.scatter(
                    Lineage_human["Pseudo_Time_normal"],
                    Lineage_human["tsne_1"],
                    Lineage_human["tsne_2"],
                    c="k",
                    marker="D",
                    s=20,
                    label=Secondary_label,
                    alpha=1,
                )

                ax.scatter(
                    lineage_TERMINAL_STATE["Pseudo_Time_normal"],
                    lineage_TERMINAL_STATE["tsne_1"],
                    lineage_TERMINAL_STATE["tsne_2"],
                    c="m",
                    marker="X",
                    s=50,
                    label="Terminal cell datapoint",
                )
                ax.scatter(
                    start_cell["Pseudo_Time_normal"],
                    start_cell["tsne_1"],
                    start_cell["tsne_2"],
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
                picknm = Picture_name + "_lineage_" + str(z + 1) + ".png"
                plt.savefig(f"{resultPath}/{picknm}")
                plt.close()

        if all_lineage_plot == True:
            """
            -----------------------------------------------------------------
            All Lineages
            -----------------------------------------------------------------
            """
            # colors = iter(cm.rainbow(np.linspace(0.7, 1, Final_lineage_counter)))
            temp = Final_lineage_df
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.view_init(elev=12, azim=-100)
            for a in range(Final_lineage_counter):
                Lineage = temp.iloc[:, 0:3]  # Select first three columns as new input
                Lineage = Lineage.dropna()  # Drop possible NAN data
                temp.columns = range(temp.shape[1])  # Rename column names
                temp = temp.drop(
                    [0, 1, 2], axis=1
                )  # Remove new pseudo data lineage from data
                Lineage.columns = ["Pseudo_Time_normal", "tsne_1", "tsne_2"]

                lineage_TERMINAL_STATE = Lineage[
                    Lineage["Pseudo_Time_normal"].isin(
                        wp_data_TSNE_ROW["Pseudo_Time_normal"].values
                    )
                ]
                Lineage_mouse = Lineage[
                    Lineage["Pseudo_Time_normal"].isin(
                        Main_pseudo_data["Pseudo_Time_normal"].values
                    )
                ]
                Lineage_human = Lineage[
                    Lineage["Pseudo_Time_normal"].isin(
                        Secondary_pseudo_data["Pseudo_Time_normal"].values
                    )
                ]

                temp1 = ax.scatter(
                    Lineage_mouse["Pseudo_Time_normal"],
                    Lineage_mouse["tsne_1"],
                    Lineage_mouse["tsne_2"],
                    c="r",
                    marker="o",
                    s=5,
                    label=Primary_label,
                    alpha=0.3,
                )
                temp2 = ax.scatter(
                    Lineage_human["Pseudo_Time_normal"],
                    Lineage_human["tsne_1"],
                    Lineage_human["tsne_2"],
                    c="k",
                    marker="D",
                    s=20,
                    label=Secondary_label,
                    alpha=0.3,
                )
                ax.scatter(
                    lineage_TERMINAL_STATE["Pseudo_Time_normal"],
                    lineage_TERMINAL_STATE["tsne_1"],
                    lineage_TERMINAL_STATE["tsne_2"],
                    c="m",
                    marker="X",
                    s=50,
                    label="Terminal cell datapoint",
                )
            ax.scatter(
                start_cell["Pseudo_Time_normal"],
                start_cell["tsne_1"],
                start_cell["tsne_2"],
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
            picknm = Picture_name + "_ALL_LINEAGE.png"
            plt.savefig(f"{resultPath}/{picknm}")
            plt.close()

        if frenet_frame_plot == True:
            """
            -----------------------------------------------------------------
            Frenet frame
            -----------------------------------------------------------------
            """

            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.view_init(elev=12, azim=-100)
            for t in range(Final_lineage_counter):

                Frenet_frame_NORMAL_VECTOR = joblib.load(
                    "Frenet_frame_NORMAL_VECTOR" + str(t) + ".pkl"
                )
                Frenet_frame_MEAN = joblib.load("Frenet_frame_MEAN" + str(t) + ".pkl")
                Frenet_frame_COUNTER = joblib.load(
                    "Frenet_frame_COUNTER" + str(t) + ".pkl"
                )
                # Frenet Frame
                # colors = iter(cm.rainbow(np.linspace(0.5, 1, Final_lineage_counter)))
                for e in range(Frenet_frame_COUNTER):

                    covariance_length = (
                        Frenet_frame_NORMAL_VECTOR.iloc[:, 0:1].values
                    ).T
                    covariance_length = covariance_length[0]

                    MEAN_window_pseudo_data = (Frenet_frame_MEAN.iloc[:, 0:1].values).T
                    MEAN_window_pseudo_data = MEAN_window_pseudo_data[0]

                    if (500 * covariance_length[0]) > MEAN_window_pseudo_data[0]:
                        test_1 = covariance_length + MEAN_window_pseudo_data
                        line = Arrow3D(
                            [MEAN_window_pseudo_data[0], test_1[0]],
                            [MEAN_window_pseudo_data[1], test_1[1]],
                            [MEAN_window_pseudo_data[2], test_1[2]],
                            mutation_scale=10,
                            lw=2,
                            arrowstyle="-|>",
                            color="g",
                        )
                        ax.add_artist(line)

                    else:
                        test_1 = -covariance_length + MEAN_window_pseudo_data
                        line = Arrow3D(
                            [MEAN_window_pseudo_data[0], test_1[0]],
                            [MEAN_window_pseudo_data[1], test_1[1]],
                            [MEAN_window_pseudo_data[2], test_1[2]],
                            mutation_scale=10,
                            lw=2,
                            arrowstyle="-|>",
                            color="g",
                        )
                        ax.add_artist(line)

                    # Remove used arrows
                    Frenet_frame_NORMAL_VECTOR.columns = range(
                        Frenet_frame_NORMAL_VECTOR.shape[1]
                    )  # Rename column names
                    Frenet_frame_NORMAL_VECTOR = Frenet_frame_NORMAL_VECTOR.drop(
                        [0], axis=1
                    )  # Remove new pseudo data lineage from data

                    Frenet_frame_MEAN.columns = range(
                        Frenet_frame_MEAN.shape[1]
                    )  # Rename column names
                    Frenet_frame_MEAN = Frenet_frame_MEAN.drop(
                        [0], axis=1
                    )  # Remove new pseudo data lineage from data
            ax.scatter(
                wp_data_TSNE_ROW["Pseudo_Time_normal"],
                wp_data_TSNE_ROW["tsne_1"],
                wp_data_TSNE_ROW["tsne_2"],
                c="m",
                marker="X",
                s=50,
                label="Terminal cell datapoint",
            )
            ax.scatter(
                start_cell["Pseudo_Time_normal"],
                start_cell["tsne_1"],
                start_cell["tsne_2"],
                c="c",
                marker="h",
                s=50,
                label="Start cell datapoint",
            )
            temp1 = ax.scatter(
                Main_pseudo_data["Pseudo_Time_normal"],
                Main_pseudo_data["tsne_1"],
                Main_pseudo_data["tsne_2"],
                c="r",
                marker="o",
                s=5,
                label=Primary_label,
                alpha=1,
            )
            temp2 = ax.scatter(
                Secondary_pseudo_data["Pseudo_Time_normal"],
                Secondary_pseudo_data["tsne_1"],
                Secondary_pseudo_data["tsne_2"],
                c="k",
                marker="D",
                s=20,
                label=Secondary_label,
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
            picknm = Picture_name + "_Frenet_frame.png"
            plt.savefig(f"{resultPath}/{picknm}")
            plt.close()

        if gene_expression_plot == True:
            """
            -----------------------------------------------------------------
            Gene expressions
            -----------------------------------------------------------------
            """
            f, ax = plt.subplots(
                1, len(genelist), figsize=(10, 5), sharex=True, sharey=True
            )
            ax = ax.ravel()

            for ig, g in enumerate(genelist):
                img = ax[ig].scatter(
                    Main_pseudo_data["tsne_1"],
                    Main_pseudo_data["tsne_2"],
                    cmap=matplotlib.cm.Spectral_r,
                    c=norm_df.loc[Main_pseudo_data.index, genelist[ig]],
                    marker="o",
                    s=5,
                    label=Primary_label,
                )
                ax[ig].scatter(
                    Secondary_pseudo_data["tsne_1"],
                    Secondary_pseudo_data["tsne_2"],
                    cmap=matplotlib.cm.Spectral_r,
                    c=norm_df.loc[Secondary_pseudo_data.index, genelist[ig]],
                    marker="D",
                    s=20,
                    label=Secondary_label,
                )
                ax[ig].scatter(
                    wp_data_TSNE_ROW["tsne_1"],
                    wp_data_TSNE_ROW["tsne_2"],
                    c="m",
                    marker="X",
                    s=50,
                    label="Terminal cell datapoint",
                )
                ax[ig].scatter(
                    start_cell["tsne_1"],
                    start_cell["tsne_2"],
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
            picknm = Picture_name + "_gene_expression.png"
            plt.savefig(f"{resultPath}/{picknm}")
            plt.close()

        if GP_with_data_plot == True:
            """
            -----------------------------------------------------------------
            Gaussian procees with data
            -----------------------------------------------------------------
            """
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.view_init(elev=12, azim=-100)
            # colors = iter(cm.rainbow(np.linspace(0.7, 1, Final_lineage_counter)))
            # Plot final lineages plot
            temp = Final_lineage_df
            for z in range(Final_lineage_counter):
                Lineage = temp.iloc[:, 0:3]  # Select first three columns as new input
                Lineage = Lineage.dropna()  # Drop possible NAN data

                temp.columns = range(temp.shape[1])  # Rename column names
                temp = temp.drop(
                    [0, 1, 2], axis=1
                )  # Remove new pseudo data lineage from data
                Lineage.columns = ["Pseudo_Time_normal", "tsne_1", "tsne_2"]
                # Dataframe to array data
                X_train = np.array(Lineage["Pseudo_Time_normal"].values.tolist())
                Y_train = Lineage["tsne_1"].to_numpy()
                f_star_1, sigma_1, X_star = gp.Gaussian_process_algorithm(
                    X_train, Y_train
                )

                Y_train = Lineage["tsne_2"].to_numpy()
                f_star_2, sigma_2, X_star = gp.Gaussian_process_algorithm(
                    X_train, Y_train
                )

                ax.plot(
                    X_star,
                    f_star_1,
                    f_star_2,
                    color=colors[z],
                    linewidth=5,
                    label="PC-lineage-" + str(z + 1),
                )
            ax.scatter(
                wp_data_TSNE_ROW["Pseudo_Time_normal"],
                wp_data_TSNE_ROW["tsne_1"],
                wp_data_TSNE_ROW["tsne_2"],
                c="m",
                marker="X",
                s=50,
                label="Terminal cell datapoint",
            )
            ax.scatter(
                start_cell["Pseudo_Time_normal"],
                start_cell["tsne_1"],
                start_cell["tsne_2"],
                c="c",
                marker="h",
                s=50,
                label="Start cell datapoint",
            )
            temp1 = ax.scatter(
                Main_pseudo_data["Pseudo_Time_normal"],
                Main_pseudo_data["tsne_1"],
                Main_pseudo_data["tsne_2"],
                c="r",
                marker="o",
                s=5,
                label=Primary_label,
                alpha=1,
            )
            temp2 = ax.scatter(
                Secondary_pseudo_data["Pseudo_Time_normal"],
                Secondary_pseudo_data["tsne_1"],
                Secondary_pseudo_data["tsne_2"],
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
            picknm = Picture_name + "_Gaussian_process_and_data.png"
            plt.savefig(f"{resultPath}/{picknm}")
            plt.close()

        if GP_per_lineage_plot == True:
            """
            -----------------------------------------------------------------
            Gaussian per lineage
            -----------------------------------------------------------------
            """
            temp = Final_lineage_df
            for z in range(Final_lineage_counter):
                Lineage = temp.iloc[:, 0:3]  # Select first three columns as new input
                Lineage = Lineage.dropna()  # Drop possible NAN data

                temp.columns = range(temp.shape[1])  # Rename column names
                temp = temp.drop(
                    [0, 1, 2], axis=1
                )  # Remove new pseudo data lineage from data
                Lineage.columns = ["Pseudo_Time_normal", "tsne_1", "tsne_2"]
                # Dataframe to array data
                X_train = np.array(Lineage["Pseudo_Time_normal"].values.tolist())
                Y_train = Lineage["tsne_1"].to_numpy()
                f_star_1, sigma_1, X_star = gp.Gaussian_process_algorithm(
                    X_train, Y_train
                )

                Y_train = Lineage["tsne_2"].to_numpy()
                f_star_2, sigma_2, X_star = gp.Gaussian_process_algorithm(
                    X_train, Y_train
                )

                lineage_TERMINAL_STATE = Lineage[
                    Lineage["Pseudo_Time_normal"].isin(
                        wp_data_TSNE_ROW["Pseudo_Time_normal"].values
                    )
                ]

                Lineage_mouse = Lineage[
                    Lineage["Pseudo_Time_normal"].isin(
                        Main_pseudo_data["Pseudo_Time_normal"].values
                    )
                ]
                Lineage_human = Lineage[
                    Lineage["Pseudo_Time_normal"].isin(
                        Secondary_pseudo_data["Pseudo_Time_normal"].values
                    )
                ]

                fig = plt.figure()

                ax = fig.add_subplot(111, projection="3d")
                ax.view_init(elev=12, azim=-100)

                ax.plot(
                    X_star,
                    f_star_1,
                    f_star_2,
                    color=colors[z],
                    linewidth=5,
                    label="PC-lineage-" + str(z + 1),
                )
                temp1 = ax.scatter(
                    Lineage_mouse["Pseudo_Time_normal"],
                    Lineage_mouse["tsne_1"],
                    Lineage_mouse["tsne_2"],
                    c="r",
                    marker="o",
                    s=5,
                    label=Primary_label,
                    alpha=1,
                )
                temp2 = ax.scatter(
                    Lineage_human["Pseudo_Time_normal"],
                    Lineage_human["tsne_1"],
                    Lineage_human["tsne_2"],
                    c="k",
                    marker="D",
                    s=20,
                    label="Human UCB",
                    alpha=1,
                )

                ax.scatter(
                    lineage_TERMINAL_STATE["Pseudo_Time_normal"],
                    lineage_TERMINAL_STATE["tsne_1"],
                    lineage_TERMINAL_STATE["tsne_2"],
                    c="m",
                    marker="X",
                    s=50,
                    label="Terminal cell",
                )
                ax.scatter(
                    start_cell["Pseudo_Time_normal"],
                    start_cell["tsne_1"],
                    start_cell["tsne_2"],
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
                    Picture_name
                    + "_Gaussian_process_PER_lineage_"
                    + str(z + 1)
                    + ".png"
                )
                plt.savefig(f"{resultPath}/{picknm}")
                plt.close()

        if GP_only_plot:
            """
            -----------------------------------------------------------------
            Gaussian procees only
            -----------------------------------------------------------------
            """
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.view_init(elev=12, azim=-100)
            # colors = iter(cm.rainbow(np.linspace(0.7, 1, Final_lineage_counter)))
            # Plot final lineages plot
            temp = Final_lineage_df
            for z in range(Final_lineage_counter):
                Lineage = temp.iloc[:, 0:3]  # Select first three columns as new input
                Lineage = Lineage.dropna()  # Drop possible NAN data

                temp.columns = range(temp.shape[1])  # Rename column names
                temp = temp.drop(
                    [0, 1, 2], axis=1
                )  # Remove new pseudo data lineage from data
                Lineage.columns = ["Pseudo_Time_normal", "tsne_1", "tsne_2"]
                # Dataframe to array data
                X_train = np.array(Lineage["Pseudo_Time_normal"].values.tolist())
                Y_train = Lineage["tsne_1"].to_numpy()
                f_star_1, sigma_1, X_star = gp.Gaussian_process_algorithm(
                    X_train, Y_train
                )

                Y_train = Lineage["tsne_2"].to_numpy()
                f_star_2, sigma_2, X_star = gp.Gaussian_process_algorithm(
                    X_train, Y_train
                )

                ax.plot(
                    X_star,
                    f_star_1,
                    f_star_2,
                    color=colors[z],
                    linewidth=5,
                    label="PC-lineage-" + str(z + 1),
                )
            ax.scatter(
                wp_data_TSNE_ROW["Pseudo_Time_normal"],
                wp_data_TSNE_ROW["tsne_1"],
                wp_data_TSNE_ROW["tsne_2"],
                c="m",
                marker="X",
                s=50,
                label="Terminal cell",
            )
            ax.scatter(
                start_cell["Pseudo_Time_normal"],
                start_cell["tsne_1"],
                start_cell["tsne_2"],
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
            picknm = Picture_name + "_Gaussian_process_only.png"
            plt.savefig(f"{resultPath}/{picknm}")
            plt.close()

    else:
        print("One data set.")
        if gene_expression_plot == True:
            """
            -----------------------------------------------------------------
            Gene expressions
            -----------------------------------------------------------------
            """
            f, ax = plt.subplots(
                1, len(genelist), figsize=(10, 5), sharex=True, sharey=True
            )
            ax = ax.ravel()

            for ig, g in enumerate(genelist):
                img = ax[ig].scatter(
                    pseudo_data["tsne_1"],
                    pseudo_data["tsne_2"],
                    c=norm_df.loc[tsne.index, genelist[ig]],
                    s=3,
                    cmap=matplotlib.cm.Spectral_r,
                    label=Primary_label,
                )
                ax[ig].scatter(
                    wp_data_TSNE_ROW["tsne_1"],
                    wp_data_TSNE_ROW["tsne_2"],
                    c="m",
                    marker="X",
                    s=50,
                    label="Terminal cell datapoint",
                )
                ax[ig].scatter(
                    start_cell["tsne_1"],
                    start_cell["tsne_2"],
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
            picknm = Picture_name + "_gene_expression.png"
            plt.savefig(f"{resultPath}/{picknm}")
            plt.close()

        if two_dimension_manifold_plot == True:
            """
            -----------------------------------------------------------------
            2d manifold
            -----------------------------------------------------------------
            """

            try:
                Selected_cell_row = joblib.load("Selected_cell_row.pkl")
                Selected_cell_row = pseudo_data.loc[Selected_cell_row.index]
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.scatter(
                    pseudo_data["tsne_1"],
                    pseudo_data["tsne_2"],
                    marker="o",
                    s=10,
                    c="r",
                    label=Primary_label,
                )
                ax.scatter(
                    Selected_cell_row["tsne_1"],
                    Selected_cell_row["tsne_2"],
                    c="g",
                    marker="s",
                    s=50,
                    label="Reference cell",
                )
                ax.scatter(
                    wp_data_TSNE_ROW["tsne_1"],
                    wp_data_TSNE_ROW["tsne_2"],
                    c="m",
                    marker="X",
                    s=50,
                    label="Terminal cell datapoint",
                )
                ax.scatter(
                    start_cell["tsne_1"],
                    start_cell["tsne_2"],
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
                picknm = Picture_name + "_2d_manifold.png"
                plt.savefig(f"{resultPath}/{picknm}")
                plt.close()
            except:
                matplotlib.pyplot.hsv()
                print("No reference cell")
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.scatter(
                    pseudo_data["tsne_1"],
                    pseudo_data["tsne_2"],
                    marker="o",
                    s=10,
                    c="r",
                    label=Primary_label,
                )
                ax.scatter(
                    wp_data_TSNE_ROW["tsne_1"],
                    wp_data_TSNE_ROW["tsne_2"],
                    c="m",
                    marker="X",
                    s=50,
                    label="Terminal cell datapoint",
                )
                ax.scatter(
                    start_cell["tsne_1"],
                    start_cell["tsne_2"],
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
                picknm = Picture_name + "_2d_manifold.png"
                plt.savefig(f"{resultPath}/{picknm}")
                plt.close()

            """
            -----------------------------------------------------------------
            2d pseudo_time
            -----------------------------------------------------------------
            """

            fig = plt.figure()
            ax = fig.add_subplot(111)
            img = ax.scatter(
                pseudo_data["tsne_1"],
                pseudo_data["tsne_2"],
                s=5,
                marker="o",
                cmap=matplotlib.cm.plasma,
                c=c,
                label=Primary_label,
            )
            ax.scatter(
                wp_data_TSNE_ROW["tsne_1"],
                wp_data_TSNE_ROW["tsne_2"],
                s=50,
                marker="X",
                c="m",
                label="Terminal cell datapoint",
            )
            ax.scatter(
                start_cell["tsne_1"],
                start_cell["tsne_2"],
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
            picknm = Picture_name + "_2d_manifold_PSEUDO_TIME.png"
            plt.savefig(f"{resultPath}/{picknm}")
            plt.close()

        if three_dimension_manifold_plot == True:
            """
            -----------------------------------------------------------------
            3d manifold
            -----------------------------------------------------------------
            """

            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.view_init(elev=12, azim=-100)
            ax.scatter(
                wp_data_TSNE_ROW["Pseudo_Time_normal"],
                wp_data_TSNE_ROW["tsne_1"],
                wp_data_TSNE_ROW["tsne_2"],
                c="m",
                marker="X",
                s=50,
                label="Terminal cell datapoint",
            )
            ax.scatter(
                start_cell["Pseudo_Time_normal"],
                start_cell["tsne_1"],
                start_cell["tsne_2"],
                c="c",
                marker="h",
                s=50,
                label="Start cell datapoint",
            )
            temp1 = ax.scatter(
                pseudo_data["Pseudo_Time_normal"],
                pseudo_data["tsne_1"],
                pseudo_data["tsne_2"],
                c="r",
                marker="o",
                s=5,
                label=Primary_label,
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
            picknm = Picture_name + "_3d_manifold.png"
            plt.savefig(f"{resultPath}/{picknm}")
            plt.close()

        if bifurcation_plot == True:
            """
            -----------------------------------------------------------------
            Bifurcation points
            -----------------------------------------------------------------
            """
            azim = [-100, -173, -172]
            elev = [12, 12, 70]
            temp = bifurcation_data
            for a in range(3):
                fig = plt.figure()
                ax = fig.add_subplot(111, projection="3d")
                bifurcation_data = temp
                ax.view_init(elev=elev[a], azim=azim[a])
                # colors = iter(cm.rainbow(np.linspace(0.5, 1, Final_lineage_counter)))
                # colors = ['b','g', 'gold', 'sienna', 'silver']

                temp1 = ax.scatter(
                    pseudo_data["Pseudo_Time_normal"],
                    pseudo_data["tsne_1"],
                    pseudo_data["tsne_2"],
                    c="r",
                    marker="o",
                    s=5,
                    label=Primary_label,
                    alpha=1,
                )
                ax.scatter(
                    wp_data_TSNE_ROW["Pseudo_Time_normal"],
                    wp_data_TSNE_ROW["tsne_1"],
                    wp_data_TSNE_ROW["tsne_2"],
                    c="m",
                    marker="X",
                    s=50,
                    label="Terminal cell datapoint",
                )
                ax.scatter(
                    start_cell["Pseudo_Time_normal"],
                    start_cell["tsne_1"],
                    start_cell["tsne_2"],
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
                    # Data = {'G1_Pseudo_Time_normal':ORIGINAL_window_1['Pseudo_Time_normal'],'G1_tsne_1':ORIGINAL_window_1['tsne_1'],'G1_tsne_2':ORIGINAL_window_1['tsne_2'],'G2_Pseudo_Time_normal':ORIGINAL_window_2['Pseudo_Time_normal'], 'G2_tsne_1':ORIGINAL_window_2['tsne_1'],'G2_tsne_2':ORIGINAL_window_2['tsne_2']}
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
                        "G1_Pseudo_Time_normal",
                        "G1_tsne_1",
                        "G1_tsne_2",
                        "G2_Pseudo_Time_normal",
                        "G2_tsne_1",
                        "G2_tsne_2",
                    ]
                    # Data in original 3d window
                    d = {
                        "Pseudo_Time_normal": bifurcation_point[
                            "G1_Pseudo_Time_normal"
                        ],
                        "tsne_1": bifurcation_point["G1_tsne_1"],
                        "tsne_2": bifurcation_point["G1_tsne_2"],
                    }
                    current_ORIGINAL_window_G1_3d = pd.DataFrame(d)
                    current_ORIGINAL_window_G1_3d = (
                        current_ORIGINAL_window_G1_3d.dropna()
                    )  # Drop NAN rows

                    d = {
                        "Pseudo_Time_normal": bifurcation_point[
                            "G2_Pseudo_Time_normal"
                        ],
                        "tsne_1": bifurcation_point["G2_tsne_1"],
                        "tsne_2": bifurcation_point["G2_tsne_2"],
                    }
                    current_ORIGINAL_window_G2_3d = pd.DataFrame(d)
                    current_ORIGINAL_window_G2_3d = (
                        current_ORIGINAL_window_G2_3d.dropna()
                    )  # Drop NAN rows

                    bifurcation_point_mean_1 = current_ORIGINAL_window_G1_3d.head(1)
                    bifurcation_point_mean_2 = current_ORIGINAL_window_G2_3d.head(1)

                    bifurcation_point_mean_1.reset_index(drop=True, inplace=True)
                    bifurcation_point_mean_2.reset_index(drop=True, inplace=True)
                    Frames = [bifurcation_point_mean_1, bifurcation_point_mean_2]
                    new_data_FRAME = pd.concat(Frames, axis=0, sort=False)

                    test = new_data_FRAME.mean(axis=0)
                    test.columns = ["Pseudo_Time_normal", "tsne_1", "tsne_2"]
                    ax.scatter(
                        (test["Pseudo_Time_normal"]),
                        test["tsne_1"],
                        test["tsne_2"],
                        color=colors[e],
                        marker="o",
                        s=200,
                        alpha=1,
                        label="Bifurcation point: " + str(e + 1),
                    )
                    # ax.scatter( (bifurcation_point_mean_2['Pseudo_Time_normal']), bifurcation_point_mean_2['tsne_1'], bifurcation_point_mean_2['tsne_2'],color =next(colors), marker='o', s = 200, alpha=1, label='Bifurcation point')

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
                picknm = Picture_name + "_bifurcation." + str(a + 1) + ".png"
                plt.savefig(f"{resultPath}/{picknm}")
                plt.close()

        if one_lineage_plot == True:
            """
            -----------------------------------------------------------------
            1 Lineage at a time
            -----------------------------------------------------------------
            """

            # colors = iter(cm.rainbow(np.linspace(0.7, 1, Final_lineage_counter)))

            temp = Final_lineage_df

            for z in range(Final_lineage_counter):
                Lineage = temp.iloc[:, 0:3]  # Select first three columns as new input
                Lineage = Lineage.dropna()  # Drop possible NAN data

                temp.columns = range(temp.shape[1])  # Rename column names
                temp = temp.drop(
                    [0, 1, 2], axis=1
                )  # Remove new pseudo data lineage from data

                Lineage.columns = ["Pseudo_Time_normal", "tsne_1", "tsne_2"]

                # Terminal states of the lineage
                lineage_TERMINAL_STATE = Lineage[
                    Lineage["Pseudo_Time_normal"].isin(
                        wp_data_TSNE_ROW["Pseudo_Time_normal"].values
                    )
                ]

                fig = plt.figure()
                ax = fig.add_subplot(111, projection="3d")
                ax.view_init(elev=12, azim=-100)

                ax.scatter(
                    lineage_TERMINAL_STATE["Pseudo_Time_normal"],
                    lineage_TERMINAL_STATE["tsne_1"],
                    lineage_TERMINAL_STATE["tsne_2"],
                    c="m",
                    marker="X",
                    s=50,
                    label="Terminal cell datapoint",
                )
                ax.scatter(
                    start_cell["Pseudo_Time_normal"],
                    start_cell["tsne_1"],
                    start_cell["tsne_2"],
                    c="c",
                    marker="h",
                    s=50,
                    label="Start cell datapoint",
                )

                ax.scatter(
                    Lineage.Pseudo_Time_normal,
                    Lineage.tsne_1,
                    Lineage.tsne_2,
                    color=colors[z],
                    marker="o",
                    s=5,
                    label="PC-lineage-" + str(z + 1),
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
                picknm = Picture_name + "_lineage_" + str(z + 1) + ".png"
                plt.savefig(f"{resultPath}/{picknm}")
                plt.close()
        if all_lineage_plot == True:
            """
            -----------------------------------------------------------------
            All Lineages
            -----------------------------------------------------------------
            """
            # colors = iter(cm.rainbow(np.linspace(0.7, 1, Final_lineage_counter)))
            temp = Final_lineage_df
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.view_init(elev=12, azim=-100)
            for a in range(Final_lineage_counter):
                Lineage = temp.iloc[:, 0:3]  # Select first three columns as new input
                Lineage = Lineage.dropna()  # Drop possible NAN data
                temp.columns = range(temp.shape[1])  # Rename column names
                temp = temp.drop(
                    [0, 1, 2], axis=1
                )  # Remove new pseudo data lineage from data
                Lineage.columns = ["Pseudo_Time_normal", "tsne_1", "tsne_2"]
                temp1 = ax.scatter(
                    Lineage.Pseudo_Time_normal,
                    Lineage.tsne_1,
                    Lineage.tsne_2,
                    color=colors[a],
                    alpha=1,
                    marker="o",
                    s=5,
                    label="PC-lineage-" + str(a + 1),
                )

            ax.scatter(
                wp_data_TSNE_ROW["Pseudo_Time_normal"],
                wp_data_TSNE_ROW["tsne_1"],
                wp_data_TSNE_ROW["tsne_2"],
                c="m",
                marker="X",
                s=50,
                label="Terminal cell datapoint",
            )
            ax.scatter(
                start_cell["Pseudo_Time_normal"],
                start_cell["tsne_1"],
                start_cell["tsne_2"],
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
            picknm = Picture_name + "_ALL_LINEAGE.png"
            plt.savefig(f"{resultPath}/{picknm}")
            plt.close()

        if frenet_frame_plot == True:
            """
            -----------------------------------------------------------------
            Frenet frame
            -----------------------------------------------------------------
            """

            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.view_init(elev=12, azim=-100)
            for t in range(Final_lineage_counter):

                Frenet_frame_NORMAL_VECTOR = joblib.load(
                    "Frenet_frame_NORMAL_VECTOR" + str(t) + ".pkl"
                )
                Frenet_frame_MEAN = joblib.load("Frenet_frame_MEAN" + str(t) + ".pkl")
                Frenet_frame_COUNTER = joblib.load(
                    "Frenet_frame_COUNTER" + str(t) + ".pkl"
                )
                # Frenet Frame
                # colors = iter(cm.rainbow(np.linspace(0.5, 1, Final_lineage_counter)))
                for e in range(Frenet_frame_COUNTER):

                    covariance_length = (
                        Frenet_frame_NORMAL_VECTOR.iloc[:, 0:1].values
                    ).T
                    covariance_length = covariance_length[0]

                    MEAN_window_pseudo_data = (Frenet_frame_MEAN.iloc[:, 0:1].values).T
                    MEAN_window_pseudo_data = MEAN_window_pseudo_data[0]

                    if (500 * covariance_length[0]) > MEAN_window_pseudo_data[0]:
                        test_1 = covariance_length + MEAN_window_pseudo_data
                        line = Arrow3D(
                            [MEAN_window_pseudo_data[0], test_1[0]],
                            [MEAN_window_pseudo_data[1], test_1[1]],
                            [MEAN_window_pseudo_data[2], test_1[2]],
                            mutation_scale=10,
                            lw=2,
                            arrowstyle="-|>",
                            color="g",
                        )
                        ax.add_artist(line)

                    else:
                        test_1 = -covariance_length + MEAN_window_pseudo_data
                        line = Arrow3D(
                            [MEAN_window_pseudo_data[0], test_1[0]],
                            [MEAN_window_pseudo_data[1], test_1[1]],
                            [MEAN_window_pseudo_data[2], test_1[2]],
                            mutation_scale=10,
                            lw=2,
                            arrowstyle="-|>",
                            color="g",
                        )
                        ax.add_artist(line)

                    # Remove used arrows
                    Frenet_frame_NORMAL_VECTOR.columns = range(
                        Frenet_frame_NORMAL_VECTOR.shape[1]
                    )  # Rename column names
                    Frenet_frame_NORMAL_VECTOR = Frenet_frame_NORMAL_VECTOR.drop(
                        [0], axis=1
                    )  # Remove new pseudo data lineage from data

                    Frenet_frame_MEAN.columns = range(
                        Frenet_frame_MEAN.shape[1]
                    )  # Rename column names
                    Frenet_frame_MEAN = Frenet_frame_MEAN.drop(
                        [0], axis=1
                    )  # Remove new pseudo data lineage from data
            temp1 = ax.scatter(
                pseudo_data["Pseudo_Time_normal"],
                pseudo_data["tsne_1"],
                pseudo_data["tsne_2"],
                c="r",
                marker="o",
                s=10,
                alpha=1,
                label=Primary_label,
            )
            ax.scatter(
                wp_data_TSNE_ROW["Pseudo_Time_normal"],
                wp_data_TSNE_ROW["tsne_1"],
                wp_data_TSNE_ROW["tsne_2"],
                c="m",
                marker="X",
                s=50,
                label="Terminal cell datapoint",
            )
            ax.scatter(
                start_cell["Pseudo_Time_normal"],
                start_cell["tsne_1"],
                start_cell["tsne_2"],
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
            picknm = Picture_name + "_Frenet_frame.png"
            plt.savefig(f"{resultPath}/{picknm}")
            plt.close()

        if GP_with_data_plot == True:
            """
            -----------------------------------------------------------------
            Gaussian procees with data
            -----------------------------------------------------------------
            """

            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.view_init(elev=12, azim=-100)
            # colors = iter(cm.rainbow(np.linspace(0.7, 1, Final_lineage_counter)))
            temp = Final_lineage_df
            # Plot final lineages plot
            for z in range(Final_lineage_counter):
                Lineage = temp.iloc[:, 0:3]  # Select first three columns as new input
                Lineage = Lineage.dropna()  # Drop possible NAN data

                temp.columns = range(temp.shape[1])  # Rename column names
                temp = temp.drop(
                    [0, 1, 2], axis=1
                )  # Remove new pseudo data lineage from data
                Lineage.columns = ["Pseudo_Time_normal", "tsne_1", "tsne_2"]
                # Dataframe to array data
                X_train = np.array(Lineage["Pseudo_Time_normal"].values.tolist())
                Y_train = Lineage["tsne_1"].to_numpy()
                f_star_1, sigma_1, X_star = gp.Gaussian_process_algorithm(
                    X_train, Y_train
                )

                Y_train = Lineage["tsne_2"].to_numpy()
                f_star_2, sigma_2, X_star = gp.Gaussian_process_algorithm(
                    X_train, Y_train
                )
                ax.plot(
                    X_star,
                    f_star_1,
                    f_star_2,
                    color=colors[z],
                    linewidth=5,
                    label="PC-lineage-" + str(z + 1),
                )
            ax.scatter(
                wp_data_TSNE_ROW["Pseudo_Time_normal"],
                wp_data_TSNE_ROW["tsne_1"],
                wp_data_TSNE_ROW["tsne_2"],
                c="m",
                marker="X",
                s=50,
                label="Terminal cell",
            )
            ax.scatter(
                start_cell["Pseudo_Time_normal"],
                start_cell["tsne_1"],
                start_cell["tsne_2"],
                c="c",
                marker="h",
                s=50,
                label="Start cell",
            )
            temp1 = ax.scatter(
                pseudo_data["Pseudo_Time_normal"],
                pseudo_data["tsne_1"],
                pseudo_data["tsne_2"],
                c="k",
                marker="o",
                s=10,
                alpha=1,
                label=Primary_label,
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
            picknm = Picture_name + "_Gaussian_process_and_data.png"
            plt.savefig(f"{resultPath}/{picknm}")
            plt.close()

        if GP_only_plot == True:
            """
            -----------------------------------------------------------------
            Gaussian procees only
            -----------------------------------------------------------------
            """

            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.view_init(elev=12, azim=-100)
            # colors = iter(cm.rainbow(np.linspace(0.7, 1, Final_lineage_counter)))
            temp = Final_lineage_df
            # Plot final lineages plot
            for z in range(Final_lineage_counter):
                Lineage = temp.iloc[:, 0:3]  # Select first three columns as new input
                Lineage = Lineage.dropna()  # Drop possible NAN data

                temp.columns = range(temp.shape[1])  # Rename column names
                temp = temp.drop(
                    [0, 1, 2], axis=1
                )  # Remove new pseudo data lineage from data
                Lineage.columns = ["Pseudo_Time_normal", "tsne_1", "tsne_2"]
                # Dataframe to array data
                X_train = np.array(Lineage["Pseudo_Time_normal"].values.tolist())
                Y_train = Lineage["tsne_1"].to_numpy()
                f_star_1, sigma_1, X_star = gp.Gaussian_process_algorithm(
                    X_train, Y_train
                )

                Y_train = Lineage["tsne_2"].to_numpy()
                f_star_2, sigma_2, X_star = gp.Gaussian_process_algorithm(
                    X_train, Y_train
                )

                ax.plot(
                    X_star,
                    f_star_1,
                    f_star_2,
                    color=colors[z],
                    linewidth=5,
                    label="PC-lineage-" + str(z + 1),
                )
            ax.scatter(
                wp_data_TSNE_ROW["Pseudo_Time_normal"],
                wp_data_TSNE_ROW["tsne_1"],
                wp_data_TSNE_ROW["tsne_2"],
                c="m",
                marker="X",
                s=50,
                label="Terminal cell",
            )
            ax.scatter(
                start_cell["Pseudo_Time_normal"],
                start_cell["tsne_1"],
                start_cell["tsne_2"],
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
            picknm = Picture_name + "_Gaussian_process_only.png"
            plt.savefig(f"{resultPath}/{picknm}")
            plt.close()

        if GP_per_lineage_plot == True:
            """
            -----------------------------------------------------------------
            Gaussian per lineage
            -----------------------------------------------------------------
            """
            temp = Final_lineage_df
            for q in range(Final_lineage_counter):
                Lineage = temp.iloc[:, 0:3]  # Select first three columns as new input
                Lineage = Lineage.dropna()  # Drop possible NAN data

                temp.columns = range(temp.shape[1])  # Rename column names
                temp = temp.drop(
                    [0, 1, 2], axis=1
                )  # Remove new pseudo data lineage from data
                Lineage.columns = ["Pseudo_Time_normal", "tsne_1", "tsne_2"]
                # Dataframe to array data
                X_train = np.array(Lineage["Pseudo_Time_normal"].values.tolist())
                Y_train = Lineage["tsne_1"].to_numpy()
                f_star_1, sigma_1, X_star = gp.Gaussian_process_algorithm(
                    X_train, Y_train
                )

                Y_train = Lineage["tsne_2"].to_numpy()
                f_star_2, sigma_2, X_star = gp.Gaussian_process_algorithm(
                    X_train, Y_train
                )

                lineage_TERMINAL_STATE = Lineage[
                    Lineage["Pseudo_Time_normal"].isin(
                        wp_data_TSNE_ROW["Pseudo_Time_normal"].values
                    )
                ]

                fig = plt.figure()

                ax = fig.add_subplot(111, projection="3d")
                ax.view_init(elev=12, azim=-100)

                ax.plot(
                    X_star,
                    f_star_1,
                    f_star_2,
                    color=colors[q],
                    linewidth=5,
                    label="PC-lineage-" + str(q + 1),
                )
                temp1 = ax.scatter(
                    Lineage["Pseudo_Time_normal"],
                    Lineage["tsne_1"],
                    Lineage["tsne_2"],
                    c="k",
                    marker="x",
                    s=2,
                    alpha=1,
                    label=Primary_label,
                )
                ax.scatter(
                    lineage_TERMINAL_STATE["Pseudo_Time_normal"],
                    lineage_TERMINAL_STATE["tsne_1"],
                    lineage_TERMINAL_STATE["tsne_2"],
                    c="m",
                    marker="X",
                    s=50,
                    label="Terminal cell",
                )
                ax.scatter(
                    start_cell["Pseudo_Time_normal"],
                    start_cell["tsne_1"],
                    start_cell["tsne_2"],
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
                    Picture_name
                    + "_Gaussian_process_PER_lineage_"
                    + str(q + 1)
                    + ".png"
                )
                plt.savefig(f"{resultPath}/{picknm}")
                plt.close()
