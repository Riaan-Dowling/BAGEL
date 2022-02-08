import numpy as np
import pandas as pd

import math
import time


from scipy.stats import dirichlet, invwishart, multivariate_normal, norm

import os
import joblib

import data_import
import GIBBS_SAMPLER_3D  # 05/12/2020
import Frenet_frame
import split_algorithm
import lineage_parameters
import plots


np.random.seed(1)

from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def lineages_estimater():

    """
    -----------------------------------------------------------------
    Preferances flags start
    -----------------------------------------------------------------
    """

    original_manifold_plot_FLAG = True
    palantir_pseudo_time_plot_FLAG = True  # Show pseudo time plot 2d

    window_projection_before_best_plane_FLAG = False  # Original 90 degrees result
    window_projection_best_plane_2d_FLAG = False  # New projection on plane 2d
    window_projection_best_plane_3d_FLAG = False  # New projection on plane 3d

    # Window plot on manifold flags
    window_plot_FLAG = False  # Should window plot be made
    plane_normal_window_FLAG = False  # Normal vector
    plane_window_FLAG = False  # show best plane

    # Final lineage plots
    two_lineage_plot_FLAG = False
    # Plot indidicating split across entire manifold
    estimate_guassians_on_manifold_plot_FLAG = False

    lineage_plot_FLAG = True  # Final lineages plot independet
    all_lineage_plot_FLAG = True

    # Global undefined lineages
    to_be_determinded = pd.DataFrame()  # To be estimated lineages
    Final_lineage_df = pd.DataFrame()  # Final lineage data_frame
    Final_lineage_counter = 0  # Count total lineages
    all_lineages_detected = False

    """
    -----------------------------------------------------------------
    Bifurcation
    -----------------------------------------------------------------
    """
    bifurcate_once = False  # Reset flag of bifurcation after each itteration.
    bifurcation_data = pd.DataFrame()
    bifurcation_plot_FLAG = True  # Plot bifurcation points on data
    total_bifurcations = 0
    """
    -----------------------------------------------------------------
    Import data
    -----------------------------------------------------------------
    """
    pseudo_data = data_import.pseudo_data_import(
        palantir_pseudo_time_plot_FLAG, original_manifold_plot_FLAG
    )  # Data

    previouse_window_number = 0

    while all_lineages_detected == False:

        """
        -----------------------------------------------------------------
        Algortihim specefics import
        -----------------------------------------------------------------
        """
        (
            pseudo_data,
            initialize_cells_offset,
            step_size,
            window_step,
            window_size,
            itterations,
            burn_period,
        ) = lineage_parameters.parameters(pseudo_data)
        (
            total_windows,
            half_window,
            cell_total,
        ) = lineage_parameters.total_windows_estimate(
            pseudo_data, initialize_cells_offset, window_size, step_size
        )  # Total windows and if there is half a window

        """
        -----------------------------------------------------------------
        Data set for before and after split
        -----------------------------------------------------------------
        """
        column = ["G1_Pseudo_Time_normal", "G1_tsne_1", "G1_tsne_2"]
        before_split = pd.DataFrame(columns=column)
        column = ["G2_Pseudo_Time_normal", "G2_tsne_1", "G2_tsne_2"]
        after_split = pd.DataFrame(columns=column)

        """
        -----------------------------------------------------------------
        Estimate split; if three splits the split is true
        -----------------------------------------------------------------
        """
        split = False  # Flag for three splits and hence split is true
        once = False  # Flag for one split
        twice = False  # Flag for two split

        # Save data for first split window
        column = ["G1_Pseudo_Time_normal", "G1_tsne_1", "G1_tsne_2"]
        once_data = pd.DataFrame(columns=column)

        # Save data for secind split window
        column = ["G2_Pseudo_Time_normal", "G2_tsne_1", "G2_tsne_2"]
        twice_data = pd.DataFrame(columns=column)

        """
        -----------------------------------------------------------------
        lineages
        -----------------------------------------------------------------
        """

        # Place holder data
        column = ["Pseudo_Time_normal", "tsne_1", "tsne_2"]
        once_data_ORIGNAL = pd.DataFrame(columns=column)
        once_data_PROJECTED = pd.DataFrame(columns=column)
        twice_data_ORIGNAL = pd.DataFrame(columns=column)
        twice_data_PROJECTED = pd.DataFrame(columns=column)

        previouse_window_L1 = pd.DataFrame(columns=column)
        previouse_window_L2 = pd.DataFrame(columns=column)

        column = ["Pseudo_Time_normal", "tsne_1", "tsne_2"]
        Lineage_1 = pd.DataFrame(columns=column)

        column = ["Pseudo_Time_normal", "tsne_1", "tsne_2"]
        Lineage_2 = pd.DataFrame(columns=column)

        first_time_association = False  # Start association after delay

        Model_1_Lineage_1_counter = (
            0  # Count how many times model 1 is selected after split detection
        )
        Model_1_Lineage_2_counter = (
            0  # Count how many times model 1 is selected after split detection
        )

        """
        -----------------------------------------------------------------
        Window flag set
        -----------------------------------------------------------------
        """
        final_window = False  # Flag for final window
        first_window = True  # Flag for first window

        """
        -----------------------------------------------------------------
        Bifurcation
        -----------------------------------------------------------------
        """
        bifurcate_once = False  # Reset flag of bifurcation after each itteration.

        """
        -----------------------------------------------------------------
        VIDEO
        -----------------------------------------------------------------
        """
        split_VIDEO = False  # Flag for three splits and hence split is true
        once_VIDEO = False  # Flag for one split
        twice_VIDEO = False  # Flag for two split

        first_time_association_VIDEO = False  # Only do delayed split values once

        # Save data for first split window
        once_data_VIDEO_WINDOW = pd.DataFrame()
        once_data_VIDEO_NOT_WINDOW = pd.DataFrame()

        # Save data for secind split window
        twice_data_VIDEO_WINDOW = pd.DataFrame()
        twice_data_VIDEO_NOT_WINDOW = pd.DataFrame()

        previouse_window_L1_PROJECTED = pd.DataFrame()
        previouse_window_L2_PROJECTED = pd.DataFrame()

        # Point of view data
        pov_data = pseudo_data

        # Original sizes
        INITIAL_step_size = step_size  # |_______/
        INITIAL_window_size = window_size  # |______|

        # Frenet frame
        Frenet_frame_NORMAL_VECTOR = pd.DataFrame()
        Frenet_frame_MEAN = pd.DataFrame()
        Frenet_frame_COUNTER = 0

        z = 0  # Plot number counter

        # Intialize proximity test
        previous_proximity = 0.00001

        while final_window == False:  # total_windows
            z = z + 1  # Plot number counter

            """
            -----------------------------------------------------------------
            Determine adaptive window
            -----------------------------------------------------------------
            """

            # Terminal state FLAGs
            # Terminal states
            wp_data_TSNE_ROW = joblib.load("wp_data_TSNE_ROW.pkl")

            step_size = INITIAL_step_size  # |_______/
            window_size = INITIAL_window_size  # |______|

            stop = False
            terminal_state_step_SIZE = INITIAL_step_size
            terminal_state_step = INITIAL_step_size

            # Terminal state detected
            TS_detected_Next_window = False

            while stop == False:

                # Select window data
                TERMINAL_STATE_TEST = pov_data.head(step_size)
                TERMINAL_STATE_TEST_POV = pov_data[
                    ~pov_data["cell_ID_number"].isin(
                        TERMINAL_STATE_TEST["cell_ID_number"].values
                    )
                ]

                # Test if TERMINAL STATE FLAG is in selected data
                TERMINAL_STATE_TEST_POV = TERMINAL_STATE_TEST_POV.head(
                    terminal_state_step
                )
                true_false = TERMINAL_STATE_TEST_POV[
                    TERMINAL_STATE_TEST_POV["Pseudo_Time_normal"].isin(
                        wp_data_TSNE_ROW["Pseudo_Time_normal"].values
                    )
                ]

                if true_false.empty == True:
                    stop = True
                    if TS_detected_Next_window == True:
                        final_window = True

                elif true_false.empty == False:
                    if max(wp_data_TSNE_ROW["Pseudo_Time_normal"]) == max(
                        true_false["Pseudo_Time_normal"]
                    ):
                        stop = True
                        final_window = True

                        len_till_TS = len(pov_data.index)
                        step_size = len_till_TS  # len_till_TS INITIAL_step_size + terminal_state_step #|_______/
                        window_size = len_till_TS  # INITIAL_step_size + terminal_state_step #|______|

                    else:

                        TS_detected_Next_window = True

                        # Update window step size till terminal state
                        pt_max_detected = max(true_false["Pseudo_Time_normal"])
                        pt_max_detected_POSITION = pov_data.loc[
                            pov_data["Pseudo_Time_normal"] == pt_max_detected,
                            "cell_ID_number",
                        ].values

                        pt_START_POSITION_ROW = pov_data.head(1)
                        pt_START_POSITION_value = pt_START_POSITION_ROW[
                            "cell_ID_number"
                        ].values

                        difference = abs(
                            pt_max_detected_POSITION[0] - pt_START_POSITION_value
                        )

                        temp = pov_data.head(pt_max_detected_POSITION[0])
                        len_till_TS = len(temp.index)
                        terminal_state_step = (
                            terminal_state_step + terminal_state_step_SIZE
                        )

                        step_size = (
                            difference[0] + 1
                        )  # len_till_TS INITIAL_step_size + terminal_state_step #|_______/
                        window_size = (
                            difference[0] + 1
                        )  # INITIAL_step_size + terminal_state_step #|______|

                        if len(pov_data.index) == (difference[0] + 1):
                            stop = True
                            final_window = True

                            len_till_TS = len(pov_data.index)
                            step_size = len_till_TS  # len_till_TS INITIAL_step_size + terminal_state_step #|_______/
                            window_size = len_till_TS  # INITIAL_step_size + terminal_state_step #|______|

            """
            -----------------------------------------------------------------
            Prepare data for split estimation
            -----------------------------------------------------------------
            """

            # #Step window forward
            window_step = step_size + window_step

            """
            -----------------------------------------------------------------
            Apply maths
            -----------------------------------------------------------------
            """

            # Stop
            if pov_data.empty:
                break

            window_itteration = z

            # Model gaussians and Estimate split
            if split == False:
                """
                -----------------------------------------------------------------
                Estimate plane
                -----------------------------------------------------------------
                """
                (
                    estimate_original_pt,
                    estimate,
                    p1_rotate,
                    normal_vector,
                    covariance_length,
                    plane_x,
                    plane_y,
                    plane_z,
                    window_pseudo_data,
                    not_window_pseudo_data,
                    front_face,
                    back_face,
                    final_window,
                    pov_data,
                ) = Frenet_frame.pov_plane_slice(
                    pseudo_data,
                    pov_data,
                    window_size,
                    step_size,
                    total_windows,
                    final_window,
                    window_itteration,
                )

                # Create  window data dataframe
                d = {
                    "tsne_1": window_pseudo_data["tsne_1"],
                    "tsne_2": window_pseudo_data["tsne_2"],
                }
                window_2d = pd.DataFrame(d)
                d2 = {
                    "Pseudo_Time_normal": window_pseudo_data["Pseudo_Time_normal"],
                    "tsne_1": window_pseudo_data["tsne_1"],
                    "tsne_2": window_pseudo_data["tsne_2"],
                }
                window_3d = pd.DataFrame(d2)
                window_3d.reset_index(drop=True, inplace=True)
                # Reset data index
                window_2d.reset_index(drop=True, inplace=True)

                """
                -----------------------------------------------------------------
                Model selection
                -----------------------------------------------------------------
                """

                (
                    previous_proximity,
                    ran_out_of_data_split,
                    Model_1,
                    Model_2,
                    data_output_gibbs_ORIGNAL,
                    data_output_gibbs_PROJECTED,
                    M1_map_mean,
                    M1_map_cov,
                    M2_map_mean_G1,
                    M2_map_cov_G1,
                    M2_map_mean_G2,
                    M2_map_cov_G2,
                ) = GIBBS_SAMPLER_3D.gibbs_sampler_2_gaussians(
                    previous_proximity,
                    estimate_original_pt,
                    window_pseudo_data,
                    itterations,
                    burn_period,
                    window_itteration,
                    pov_data,
                )

                """
                -----------------------------------------------------------------
                Save data
                -----------------------------------------------------------------
                """

                data_Association_after_split_Flag = False

                # Estimate global split
                (
                    once,
                    twice,
                    split,
                    once_data,
                    twice_data,
                    once_data_ORIGNAL,
                    once_data_PROJECTED,
                    twice_data_ORIGNAL,
                    twice_data_PROJECTED,
                    first_time_association,
                    Lineage_1,
                    Lineage_2,
                ) = split_algorithm.estimate_guassians_on_manifold(
                    once,
                    twice,
                    Model_1,
                    Model_2,
                    once_data,
                    twice_data,
                    window_3d,
                    split,
                    final_window,
                    step_size,
                    data_output_gibbs_ORIGNAL,
                    data_output_gibbs_PROJECTED,
                    first_time_association,
                    once_data_ORIGNAL,
                    once_data_PROJECTED,
                    twice_data_ORIGNAL,
                    twice_data_PROJECTED,
                    Lineage_1,
                    Lineage_2,
                    previouse_window_L1,
                    previouse_window_L2,
                    previouse_window_L1_PROJECTED,
                    previouse_window_L2_PROJECTED,
                    Model_1_Lineage_1_counter,
                    Model_1_Lineage_2_counter,
                )

                if split == True:
                    print("Verify possible split")
                    if pov_data.empty == False:
                        (
                            _,
                            _,
                            Lineage_1_Test,
                            Lineage_2_Test,
                            _,
                            _,
                            _,
                            _,
                            _,
                            _,
                            _,
                        ) = split_algorithm.association(
                            once,
                            twice,
                            split,
                            window_3d,
                            Model_1,
                            Model_2,
                            Lineage_1,
                            Lineage_2,
                            data_output_gibbs_ORIGNAL,
                            data_output_gibbs_PROJECTED,
                            once_data_ORIGNAL,
                            once_data_PROJECTED,
                            twice_data_ORIGNAL,
                            twice_data_PROJECTED,
                            first_time_association,
                            previouse_window_L1,
                            previouse_window_L2,
                            previouse_window_L1_PROJECTED,
                            previouse_window_L2_PROJECTED,
                            Model_1_Lineage_1_counter,
                            Model_1_Lineage_2_counter,
                        )
                        (
                            Lineage_1_Test,
                            Lineage_2_Test,
                            previouse_window_L1_PROJECTED,
                            previouse_window_L2_PROJECTED,
                        ) = split_algorithm.AFTER_split_euclidean_dist_association(
                            Lineage_1, Lineage_2, pov_data
                        )

                        Lineage_1_TERMINAL_STATE = Lineage_1_Test[
                            Lineage_1_Test["Pseudo_Time_normal"].isin(
                                wp_data_TSNE_ROW["Pseudo_Time_normal"].values
                            )
                        ]
                        Lineage_2_TERMINAL_STATE = Lineage_2_Test[
                            Lineage_2_Test["Pseudo_Time_normal"].isin(
                                wp_data_TSNE_ROW["Pseudo_Time_normal"].values
                            )
                        ]
                    else:
                        (
                            _,
                            _,
                            Lineage_1_Test,
                            Lineage_2_Test,
                            _,
                            _,
                            _,
                            _,
                            _,
                            _,
                            _,
                        ) = split_algorithm.association(
                            once,
                            twice,
                            split,
                            window_3d,
                            Model_1,
                            Model_2,
                            Lineage_1,
                            Lineage_2,
                            data_output_gibbs_ORIGNAL,
                            data_output_gibbs_PROJECTED,
                            once_data_ORIGNAL,
                            once_data_PROJECTED,
                            twice_data_ORIGNAL,
                            twice_data_PROJECTED,
                            first_time_association,
                            previouse_window_L1,
                            previouse_window_L2,
                            previouse_window_L1_PROJECTED,
                            previouse_window_L2_PROJECTED,
                            Model_1_Lineage_1_counter,
                            Model_1_Lineage_2_counter,
                        )
                        Lineage_1_TERMINAL_STATE = Lineage_1_Test[
                            Lineage_1_Test["Pseudo_Time_normal"].isin(
                                wp_data_TSNE_ROW["Pseudo_Time_normal"].values
                            )
                        ]
                        Lineage_2_TERMINAL_STATE = Lineage_2_Test[
                            Lineage_2_Test["Pseudo_Time_normal"].isin(
                                wp_data_TSNE_ROW["Pseudo_Time_normal"].values
                            )
                        ]

                    if (Lineage_1_TERMINAL_STATE.empty == True) or (
                        Lineage_2_TERMINAL_STATE.empty == True
                    ):
                        # Fake lineages
                        split = False
                        Model_1 = True
                        Model_2 = False
                        print("False split")
                    else:
                        print("Split verifyed correct")

                # Asociate data to lineage
                (
                    once,
                    twice,
                    Lineage_1,
                    Lineage_2,
                    previouse_window_L1,
                    previouse_window_L2,
                    previouse_window_L1_PROJECTED,
                    previouse_window_L2_PROJECTED,
                    first_time_association,
                    Model_1_Lineage_1_counter,
                    Model_1_Lineage_2_counter,
                ) = split_algorithm.association(
                    once,
                    twice,
                    split,
                    window_3d,
                    Model_1,
                    Model_2,
                    Lineage_1,
                    Lineage_2,
                    data_output_gibbs_ORIGNAL,
                    data_output_gibbs_PROJECTED,
                    once_data_ORIGNAL,
                    once_data_PROJECTED,
                    twice_data_ORIGNAL,
                    twice_data_PROJECTED,
                    first_time_association,
                    previouse_window_L1,
                    previouse_window_L2,
                    previouse_window_L1_PROJECTED,
                    previouse_window_L2_PROJECTED,
                    Model_1_Lineage_1_counter,
                    Model_1_Lineage_2_counter,
                )

                Frenet_frame_COUNTER = Frenet_frame_COUNTER + 1

                # Create Frenet Frame

                if Frenet_frame_NORMAL_VECTOR.empty:
                    # covariance_length.reset_index(drop=True, inplace=True)
                    Frenet_frame_NORMAL_VECTOR = pd.DataFrame(covariance_length)
                    Frenet_frame_MEAN = pd.DataFrame(
                        (window_pseudo_data.iloc[:, 0:3].mean(axis=0)).values
                    )

                else:
                    # Normal vectors
                    Frenet_frame_NORMAL_VECTOR.reset_index(drop=True, inplace=True)
                    # covariance_length.reset_index(drop=True, inplace=True)
                    frames = [
                        Frenet_frame_NORMAL_VECTOR,
                        pd.DataFrame(covariance_length),
                    ]
                    Frenet_frame_NORMAL_VECTOR = pd.concat(frames, axis=1, sort=False)

                    # Mean data
                    Frenet_frame_MEAN.reset_index(drop=True, inplace=True)
                    frames = [
                        Frenet_frame_MEAN,
                        pd.DataFrame(
                            (window_pseudo_data.iloc[:, 0:3].mean(axis=0)).values
                        ),
                    ]
                    Frenet_frame_MEAN = pd.concat(frames, axis=1, sort=False)

            elif split == True:
                (
                    Lineage_1,
                    Lineage_2,
                    previouse_window_L1_PROJECTED,
                    previouse_window_L2_PROJECTED,
                ) = split_algorithm.AFTER_split_euclidean_dist_association(
                    Lineage_1, Lineage_2, pov_data
                )
                final_window = True

                # Define 'window' data for final window
                window_pseudo_data = pov_data
                not_window_pseudo_data = pseudo_data[
                    ~pseudo_data["cell_ID_number"].isin(
                        window_pseudo_data["cell_ID_number"].values
                    )
                ]

                data_Association_after_split_Flag = True

            """
            -----------------------------------------------------------------
            Display
            -----------------------------------------------------------------
            """

            # Plot VIDEO window data on manifold
            VIDEO_window_plot_FOLLOW_FLAG = True  # Plane follow
            window_number = previouse_window_number + z

            # --------------------------------------
            # ------------Split detection start-----
            # --------------------------------------
            if split == False:
                previouse_window_L1_PROJECTED = estimate_original_pt

            if (
                (Model_2 == True)
                and (once_VIDEO == False)
                and (twice_VIDEO == False)
                and (split_VIDEO == False)
            ):
                once_VIDEO = True  # First split
                once_data_VIDEO_WINDOW = window_pseudo_data
                once_data_VIDEO_NOT_WINDOW = not_window_pseudo_data

                # Data in original 3d window
                d = {
                    "Pseudo_Time_normal": once_data_PROJECTED["G1_Pseudo_Time_normal"],
                    "tsne_1": once_data_PROJECTED["G1_tsne_1"],
                    "tsne_2": once_data_PROJECTED["G1_tsne_2"],
                }
                current_ORIGINAL_window_G1_3d = pd.DataFrame(d)
                current_ORIGINAL_window_G1_3d = (
                    current_ORIGINAL_window_G1_3d.dropna()
                )  # Drop NAN rows

                d = {
                    "Pseudo_Time_normal": once_data_PROJECTED["G2_Pseudo_Time_normal"],
                    "tsne_1": once_data_PROJECTED["G2_tsne_1"],
                    "tsne_2": once_data_PROJECTED["G2_tsne_2"],
                }
                current_ORIGINAL_window_G2_3d = pd.DataFrame(d)
                current_ORIGINAL_window_G2_3d = (
                    current_ORIGINAL_window_G2_3d.dropna()
                )  # Drop NAN rows

                # Assign NEW previouse windows
                once_VIDEO_previouse_window_L1 = current_ORIGINAL_window_G1_3d
                once_VIDEO_previouse_window_L2 = current_ORIGINAL_window_G2_3d

                # Once Guassian paramets
                once_M1_map_mean = M1_map_mean
                once_M1_map_cov = M1_map_cov
                once_M2_map_mean_G1 = M2_map_mean_G1
                once_M2_map_cov_G1 = M2_map_cov_G1
                once_M2_map_mean_G2 = M2_map_mean_G2
                once_M2_map_cov_G2 = M2_map_cov_G2

                # Once model parameters
                once_Model_1 = Model_1
                once_Model_2 = Model_2

                # Window parameters
                once_window_number = window_number
                once_back_face = back_face
                once_front_face = front_face

            elif (
                (Model_2 == True)
                and (once_VIDEO == True)
                and (twice_VIDEO == False)
                and (split_VIDEO == False)
            ):
                twice_VIDEO = True  # Second split is true
                twice_data_VIDEO_WINDOW = window_pseudo_data
                twice_data_VIDEO_NOT_WINDOW = not_window_pseudo_data

                # Twice Guassian paramets
                twice_M1_map_mean = M1_map_mean
                twice_M1_map_cov = M1_map_cov
                twice_M2_map_mean_G1 = M2_map_mean_G1
                twice_M2_map_cov_G1 = M2_map_cov_G1
                twice_M2_map_mean_G2 = M2_map_mean_G2
                twice_M2_map_cov_G2 = M2_map_cov_G2

                # Window parameters
                twice_window_number = window_number
                twice_back_face = back_face
                twice_front_face = front_face

                # Once model parameters
                twice_Model_1 = Model_1
                twice_Model_2 = Model_2

                (
                    _,
                    _,
                    _,
                    _,
                    twice_VIDEO_previouse_window_L1,
                    twice_VIDEO_previouse_window_L2,
                    _,
                    _,
                ) = split_algorithm.euclidean_dist_association(
                    once_VIDEO_previouse_window_L1,
                    once_VIDEO_previouse_window_L2,
                    once_VIDEO_previouse_window_L1,
                    once_VIDEO_previouse_window_L2,
                    twice_data_ORIGNAL,
                    twice_data_PROJECTED,
                    Lineage_1,
                    Lineage_2,
                    Model_1,
                    Model_2,
                    Model_1_Lineage_1_counter,
                    Model_1_Lineage_2_counter,
                )

            elif (
                (Model_2 == True)
                and (split_VIDEO == False)
                and (once_VIDEO == True)
                and (twice_VIDEO == True)
                and (first_time_association_VIDEO == False)
            ):
                split_VIDEO = True  # Third split is true, hence save as split
                first_time_association_VIDEO = True  # Split first time association

                once_VIDEO = False
                twice_VIDEO = False

                # Once projected
                plots.VIDEO_window_plot_FOLLOW(
                    once_window_number,
                    normal_vector,
                    covariance_length,
                    once_back_face,
                    once_front_face,
                    once_data_VIDEO_WINDOW,
                    once_data_VIDEO_NOT_WINDOW,
                    p1_rotate,
                    plane_x,
                    plane_y,
                    plane_z,
                    once_M1_map_mean,
                    once_M1_map_cov,
                    once_M2_map_mean_G1,
                    once_M2_map_cov_G1,
                    once_M2_map_mean_G2,
                    once_M2_map_cov_G2,
                    Model_1,
                    Model_2,
                    split_VIDEO,
                    once_VIDEO_previouse_window_L1,
                    once_VIDEO_previouse_window_L2,
                    estimate_original_pt,
                    plane_normal_window_FLAG,
                    plane_window_FLAG,
                    VIDEO_window_plot_FOLLOW_FLAG,
                    data_Association_after_split_Flag,
                )

                # Twice projected
                plots.VIDEO_window_plot_FOLLOW(
                    twice_window_number,
                    normal_vector,
                    covariance_length,
                    twice_back_face,
                    twice_front_face,
                    twice_data_VIDEO_WINDOW,
                    twice_data_VIDEO_NOT_WINDOW,
                    p1_rotate,
                    plane_x,
                    plane_y,
                    plane_z,
                    twice_M1_map_mean,
                    twice_M1_map_cov,
                    twice_M2_map_mean_G1,
                    twice_M2_map_cov_G1,
                    twice_M2_map_mean_G2,
                    twice_M2_map_cov_G2,
                    Model_1,
                    Model_2,
                    split_VIDEO,
                    twice_VIDEO_previouse_window_L1,
                    twice_VIDEO_previouse_window_L2,
                    estimate_original_pt,
                    plane_normal_window_FLAG,
                    plane_window_FLAG,
                    VIDEO_window_plot_FOLLOW_FLAG,
                    data_Association_after_split_Flag,
                )

            # --------------------------------------
            # ------------Split detection end-------
            # --------------------------------------

            # --------------------------------------
            # ------------Plot Start----------------
            # --------------------------------------

            if split_VIDEO == False:
                if (
                    (once_VIDEO == True)
                    and (twice_VIDEO == False)
                    and (Model_1 == True)
                ):
                    once_VIDEO = False
                    twice_VIDEO = False
                    # Once

                    # Once model parameters
                    once_Model_1 = Model_1
                    once_Model_2 = Model_2

                    frames = [
                        once_VIDEO_previouse_window_L1,
                        once_VIDEO_previouse_window_L2,
                    ]
                    once_VIDEO_previouse_window_L1 = pd.concat(
                        frames, axis=0, ignore_index=False, sort=True
                    )

                    plots.VIDEO_window_plot_FOLLOW(
                        once_window_number,
                        normal_vector,
                        covariance_length,
                        once_back_face,
                        once_front_face,
                        once_data_VIDEO_WINDOW,
                        once_data_VIDEO_NOT_WINDOW,
                        p1_rotate,
                        plane_x,
                        plane_y,
                        plane_z,
                        once_M1_map_mean,
                        once_M1_map_cov,
                        once_M2_map_mean_G1,
                        once_M2_map_cov_G1,
                        once_M2_map_mean_G2,
                        once_M2_map_cov_G2,
                        Model_1,
                        Model_2,
                        split_VIDEO,
                        once_VIDEO_previouse_window_L1,
                        once_VIDEO_previouse_window_L2,
                        estimate_original_pt,
                        plane_normal_window_FLAG,
                        plane_window_FLAG,
                        VIDEO_window_plot_FOLLOW_FLAG,
                        data_Association_after_split_Flag,
                    )

                    # Current
                    plots.VIDEO_window_plot_FOLLOW(
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
                        previouse_window_L1_PROJECTED,
                        previouse_window_L2_PROJECTED,
                        estimate_original_pt,
                        plane_normal_window_FLAG,
                        plane_window_FLAG,
                        VIDEO_window_plot_FOLLOW_FLAG,
                        data_Association_after_split_Flag,
                    )

                elif (twice_VIDEO == True) and (Model_1 == True):
                    once_VIDEO = False
                    twice_VIDEO = False
                    # Once
                    # Once model parameters
                    once_Model_1 = Model_1
                    once_Model_2 = Model_2

                    frames = [
                        once_VIDEO_previouse_window_L1,
                        once_VIDEO_previouse_window_L2,
                    ]
                    once_VIDEO_previouse_window_L1 = pd.concat(
                        frames, axis=0, ignore_index=False, sort=True
                    )

                    plots.VIDEO_window_plot_FOLLOW(
                        once_window_number,
                        normal_vector,
                        covariance_length,
                        once_back_face,
                        once_front_face,
                        once_data_VIDEO_WINDOW,
                        once_data_VIDEO_NOT_WINDOW,
                        p1_rotate,
                        plane_x,
                        plane_y,
                        plane_z,
                        once_M1_map_mean,
                        once_M1_map_cov,
                        once_M2_map_mean_G1,
                        once_M2_map_cov_G1,
                        once_M2_map_mean_G2,
                        once_M2_map_cov_G2,
                        Model_1,
                        Model_2,
                        split_VIDEO,
                        once_VIDEO_previouse_window_L1,
                        once_VIDEO_previouse_window_L2,
                        estimate_original_pt,
                        plane_normal_window_FLAG,
                        plane_window_FLAG,
                        VIDEO_window_plot_FOLLOW_FLAG,
                        data_Association_after_split_Flag,
                    )

                    # Twice
                    # Once model parameters
                    twice_Model_1 = Model_1
                    twice_Model_2 = Model_2

                    frames = [
                        twice_VIDEO_previouse_window_L1,
                        twice_VIDEO_previouse_window_L2,
                    ]
                    twice_VIDEO_previouse_window_L1 = pd.concat(
                        frames, axis=0, ignore_index=False, sort=True
                    )

                    plots.VIDEO_window_plot_FOLLOW(
                        twice_window_number,
                        normal_vector,
                        covariance_length,
                        twice_back_face,
                        twice_front_face,
                        twice_data_VIDEO_WINDOW,
                        twice_data_VIDEO_NOT_WINDOW,
                        p1_rotate,
                        plane_x,
                        plane_y,
                        plane_z,
                        twice_M1_map_mean,
                        twice_M1_map_cov,
                        twice_M2_map_mean_G1,
                        twice_M2_map_cov_G1,
                        twice_M2_map_mean_G2,
                        twice_M2_map_cov_G2,
                        Model_1,
                        Model_2,
                        split_VIDEO,
                        twice_VIDEO_previouse_window_L1,
                        twice_VIDEO_previouse_window_L2,
                        estimate_original_pt,
                        plane_normal_window_FLAG,
                        plane_window_FLAG,
                        VIDEO_window_plot_FOLLOW_FLAG,
                        data_Association_after_split_Flag,
                    )

                    # Current
                    plots.VIDEO_window_plot_FOLLOW(
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
                        previouse_window_L1_PROJECTED,
                        previouse_window_L2_PROJECTED,
                        estimate_original_pt,
                        plane_normal_window_FLAG,
                        plane_window_FLAG,
                        VIDEO_window_plot_FOLLOW_FLAG,
                        data_Association_after_split_Flag,
                    )

                elif (once_VIDEO == False) and (twice_VIDEO == False):
                    plots.VIDEO_window_plot_FOLLOW(
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
                        previouse_window_L1_PROJECTED,
                        previouse_window_L2_PROJECTED,
                        estimate_original_pt,
                        plane_normal_window_FLAG,
                        plane_window_FLAG,
                        VIDEO_window_plot_FOLLOW_FLAG,
                        data_Association_after_split_Flag,
                    )

            elif (split_VIDEO == True) and (first_time_association_VIDEO == True):
                plots.VIDEO_window_plot_FOLLOW(
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
                    previouse_window_L1_PROJECTED,
                    previouse_window_L2_PROJECTED,
                    estimate_original_pt,
                    plane_normal_window_FLAG,
                    plane_window_FLAG,
                    VIDEO_window_plot_FOLLOW_FLAG,
                    data_Association_after_split_Flag,
                )

            # --------------------------------------
            # ------------Plot end------------------
            # --------------------------------------

            # --------------------------------------
            # ------------Plot Anomaly (not three in a row but data end)------------------
            # --------------------------------------

            if (
                final_window == True
            ):  # Plot split if 111 is not acheived before end of data
                if (once_VIDEO == True) and (twice_VIDEO == False):
                    # Determine if there is till two terminal states
                    if pov_data.empty == False:  # There is still pov data left
                        POV_terminal_State = pov_data[
                            pov_data["Pseudo_Time_normal"].isin(
                                wp_data_TSNE_ROW["Pseudo_Time_normal"].values
                            )
                        ]
                        if (
                            POV_terminal_State.empty == False
                        ):  # There is a terminal state in the POV data
                            # Determine if once data has a terminal state
                            once_data_terminal_State = once_data[
                                once_data["Pseudo_Time_normal"].isin(
                                    wp_data_TSNE_ROW["Pseudo_Time_normal"].values
                                )
                            ]
                            if once_data_terminal_State.empty == False:
                                # Once model parameters
                                split = True
                                split_VIDEO = True
                            else:
                                # Once model parameters
                                split = False
                                split_VIDEO = False
                                Model_1 = True
                                Model_2 = False

                        else:
                            split = False
                            split_VIDEO = False
                            Model_1 = True
                            Model_2 = False

                    elif pov_data.empty == True:  # There is NO pov data left
                        once_data_terminal_State = once_data[
                            once_data["Pseudo_Time_normal"].isin(
                                wp_data_TSNE_ROW["Pseudo_Time_normal"].values
                            )
                        ]
                        if len(once_data_terminal_State.index) >= 2:
                            # Once model parameters
                            split = True
                            split_VIDEO = True
                        else:
                            # Once model parameters
                            split = False
                            split_VIDEO = False
                            Model_1 = True
                            Model_2 = False

                    once_VIDEO = False
                    twice_VIDEO = False
                    plots.VIDEO_window_plot_FOLLOW(
                        once_window_number,
                        normal_vector,
                        covariance_length,
                        once_back_face,
                        once_front_face,
                        once_data_VIDEO_WINDOW,
                        once_data_VIDEO_NOT_WINDOW,
                        p1_rotate,
                        plane_x,
                        plane_y,
                        plane_z,
                        once_M1_map_mean,
                        once_M1_map_cov,
                        once_M2_map_mean_G1,
                        once_M2_map_cov_G1,
                        once_M2_map_mean_G2,
                        once_M2_map_cov_G2,
                        Model_1,
                        Model_2,
                        split_VIDEO,
                        once_VIDEO_previouse_window_L1,
                        once_VIDEO_previouse_window_L2,
                        estimate_original_pt,
                        plane_normal_window_FLAG,
                        plane_window_FLAG,
                        VIDEO_window_plot_FOLLOW_FLAG,
                        data_Association_after_split_Flag,
                    )

                    # plots.VIDEO_window_plot_FOLLOW(twice_window_number,normal_vector, covariance_length, twice_back_face,twice_front_face,twice_data_VIDEO_WINDOW, twice_data_VIDEO_NOT_WINDOW, p1_rotate, plane_x, plane_y, plane_z,twice_M1_map_mean, twice_M1_map_cov, twice_M2_map_mean_G1, twice_M2_map_cov_G1, twice_M2_map_mean_G2, twice_M2_map_cov_G2, Model_1, Model_2, split_VIDEO,twice_VIDEO_previouse_window_L1, twice_VIDEO_previouse_window_L2,estimate_original_pt, plane_normal_window_FLAG,plane_window_FLAG, VIDEO_window_plot_FOLLOW_FLAG, data_Association_after_split_Flag)
                    # Assign broken lineages
                    first_time_association = True
                    (
                        once,
                        twice,
                        Lineage_1,
                        Lineage_2,
                        previouse_window_L1,
                        previouse_window_L2,
                        previouse_window_L1_PROJECTED,
                        previouse_window_L2_PROJECTED,
                        first_time_association,
                        Model_1_Lineage_1_counter,
                        Model_1_Lineage_2_counter,
                    ) = split_algorithm.association(
                        once,
                        twice,
                        split,
                        before_split,
                        Model_1,
                        Model_2,
                        Lineage_1,
                        Lineage_2,
                        data_output_gibbs_ORIGNAL,
                        data_output_gibbs_PROJECTED,
                        once_data_ORIGNAL,
                        once_data_PROJECTED,
                        twice_data_ORIGNAL,
                        twice_data_PROJECTED,
                        first_time_association,
                        previouse_window_L1,
                        previouse_window_L2,
                        previouse_window_L1_PROJECTED,
                        previouse_window_L2_PROJECTED,
                        Model_1_Lineage_1_counter,
                        Model_1_Lineage_2_counter,
                    )

                    if pov_data.empty == False:
                        (
                            Lineage_1,
                            Lineage_2,
                            previouse_window_L1_PROJECTED,
                            previouse_window_L2_PROJECTED,
                        ) = split_algorithm.AFTER_split_euclidean_dist_association(
                            Lineage_1, Lineage_2, pov_data
                        )
                        final_window = True
                        # Define 'window' data for final window
                        window_pseudo_data = pov_data
                        not_window_pseudo_data = pseudo_data[
                            ~pseudo_data["cell_ID_number"].isin(
                                window_pseudo_data["cell_ID_number"].values
                            )
                        ]

                        plots.VIDEO_window_plot_FOLLOW(
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
                            previouse_window_L1_PROJECTED,
                            previouse_window_L2_PROJECTED,
                            estimate_original_pt,
                            plane_normal_window_FLAG,
                            plane_window_FLAG,
                            VIDEO_window_plot_FOLLOW_FLAG,
                            data_Association_after_split_Flag,
                        )

                elif twice_VIDEO == True:

                    # Determine if there is till two terminal states
                    if pov_data.empty == False:  # There is still pov data left
                        POV_terminal_State = pov_data[
                            pov_data["Pseudo_Time_normal"].isin(
                                wp_data_TSNE_ROW["Pseudo_Time_normal"].values
                            )
                        ]
                        if (
                            POV_terminal_State.empty == False
                        ):  # There is a terminal state in the POV data
                            # Determine if once data has a terminal state
                            once_data_terminal_State = once_data[
                                once_data["Pseudo_Time_normal"].isin(
                                    wp_data_TSNE_ROW["Pseudo_Time_normal"].values
                                )
                            ]
                            twice_data_terminal_State = twice_data[
                                twice_data["Pseudo_Time_normal"].isin(
                                    wp_data_TSNE_ROW["Pseudo_Time_normal"].values
                                )
                            ]

                            if (once_data_terminal_State.empty == False) or (
                                twice_data_terminal_State.empty == False
                            ):
                                # Once model parameters
                                split = True
                                split_VIDEO = True
                            else:
                                # Once model parameters
                                split = False
                                split_VIDEO = False
                                Model_1 = True
                                Model_2 = False

                        else:
                            split = False
                            split_VIDEO = False
                            Model_1 = True
                            Model_2 = False

                    elif pov_data.empty == True:  # There is NO pov data left
                        once_data_terminal_State = once_data[
                            once_data["Pseudo_Time_normal"].isin(
                                wp_data_TSNE_ROW["Pseudo_Time_normal"].values
                            )
                        ]
                        twice_data_terminal_State = twice_data[
                            twice_data["Pseudo_Time_normal"].isin(
                                wp_data_TSNE_ROW["Pseudo_Time_normal"].values
                            )
                        ]

                        if len(once_data_terminal_State.index) >= 2:
                            # Once model parameters
                            split = True
                            split_VIDEO = True
                        elif len(twice_data_terminal_State.index) >= 2:
                            # Once model parameters
                            split = True
                            split_VIDEO = True
                        elif (once_data_terminal_State.empty == False) and (
                            twice_data_terminal_State.empty == False
                        ):
                            # Once model parameters
                            split = True
                            split_VIDEO = True
                        else:
                            # Once model parameters
                            split = False
                            split_VIDEO = False
                            Model_1 = True
                            Model_2 = False

                    once_VIDEO = False
                    twice_VIDEO = False
                    # Once
                    # Once model parameters
                    # once_Model_1 = False
                    # once_Model_2 = True
                    plots.VIDEO_window_plot_FOLLOW(
                        once_window_number,
                        normal_vector,
                        covariance_length,
                        once_back_face,
                        once_front_face,
                        once_data_VIDEO_WINDOW,
                        once_data_VIDEO_NOT_WINDOW,
                        p1_rotate,
                        plane_x,
                        plane_y,
                        plane_z,
                        once_M1_map_mean,
                        once_M1_map_cov,
                        once_M2_map_mean_G1,
                        once_M2_map_cov_G1,
                        once_M2_map_mean_G2,
                        once_M2_map_cov_G2,
                        Model_1,
                        Model_2,
                        split_VIDEO,
                        once_VIDEO_previouse_window_L1,
                        once_VIDEO_previouse_window_L2,
                        estimate_original_pt,
                        plane_normal_window_FLAG,
                        plane_window_FLAG,
                        VIDEO_window_plot_FOLLOW_FLAG,
                        data_Association_after_split_Flag,
                    )

                    # Twice
                    # Twice model parameters
                    # twice_Model_1 = False
                    # twice_Model_2 = True
                    plots.VIDEO_window_plot_FOLLOW(
                        twice_window_number,
                        normal_vector,
                        covariance_length,
                        twice_back_face,
                        twice_front_face,
                        twice_data_VIDEO_WINDOW,
                        twice_data_VIDEO_NOT_WINDOW,
                        p1_rotate,
                        plane_x,
                        plane_y,
                        plane_z,
                        twice_M1_map_mean,
                        twice_M1_map_cov,
                        twice_M2_map_mean_G1,
                        twice_M2_map_cov_G1,
                        twice_M2_map_mean_G2,
                        twice_M2_map_cov_G2,
                        Model_1,
                        Model_2,
                        split_VIDEO,
                        twice_VIDEO_previouse_window_L1,
                        twice_VIDEO_previouse_window_L2,
                        estimate_original_pt,
                        plane_normal_window_FLAG,
                        plane_window_FLAG,
                        VIDEO_window_plot_FOLLOW_FLAG,
                        data_Association_after_split_Flag,
                    )

                    # Assign broken lineages
                    first_time_association = True
                    (
                        once,
                        twice,
                        Lineage_1,
                        Lineage_2,
                        previouse_window_L1,
                        previouse_window_L2,
                        previouse_window_L1_PROJECTED,
                        previouse_window_L2_PROJECTED,
                        first_time_association,
                        Model_1_Lineage_1_counter,
                        Model_1_Lineage_2_counter,
                    ) = split_algorithm.association(
                        once,
                        twice,
                        split,
                        before_split,
                        Model_1,
                        Model_2,
                        Lineage_1,
                        Lineage_2,
                        data_output_gibbs_ORIGNAL,
                        data_output_gibbs_PROJECTED,
                        once_data_ORIGNAL,
                        once_data_PROJECTED,
                        twice_data_ORIGNAL,
                        twice_data_PROJECTED,
                        first_time_association,
                        previouse_window_L1,
                        previouse_window_L2,
                        previouse_window_L1_PROJECTED,
                        previouse_window_L2_PROJECTED,
                        Model_1_Lineage_1_counter,
                        Model_1_Lineage_2_counter,
                    )

                    if pov_data.empty == False:
                        (
                            Lineage_1,
                            Lineage_2,
                            previouse_window_L1_PROJECTED,
                            previouse_window_L2_PROJECTED,
                        ) = split_algorithm.AFTER_split_euclidean_dist_association(
                            Lineage_1, Lineage_2, pov_data
                        )
                        final_window = True
                        # Define 'window' data for final window
                        window_pseudo_data = pov_data
                        not_window_pseudo_data = pseudo_data[
                            ~pseudo_data["cell_ID_number"].isin(
                                window_pseudo_data["cell_ID_number"].values
                            )
                        ]

                        plots.VIDEO_window_plot_FOLLOW(
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
                            previouse_window_L1_PROJECTED,
                            previouse_window_L2_PROJECTED,
                            estimate_original_pt,
                            plane_normal_window_FLAG,
                            plane_window_FLAG,
                            VIDEO_window_plot_FOLLOW_FLAG,
                            data_Association_after_split_Flag,
                        )

            # --------------------------------------
            # ------------Plot Anomaly END (not three in a row but data end)------------------
            # --------------------------------------

        previouse_window_number = window_number  # Adjust labeling for VIDEO output

        # Remove any duplicate assignments due to window and stepsize mismatch
        Lineage_1 = Lineage_1.drop_duplicates()
        Lineage_2 = Lineage_2.drop_duplicates()
        # Drop if fake detection

        before_split = before_split.drop_duplicates()
        after_split = after_split.drop_duplicates()

        # Bifurcation
        (
            bifurcation_data,
            bifurcate_once,
            total_bifurcations,
        ) = split_algorithm.bifurcation_points(
            split,
            bifurcation_data,
            bifurcate_once,
            total_bifurcations,
            once_data_ORIGNAL,
        )

        if split == True:
            # Plot indidicating split across entire manifold

            # plots.estimate_guassians_on_manifold_plot(before_split,after_split, estimate_guassians_on_manifold_plot_FLAG)
            # plots.two_lineage_plot(Lineage_1, Lineage_2, two_lineage_plot_FLAG)

            if to_be_determinded.empty:
                Lineage_1.reset_index(drop=True, inplace=True)
                Lineage_2.reset_index(drop=True, inplace=True)
                to_be_determinded.reset_index(drop=True, inplace=True)
                frames = [Lineage_1, Lineage_2]
                to_be_determinded = pd.concat(
                    frames, axis=1, sort=False
                )  # Append new lineage data
            else:
                Lineage_1.reset_index(drop=True, inplace=True)
                Lineage_2.reset_index(drop=True, inplace=True)
                to_be_determinded.reset_index(drop=True, inplace=True)
                frames = [to_be_determinded, Lineage_1, Lineage_2]
                to_be_determinded = pd.concat(
                    frames, axis=1, sort=False
                )  # Append new lineage data

            pseudo_data = to_be_determinded.iloc[
                :, 0:3
            ]  # to_be_determinded[to_be_determinded.columns[0:3]] #Select first three columns as new input
            pseudo_data = pseudo_data.dropna()  # Drop possible NAN data
            pseudo_data.columns = ["Pseudo_Time_normal", "tsne_1", "tsne_2"]

            # Sort data
            pseudo_data = pseudo_data.sort_values("Pseudo_Time_normal")
            # Provide each cell with a number
            pt_samples = len(pseudo_data["tsne_1"])
            cell_ID_number = range(pt_samples)
            # Number of cell
            pseudo_data["cell_ID_number"] = cell_ID_number

            to_be_determinded.columns = range(
                to_be_determinded.shape[1]
            )  # Rename column names
            to_be_determinded = to_be_determinded.drop(
                [0, 1, 2], axis=1
            )  # Remove new pseudo data lineage from data

        else:

            # Dump frenet frame data
            joblib.dump(
                Frenet_frame_NORMAL_VECTOR,
                "Frenet_frame_NORMAL_VECTOR" + str(Final_lineage_counter) + ".pkl",
                compress=3,
            )
            joblib.dump(
                Frenet_frame_MEAN,
                "Frenet_frame_MEAN" + str(Final_lineage_counter) + ".pkl",
                compress=3,
            )
            joblib.dump(
                Frenet_frame_COUNTER,
                "Frenet_frame_COUNTER" + str(Final_lineage_counter) + ".pkl",
                compress=3,
            )

            # Lineages
            if Final_lineage_df.empty:
                Final_lineage_df.reset_index(drop=True, inplace=True)
                pseudo_data.reset_index(drop=True, inplace=True)
                Final_lineage_df = pseudo_data.iloc[:, 0:3]

            else:
                Final_lineage_df.reset_index(drop=True, inplace=True)
                pseudo_data.reset_index(drop=True, inplace=True)
                frames = [Final_lineage_df, pseudo_data.iloc[:, 0:3]]
                Final_lineage_df = pd.concat(
                    frames, axis=1, sort=False
                )  # Append new lineage data

            Final_lineage_counter = Final_lineage_counter + 1

            if to_be_determinded.empty:
                print("All lineages detected")
                all_lineages_detected = True
                break
            else:
                pseudo_data = to_be_determinded.iloc[
                    :, 0:3
                ]  # to_be_determinded[to_be_determinded.columns[0:3]] #Select first three columns as new input
                pseudo_data = pseudo_data.dropna()  # Drop possible NAN data
                pseudo_data.columns = ["Pseudo_Time_normal", "tsne_1", "tsne_2"]

                # Sort data
                pseudo_data = pseudo_data.sort_values("Pseudo_Time_normal")
                # Provide each cell with a number
                pt_samples = len(pseudo_data["tsne_1"])
                cell_ID_number = range(pt_samples)
                # Number of cell
                pseudo_data["cell_ID_number"] = cell_ID_number

                to_be_determinded.columns = range(
                    to_be_determinded.shape[1]
                )  # Rename column names
                to_be_determinded = to_be_determinded.drop(
                    [0, 1, 2], axis=1
                )  # Remove new pseudo data lineage from data

    # Dump pkl file
    joblib.dump(Final_lineage_df, "Final_lineage_df.pkl", compress=3)
    joblib.dump(Final_lineage_counter, "Final_lineage_counter.pkl", compress=3)
    joblib.dump(bifurcation_data, "bifurcation_data.pkl", compress=3)
    joblib.dump(total_bifurcations, "total_bifurcations.pkl", compress=3)
    joblib.dump(previouse_window_number, "previouse_window_number.pkl", compress=3)

    """
    plots.bifurcation_plot(Final_lineage_df, Final_lineage_counter, bifurcation_data,total_bifurcations, bifurcation_plot_FLAG)


    palantir_pseudo_time_plot_FLAG = False
    original_manifold_plot_FLAG = False
    pseudo_data = data_import.pseudo_data_import(palantir_pseudo_time_plot_FLAG, original_manifold_plot_FLAG)#Data
    
    matplotlib.use('TkAgg')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=15, azim=-45)
    for t in range (Final_lineage_counter):

        Frenet_frame_NORMAL_VECTOR  = joblib.load("Frenet_frame_NORMAL_VECTOR" + str(t) + ".pkl")
        Frenet_frame_MEAN  = joblib.load("Frenet_frame_MEAN" + str(t) + ".pkl")
        Frenet_frame_COUNTER  = joblib.load("Frenet_frame_COUNTER" + str(t) + ".pkl")
        #Frenet Frame
        for e in range (Frenet_frame_COUNTER):

            covariance_length = (Frenet_frame_NORMAL_VECTOR.iloc[:, 0:1].values).T
            covariance_length = covariance_length[0]

            MEAN_window_pseudo_data = (Frenet_frame_MEAN.iloc[:, 0:1].values).T
            MEAN_window_pseudo_data = MEAN_window_pseudo_data[0]

            if (300*covariance_length[0])> MEAN_window_pseudo_data[0]:
                test_1 = v  + MEAN_window_pseudo_data
                line = Arrow3D([MEAN_window_pseudo_data[0], test_1[0]], [MEAN_window_pseudo_data[1], test_1[1]], 
                    [MEAN_window_pseudo_data[2], test_1[2]], mutation_scale=30, 
                    lw=5, arrowstyle="-|>", color="r")
                ax.add_artist(line)

            else:
                test_1 = -v  + MEAN_window_pseudo_data
                line = Arrow3D([MEAN_window_pseudo_data[0], test_1[0]], [MEAN_window_pseudo_data[1], test_1[1]], 
                    [MEAN_window_pseudo_data[2], test_1[2]], mutation_scale=30, 
                    lw=5, arrowstyle="-|>", color="r")
                ax.add_artist(line)

            #Remove used arrows
            Frenet_frame_NORMAL_VECTOR.columns = range(Frenet_frame_NORMAL_VECTOR.shape[1])#Rename column names
            Frenet_frame_NORMAL_VECTOR = Frenet_frame_NORMAL_VECTOR.drop([0], axis=1)#Remove new pseudo data lineage from data

            Frenet_frame_MEAN.columns = range(Frenet_frame_MEAN.shape[1])#Rename column names
            Frenet_frame_MEAN = Frenet_frame_MEAN.drop([0], axis=1)#Remove new pseudo data lineage from data
    ax.scatter( pseudo_data['Pseudo_Time_normal'], pseudo_data['tsne_1'], pseudo_data['tsne_2'], c='k', marker='o', s = 10, alpha = 0.05)
    ax.set_zlabel('t-SNE y')
    ax.set_ylabel('t-SNE x')
    ax.set_xlabel('Pseudo time')
    plt.legend()
    plt.show()



    #Plot final lineages plot
    temp = Final_lineage_df
    for i in range(1): #Show these plots 10 times
        z = 0
        Final_lineage_df = temp
        for z in range(Final_lineage_counter):
            Lineage = Final_lineage_df.iloc[:, 0:3]#Select first three columns as new input
            Lineage = Lineage.dropna() #Drop possible NAN data


            Final_lineage_df.columns = range(Final_lineage_df.shape[1])#Rename column names
            Final_lineage_df = Final_lineage_df.drop([0,1,2], axis=1)#Remove new pseudo data lineage from data 


            Lineage.columns = [ 'Pseudo_Time_normal','tsne_1','tsne_2']

            plots.lineage_plot(Lineage, lineage_plot_FLAG)

        Final_lineage_df = temp
        plots.all_lineage_plot(Final_lineage_df,Final_lineage_counter, all_lineage_plot_FLAG)
    
    return Final_lineage_df, Final_lineage_counter

    # print('Stop')
    """
