"""
This is BAGEL
"""

import os
import shutil
from sklearn import preprocessing
import joblib
import pandas as pd

import library.phenotypic_manifold_script as pheno_sup
import library.frenet_frame as frenet_frame_sup
import library.bayesian_model_selection as bays_sup
import library.majority_vote as majority_sup
import library.plot_results as plot_sup


class BAGEL(object):
    """
    BAGEL
    """

    def __init__(
        self,
        bagel_config,
        load_old_manifold,
        results_only,
        main_data_file,
        secondary_data_file,
        early_cell,
        output_version_no,
    ):
        self.bagel_config = bagel_config
        self.load_old_manifold = load_old_manifold
        self.results_only = results_only
        self.main_data_file = main_data_file
        self.secondary_data_file = secondary_data_file
        self.early_cell = early_cell

        self.output_version_no = output_version_no
        self.total_datasets = 1

        # init temp data dataframes
        self.main_df = pd.DataFrame()
        self.secondary_df = pd.DataFrame()

        # parent_dir
        self.parent_dir = os.path.dirname(os.path.realpath(__file__)).split("/library")[
            0
        ]
        self.result_folder = os.path.join(
            self.parent_dir, f"run_{self.output_version_no}_result_folder"
        )

        # Init phenotypic manifold variables
        self.phenotypic_manifold_pca_projections = ""
        self.pseudo_time = ""
        self.waypoints = ""
        self.wp_data = ""
        self.constructed_markov_chain = ""
        self.terminal_states = ""
        self.excluded_boundaries = ""

        # dataset
        self.bagel_loop_data_terminal_state = None
        self.bagel_loop_data = None
        self.bagel_loop_data_columns = []
        self.window_removed_bagel_loop_data = None

    def create_processing_dir(self):
        """
        create_processing_dir
        """

        if self.load_old_manifold is False:
            # Result folder
            if not os.path.exists(self.result_folder):
                os.makedirs(self.result_folder)

            # Save early cell
            joblib.dump(
                self.early_cell, f"{self.result_folder}/early_cell.pkl", compress=3
            )

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

    def read_from_csv(self, csv_path, delimiter=","):
        """
        read_from_csv
        """

        if os.path.exists(f"{self.parent_dir}/{csv_path}"):
            df_in = pd.read_csv(
                f"{self.parent_dir}/{csv_path}", sep=delimiter, index_col=[0]
            )  # obtain data frame
            clean_df = self.palantir_clean_up_cell_data_files(df_in)
        else:
            print(
                "The input data set does not exist. Please ensure that the correct file name is specified (Hint: Confirm spelling)."
            )
            os._exit(1)
        return clean_df

    def load_datasets(self):
        """
        load_datasets
        """

        if self.secondary_data_file != "":
            if self.main_data_file != self.secondary_data_file:
                print("Input: Two datasets.")
                self.total_datasets = 2
                self.main_df = self.read_from_csv(self.main_data_file)
                self.secondary_df = self.read_from_csv(self.secondary_data_file)

        else:
            print("Input: One dataset.")
            # Import MAIN data set
            self.main_df = self.read_from_csv(self.main_data_file)

    def load_phenotypic_manifold(self):
        """
        load_phenotypic_manifold
        """
        phenotypic_manifold_object = pheno_sup.PhenotypicManifold(
            self.total_datasets,
            self.main_df,
            self.secondary_df,
            self.result_folder,
            self.early_cell,
            self.bagel_config,
            self.load_old_manifold,
        )

        (
            self.phenotypic_manifold_pca_projections,
            self.pseudo_time,
            self.waypoints,
            self.wp_data,
            self.constructed_markov_chain,
            self.terminal_states,
            self.excluded_boundaries,
        ) = phenotypic_manifold_object.create_phenotypic_manifold()

    def create_bagel_loop_data(self):
        """
        create_bagel_loop_data
        """
        # Scale pseudo_time between [0,1]
        min_max_scaler = preprocessing.MinMaxScaler()  # [0,1]
        pseudo_time = min_max_scaler.fit_transform(
            self.pseudo_time.values.reshape(-1, 1)
        )
        pseudo_time = pseudo_time.tolist()
        pseudo_time = [j for i in pseudo_time for j in i]

        # Scale PCA projections
        abs_scaler = preprocessing.MaxAbsScaler()
        pca_1 = abs_scaler.fit_transform(
            self.phenotypic_manifold_pca_projections["x"].values.reshape(-1, 1)
        )
        pca_2 = abs_scaler.fit_transform(
            self.phenotypic_manifold_pca_projections["y"].values.reshape(-1, 1)
        )

        pca_1 = pca_1.tolist()
        pca_1 = [j for i in pca_1 for j in i]

        pca_2 = pca_2.tolist()
        pca_2 = [j for i in pca_2 for j in i]

        # Create bagel loop data
        bagel_loop_data = pd.DataFrame(
            {"pseudo_time_normal": pseudo_time, "pca_1": pca_1, "pca_2": pca_2},
            index=self.phenotypic_manifold_pca_projections.index,
        )
        bagel_loop_data = bagel_loop_data.sort_values("pseudo_time_normal")
        self.bagel_loop_data_columns = bagel_loop_data.columns.tolist()

        joblib.dump(
            bagel_loop_data,
            f"{self.result_folder}/bagel_loop_data.pkl",
            compress=3,
        )

        # Link terminal state to normalized manifold
        bagel_loop_data_terminal_state = bagel_loop_data.loc[self.terminal_states]
        # bagel_loop_data_terminal_state = self.bagel_loop_data_terminal_state
        joblib.dump(
            bagel_loop_data_terminal_state,
            f"{self.result_folder}/bagel_loop_data_terminal_state.pkl",
            compress=3,
        )

        # Add cell id number
        cell_id_number = range(len(self.phenotypic_manifold_pca_projections["x"]))
        bagel_loop_data["cell_id_number"] = cell_id_number

        self.bagel_loop_data = bagel_loop_data
        self.bagel_loop_data_terminal_state = bagel_loop_data_terminal_state

    def determine_total_bagel_loop_windows(
        self,
        initialize_cells_offset,
        window_interval_actual_samples,
        window_interval_extra_samples,
    ):
        """
        determine_total_bagel_loop_windows
        """
        # Estimate total windows
        cell_total = len(self.bagel_loop_data["pseudo_time_normal"])

        total_windows = (
            cell_total - initialize_cells_offset - window_interval_extra_samples
        ) / window_interval_actual_samples
        total_windows_trunc = (total_windows) % 1
        half_window = None
        if total_windows_trunc > 0:
            total_windows = (
                int(total_windows) + 1
            )  # Truncated data and leava as is because for loop starts at 0
            half_window = True
        else:
            total_windows = int(
                total_windows
            )  # Truncated data and remove one itteration because for loop starts a 0
            half_window = False

        return total_windows, half_window, cell_total

    def determine_window_size_with_respect_to_ts(
        self,
        default_window_interval_extra_samples,
        default_window_interval_actual_samples,
        is_final_window,
    ):
        """
        determine_window_size_with_respect_to_ts
        """
        # Initialze windows size to default
        window_interval_extra_samples = (
            default_window_interval_extra_samples  # |_______/
        )
        window_interval_actual_samples = (
            default_window_interval_actual_samples  # |______|
        )

        # Assume terminal state window size is equal to default_window_size
        terminal_state_window_interval_extra_samples = (
            default_window_interval_extra_samples
        )
        terminal_state_step = default_window_interval_extra_samples

        ts_detected_next_window = False

        stop_loop = False
        while stop_loop is False:

            # Select window data
            terminal_state_test = self.window_removed_bagel_loop_data.head(
                window_interval_extra_samples
            )
            terminal_state_test_df_all = self.window_removed_bagel_loop_data[
                ~self.window_removed_bagel_loop_data["cell_id_number"].isin(
                    terminal_state_test["cell_id_number"].values
                )
            ]

            # Test if TERMINAL STATE FLAG is in selected data
            terminal_state_test_df_all = terminal_state_test_df_all.head(
                terminal_state_step
            )
            terminal_state_test_df_ts = terminal_state_test_df_all[
                terminal_state_test_df_all["pseudo_time_normal"].isin(
                    self.bagel_loop_data_terminal_state["pseudo_time_normal"].values
                )
            ]

            if terminal_state_test_df_ts.empty is True:
                stop_loop = True
                if ts_detected_next_window is True:
                    is_final_window = True

            elif terminal_state_test_df_ts.empty is False:
                if max(
                    self.bagel_loop_data_terminal_state["pseudo_time_normal"]
                ) == max(terminal_state_test_df_ts["pseudo_time_normal"]):
                    stop_loop = True
                    is_final_window = True

                    cells_till_ts = len(self.window_removed_bagel_loop_data.index)
                    window_interval_extra_samples = cells_till_ts  # cells_till_ts default_window_interval_extra_samples + terminal_state_step #|_______/
                    default_window_interval_actual_samples = cells_till_ts  # default_window_interval_extra_samples + terminal_state_step #|______|

                else:
                    # TODO needs refactoring
                    ts_detected_next_window = True

                    # Update window step size till terminal state
                    pt_max_detected = max(
                        terminal_state_test_df_ts["pseudo_time_normal"]
                    )
                    pt_max_detected_position = self.window_removed_bagel_loop_data.loc[
                        self.window_removed_bagel_loop_data["pseudo_time_normal"]
                        == pt_max_detected,
                        "cell_id_number",
                    ].values

                    pt_start_position_ROW = self.window_removed_bagel_loop_data.head(1)
                    pt_start_position_value = pt_start_position_ROW[
                        "cell_id_number"
                    ].values

                    difference = abs(
                        pt_max_detected_position[0] - pt_start_position_value
                    )

                    temp = self.window_removed_bagel_loop_data.head(
                        pt_max_detected_position[0]
                    )
                    cells_till_ts = len(temp.index)
                    terminal_state_step = (
                        terminal_state_step
                        + terminal_state_window_interval_extra_samples
                    )

                    window_interval_extra_samples = (
                        difference[0] + 1
                    )  # cells_till_ts default_window_interval_extra_samples + terminal_state_step #|_______/
                    window_interval_actual_samples = (
                        difference[0] + 1
                    )  # default_window_interval_extra_samples + terminal_state_step #|______|

                    if len(self.window_removed_bagel_loop_data.index) == (
                        difference[0] + 1
                    ):
                        stop_loop = True
                        is_final_window = True

                        cells_till_ts = len(self.window_removed_bagel_loop_data.index)
                        window_interval_extra_samples = cells_till_ts  # cells_till_ts default_window_interval_extra_samples + terminal_state_step #|_______/
                        window_interval_actual_samples = cells_till_ts  # default_window_interval_extra_samples + terminal_state_step #|______|

        return (
            is_final_window,
            window_interval_extra_samples,
            window_interval_actual_samples,
        )

    def bagel_loop(self):
        """
        bagel_loop
        """
        # """
        # -----------------------------------------------------------------
        # Preferances flags start
        # -----------------------------------------------------------------
        # """

        if self.bagel_config["bagel_loop_config"]["load_old_bagel_loop"] is False:
            # TODO check flags
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
            final_lineage_df = pd.DataFrame()  # Final lineage data_frame
            final_lineage_counter = 0  # Count total lineages
            all_lineages_detected = False

            # """
            # -----------------------------------------------------------------
            # Bifurcation
            # -----------------------------------------------------------------
            # """
            bifurcate_once = False  # Reset flag of bifurcation after each itteration.
            bifurcation_data = pd.DataFrame()
            bifurcation_plot_FLAG = True  # Plot bifurcation points on data
            total_bifurcations = 0
            # """
            # -----------------------------------------------------------------
            # Import data
            # -----------------------------------------------------------------
            # """
            self.create_bagel_loop_data()

            previouse_window_number = 0
            while all_lineages_detected is False:

                # """
                # -----------------------------------------------------------------
                # Algortihim specefics import
                # -----------------------------------------------------------------
                # """
                initialize_cells_offset = self.bagel_config["bagel_loop_config"][
                    "initialize_cells_offset"
                ]
                window_interval_extra_samples = self.bagel_config["bagel_loop_config"][
                    "window_interval_extra_samples"
                ]
                window_interval_actual_samples = self.bagel_config["bagel_loop_config"][
                    "window_interval_actual_samples"
                ]
                gibbs_samples = self.bagel_config["bagel_loop_config"]["gibbs_samples"]
                gibbs_burn_in_period = self.bagel_config["bagel_loop_config"][
                    "gibbs_burn_in_period"
                ]

                # Define default window sizes to be the same as config.
                # Note window intervals may change when close to terminal state.
                default_window_interval_extra_samples = (
                    window_interval_extra_samples  # |_______/
                )
                default_window_interval_actual_samples = (
                    window_interval_actual_samples  # |______|
                )

                # NOTE mapping
                # pseudo_time_interval = window_interval_extra_samples
                # window_interval = window_interval_actual_samples
                # pseudo_data = bagel_loop_data
                # once = split_in_lineage_once
                # twice = split_in_lineage_once
                # split = lineage_split_flag
                # once_data = split_in_lineage_once_data
                # twice_data = split_in_lineage_twice_data
                # Lineage_1 = main_lineage_1_df
                # Lineage_2 = main_lineage_2_df
                # wp_data_TSNE_ROW = bagel_loop_data_terminal_state
                # z = current_window_itteration
                # true_false = terminal_state_test_df_ts
                # pov_data = window_removed_bagel_loop_data
                # estimate_original_pt = window_bagel_loop_data,
                # estimate = window_bagel_loop_data,
                # p1_rotate = mean_window_bagel_loop_data,
                # Model_1 = one_model_flag
                # Model_2 = two_model_flag
                # window_3d = window_bagel_loop_data_no_cell_id

                # window_pseudo_data = window_bagel_loop_data
                (
                    total_windows,
                    half_window,
                    cell_total,
                ) = self.determine_total_bagel_loop_windows(
                    initialize_cells_offset,
                    window_interval_actual_samples,
                    window_interval_extra_samples,
                )  # Total windows and if there is half a window

                # """
                # -----------------------------------------------------------------
                # Data set for before and after split
                # -----------------------------------------------------------------
                # """
                g1_columns = ["g1_pseudo_time_normal", "g1_pca_1", "g1_pca_2"]
                before_split = pd.DataFrame(columns=g1_columns)
                g2_columns = ["g2_pseudo_time_normal", "g2_pca_1", "g2_pca_2"]
                after_split = pd.DataFrame(columns=g2_columns)

                # """
                # -----------------------------------------------------------------
                # Estimate split; if three splits the split is true
                # -----------------------------------------------------------------
                # """
                lineage_split_flag = (
                    False  # Flag for three splits and hence split is true
                )
                split_in_lineage_once = False  # Flag for one split
                split_in_lineage_twice = False  # Flag for two split

                # Save data for first split window
                split_in_lineage_once_data = pd.DataFrame(columns=g1_columns)

                # Save data for second split window
                split_in_lineage_twice_data = pd.DataFrame(columns=g2_columns)

                # """
                # -----------------------------------------------------------------
                # Compute lineages
                # -----------------------------------------------------------------
                # """

                # Place holder data
                column_place_holder_data = ["pseudo_time_normal", "pca_1", "pca_2"]
                once_data_original = pd.DataFrame(columns=column_place_holder_data)
                once_data_projected = pd.DataFrame(columns=column_place_holder_data)
                twice_data_original = pd.DataFrame(columns=column_place_holder_data)
                twice_data_projected = pd.DataFrame(columns=column_place_holder_data)

                previouse_window_l1 = pd.DataFrame(columns=column_place_holder_data)
                previouse_window_l2 = pd.DataFrame(columns=column_place_holder_data)

                main_lineage_1_df = pd.DataFrame(columns=column_place_holder_data)
                main_lineage_2_df = pd.DataFrame(columns=column_place_holder_data)

                first_time_association = False  # Start association after delay

                model_1_lineage_1_counter = (
                    0  # Count how many times model 1 is selected after split detection
                )
                model_1_lineage_2_counter = (
                    0  # Count how many times model 1 is selected after split detection
                )

                # """
                # -----------------------------------------------------------------
                # Window flag set
                # -----------------------------------------------------------------
                # """
                is_final_window = False  # Flag for final window

                # """
                # -----------------------------------------------------------------
                # Bifurcation
                # -----------------------------------------------------------------
                # """
                bifurcate_once = (
                    False  # Reset flag of bifurcation after each itteration.
                )

                previouse_window_l1_projected = pd.DataFrame()
                previouse_window_l2_projected = pd.DataFrame()

                # Data used during bagel calculation
                self.window_removed_bagel_loop_data = self.bagel_loop_data.copy()

                # Frenet frame
                frenet_frame_normal_vector = pd.DataFrame()
                frenet_frame_mean = pd.DataFrame()
                frenet_frame_counter = 0

                current_window_itteration = 0  # Plot number counter

                # Intialize proximity test
                previous_proximity = 0.00001

                while is_final_window is False:  # total_windows
                    current_window_itteration = (
                        current_window_itteration + 1
                    )  # Plot number counter

                    # """
                    # -----------------------------------------------------------------
                    # Determine adaptive window
                    # -----------------------------------------------------------------
                    # """
                    # TODO check all if states
                    (
                        is_final_window,
                        window_interval_extra_samples,
                        window_interval_actual_samples,
                    ) = self.determine_window_size_with_respect_to_ts(
                        default_window_interval_extra_samples,
                        default_window_interval_actual_samples,
                        is_final_window,
                    )

                    # """
                    # -----------------------------------------------------------------
                    # Apply maths
                    # -----------------------------------------------------------------
                    # """

                    # Determine if all windows has been processd
                    if self.window_removed_bagel_loop_data.empty is True:
                        break

                    # Model gaussians and Estimate split
                    if lineage_split_flag is False:
                        # """
                        # -----------------------------------------------------------------
                        # Estimate plane
                        # -----------------------------------------------------------------
                        (
                            _,
                            _,
                            covariance_length,
                            window_bagel_loop_data,
                            _,
                            self.window_removed_bagel_loop_data,
                        ) = frenet_frame_sup.frenet_frame_slice(
                            self.bagel_loop_data,
                            self.window_removed_bagel_loop_data.copy(),
                            window_interval_actual_samples,
                            window_interval_extra_samples,
                            is_final_window,
                        )

                        # Create  window data dataframe
                        window_bagel_loop_data_no_cell_id = window_bagel_loop_data[
                            ["pseudo_time_normal", "pca_1", "pca_2"]
                        ]
                        window_bagel_loop_data_no_cell_id.reset_index(
                            drop=True, inplace=True
                        )

                        # """
                        # -----------------------------------------------------------------
                        # Model selection
                        # -----------------------------------------------------------------
                        # """
                        (
                            previous_proximity,
                            ran_out_of_data_split,
                            one_model_flag,
                            two_model_flag,
                            data_output_gibbs_original,
                            data_output_gibbs_projected,
                            model_1_map_mean,
                            model_1_map_cov,
                            model_2_map_mean_G1,
                            model_2_map_cov_G1,
                            model_2_map_mean_G2,
                            model_2_map_cov_G2,
                        ) = bays_sup.gibbs_sampler_2_gaussians(
                            previous_proximity,
                            window_bagel_loop_data,
                            window_bagel_loop_data,
                            gibbs_samples,
                            gibbs_burn_in_period,
                            current_window_itteration,
                            self.window_removed_bagel_loop_data,
                            self.bagel_loop_data_terminal_state,
                        )

                        # """
                        # -----------------------------------------------------------------
                        # Save data
                        # -----------------------------------------------------------------
                        # """

                        # Estimate global split
                        (
                            split_in_lineage_once,
                            split_in_lineage_twice,
                            lineage_split_flag,
                            split_in_lineage_once_data,
                            split_in_lineage_twice_data,
                            once_data_original,
                            once_data_projected,
                            twice_data_original,
                            twice_data_projected,
                            first_time_association,
                            main_lineage_1_df,
                            main_lineage_2_df,
                        ) = majority_sup.estimate_guassians_on_manifold(
                            split_in_lineage_once,
                            split_in_lineage_twice,
                            one_model_flag,
                            two_model_flag,
                            split_in_lineage_once_data,
                            split_in_lineage_twice_data,
                            window_bagel_loop_data_no_cell_id,
                            lineage_split_flag,
                            is_final_window,
                            window_interval_extra_samples,
                            data_output_gibbs_original,
                            data_output_gibbs_projected,
                            first_time_association,
                            once_data_original,
                            once_data_projected,
                            twice_data_original,
                            twice_data_projected,
                            main_lineage_1_df,
                            main_lineage_2_df,
                            previouse_window_l1,
                            previouse_window_l2,
                            previouse_window_l1_projected,
                            previouse_window_l2_projected,
                            model_1_lineage_1_counter,
                            model_1_lineage_2_counter,
                        )

                        print()

                        if lineage_split_flag is True:
                            print("Verify possible split")
                            if self.window_removed_bagel_loop_data.empty is False:
                                (
                                    _,
                                    _,
                                    lineage_1_test,
                                    lineage_2_test,
                                    _,
                                    _,
                                    _,
                                    _,
                                    _,
                                    _,
                                    _,
                                ) = majority_sup.association(
                                    split_in_lineage_once,
                                    split_in_lineage_twice,
                                    lineage_split_flag,
                                    window_bagel_loop_data_no_cell_id,
                                    one_model_flag,
                                    two_model_flag,
                                    main_lineage_1_df,
                                    main_lineage_2_df,
                                    data_output_gibbs_original,
                                    data_output_gibbs_projected,
                                    once_data_original,
                                    once_data_projected,
                                    twice_data_original,
                                    twice_data_projected,
                                    first_time_association,
                                    previouse_window_l1,
                                    previouse_window_l2,
                                    previouse_window_l1_projected,
                                    previouse_window_l2_projected,
                                    model_1_lineage_1_counter,
                                    model_1_lineage_2_counter,
                                )
                                (
                                    lineage_1_test,
                                    lineage_2_test,
                                    previouse_window_l1_projected,
                                    previouse_window_l2_projected,
                                ) = majority_sup.after_split_euclidean_dist_association(
                                    lineage_1_test,
                                    lineage_2_test,
                                    self.window_removed_bagel_loop_data,
                                )

                                lineage_1_terminal_state = lineage_1_test[
                                    lineage_1_test["pseudo_time_normal"].isin(
                                        self.bagel_loop_data_terminal_state[
                                            "pseudo_time_normal"
                                        ].values
                                    )
                                ]
                                lineage_2_terminal_state = lineage_2_test[
                                    lineage_2_test["pseudo_time_normal"].isin(
                                        self.bagel_loop_data_terminal_state[
                                            "pseudo_time_normal"
                                        ].values
                                    )
                                ]
                            else:
                                (
                                    _,
                                    _,
                                    lineage_1_test,
                                    lineage_2_test,
                                    _,
                                    _,
                                    _,
                                    _,
                                    _,
                                    _,
                                    _,
                                ) = majority_sup.association(
                                    split_in_lineage_once,
                                    split_in_lineage_twice,
                                    lineage_split_flag,
                                    window_bagel_loop_data_no_cell_id,
                                    one_model_flag,
                                    two_model_flag,
                                    main_lineage_1_df,
                                    main_lineage_2_df,
                                    data_output_gibbs_original,
                                    data_output_gibbs_projected,
                                    once_data_original,
                                    once_data_projected,
                                    twice_data_original,
                                    twice_data_projected,
                                    first_time_association,
                                    previouse_window_l1,
                                    previouse_window_l2,
                                    previouse_window_l1_projected,
                                    previouse_window_l2_projected,
                                    model_1_lineage_1_counter,
                                    model_1_lineage_2_counter,
                                )
                                lineage_1_terminal_state = lineage_1_test[
                                    lineage_1_test["pseudo_time_normal"].isin(
                                        self.bagel_loop_data_terminal_state[
                                            "pseudo_time_normal"
                                        ].values
                                    )
                                ]
                                lineage_2_terminal_state = lineage_2_test[
                                    lineage_2_test["pseudo_time_normal"].isin(
                                        self.bagel_loop_data_terminal_state[
                                            "pseudo_time_normal"
                                        ].values
                                    )
                                ]

                            # State if valid split
                            if (lineage_1_terminal_state.empty is True) or (
                                lineage_2_terminal_state.empty is True
                            ):
                                # Fake lineages
                                lineage_split_flag = False
                                one_model_flag = True
                                two_model_flag = False
                                print("False split - reset paramters")
                            else:
                                print("Split verifyed correct")

                        # Asociate data to lineage
                        (
                            split_in_lineage_once,
                            split_in_lineage_twice,
                            main_lineage_1_df,
                            main_lineage_2_df,
                            previouse_window_l1,
                            previouse_window_l2,
                            previouse_window_l1_projected,
                            previouse_window_l2_projected,
                            first_time_association,
                            model_1_lineage_1_counter,
                            model_1_lineage_2_counter,
                        ) = majority_sup.association(
                            split_in_lineage_once,
                            split_in_lineage_twice,
                            lineage_split_flag,
                            window_bagel_loop_data_no_cell_id,
                            one_model_flag,
                            two_model_flag,
                            main_lineage_1_df,
                            main_lineage_2_df,
                            data_output_gibbs_original,
                            data_output_gibbs_projected,
                            once_data_original,
                            once_data_projected,
                            twice_data_original,
                            twice_data_projected,
                            first_time_association,
                            previouse_window_l1,
                            previouse_window_l2,
                            previouse_window_l1_projected,
                            previouse_window_l2_projected,
                            model_1_lineage_1_counter,
                            model_1_lineage_2_counter,
                        )

                        frenet_frame_counter = frenet_frame_counter + 1

                        # Create Frenet Frame

                        if frenet_frame_normal_vector.empty:
                            # covariance_length.reset_index(drop=True, inplace=True)
                            frenet_frame_normal_vector = covariance_length
                            frenet_frame_mean = pd.DataFrame(
                                [
                                    window_bagel_loop_data.iloc[:, 0:3]
                                    .mean(axis=0)
                                    .values.tolist()
                                ],
                                columns=window_bagel_loop_data.iloc[
                                    :, 0:3
                                ].columns.tolist(),
                            )

                        else:
                            # Normal vectors
                            frenet_frame_normal_vector = pd.concat(
                                [
                                    frenet_frame_normal_vector,
                                    covariance_length,
                                ],
                                axis=0,
                                sort=False,
                            )
                            frenet_frame_normal_vector = (
                                frenet_frame_normal_vector.reset_index(drop=True)
                            )

                            # Mean data
                            frenet_frame_mean = pd.concat(
                                [
                                    frenet_frame_mean,
                                    pd.DataFrame(
                                        [
                                            window_bagel_loop_data.iloc[:, 0:3]
                                            .mean(axis=0)
                                            .values.tolist()
                                        ],
                                        columns=window_bagel_loop_data.iloc[
                                            :, 0:3
                                        ].columns.tolist(),
                                    ),
                                ],
                                axis=0,
                                sort=False,
                            )
                            frenet_frame_mean = frenet_frame_mean.reset_index(drop=True)

                    elif lineage_split_flag is True:
                        (
                            main_lineage_1_df,
                            main_lineage_2_df,
                            previouse_window_l1_projected,
                            previouse_window_l2_projected,
                        ) = majority_sup.after_split_euclidean_dist_association(
                            main_lineage_1_df,
                            main_lineage_2_df,
                            self.window_removed_bagel_loop_data,
                        )
                        is_final_window = True

                        # Define 'window' data for final window
                        self.window_bagel_loop_data = (
                            self.window_removed_bagel_loop_data
                        )
                        # not_window_bagel_loop_data = self.bagel_loop_data[
                        #     ~self.bagel_loop_data["cell_id_number"].isin(
                        #         window_bagel_loop_data["cell_id_number"].values
                        #     )
                        # ]

                    # """
                    # -----------------------------------------------------------------
                    # Display
                    # -----------------------------------------------------------------
                    # """

                    # Plot VIDEO window data on manifold
                    window_number = previouse_window_number + current_window_itteration

                    # --------------------------------------
                    # ------------Split detection start-----
                    # --------------------------------------
                    if lineage_split_flag is False:
                        previouse_window_l1_projected = window_bagel_loop_data

                previouse_window_number = (
                    window_number  # Adjust labeling for VIDEO output
                )

                # Remove any duplicate assignments due to window and stepsize mismatch
                main_lineage_1_df = main_lineage_1_df.drop_duplicates()
                main_lineage_2_df = main_lineage_2_df.drop_duplicates()

                # Bifurcation
                (
                    bifurcation_data,
                    bifurcate_once,
                    total_bifurcations,
                ) = majority_sup.bifurcation_points(
                    lineage_split_flag,
                    bifurcation_data,
                    bifurcate_once,
                    total_bifurcations,
                    once_data_original,
                )

                if lineage_split_flag is True:
                    # Plot indidicating split across entire manifold

                    # plots.estimate_guassians_on_manifold_plot(before_split,after_split, estimate_guassians_on_manifold_plot_FLAG)
                    # plots.two_lineage_plot(Lineage_1, Lineage_2, two_lineage_plot_FLAG)

                    if to_be_determinded.empty is True:
                        main_lineage_1_df.reset_index(drop=True, inplace=True)
                        main_lineage_2_df.reset_index(drop=True, inplace=True)
                        to_be_determinded.reset_index(drop=True, inplace=True)
                        to_be_determinded = pd.concat(
                            [main_lineage_1_df, main_lineage_2_df], axis=1, sort=False
                        )  # Append new lineage data
                    else:
                        main_lineage_1_df.reset_index(drop=True, inplace=True)
                        main_lineage_2_df.reset_index(drop=True, inplace=True)
                        to_be_determinded.reset_index(drop=True, inplace=True)
                        to_be_determinded = pd.concat(
                            [to_be_determinded, main_lineage_1_df, main_lineage_2_df],
                            axis=1,
                            sort=False,
                        )  # Append new lineage data

                    self.bagel_loop_data = to_be_determinded.iloc[
                        :, 0:3
                    ]  # to_be_determinded[to_be_determinded.columns[0:3]] #Select first three columns as new input
                    self.bagel_loop_data = (
                        self.bagel_loop_data.dropna()
                    )  # Drop possible NAN data

                    # Sort data
                    self.bagel_loop_data = self.bagel_loop_data.sort_values(
                        "pseudo_time_normal"
                    )
                    # Provide each cell with a number
                    pt_samples = len(self.bagel_loop_data["pca_1"])
                    cell_id_number = range(pt_samples)
                    # Number of cell
                    self.bagel_loop_data["cell_id_number"] = cell_id_number

                    # Remove lineage from from to_be
                    to_be_determinded = to_be_determinded.iloc[:, 3:]

                    # to_be_determinded.columns = range(
                    #     to_be_determinded.shape[1]
                    # )  # Rename column names
                    # to_be_determinded = to_be_determinded.drop(
                    #     [0, 1, 2], axis=1
                    # )  # Remove new pseudo data lineage from data

                else:

                    # Dump frenet frame data
                    joblib.dump(
                        frenet_frame_normal_vector,
                        f"{self.result_folder}/frenet_frame_normal_vector"
                        + str(final_lineage_counter)
                        + ".pkl",
                        compress=3,
                    )
                    joblib.dump(
                        frenet_frame_mean,
                        f"{self.result_folder}/frenet_frame_mean"
                        + str(final_lineage_counter)
                        + ".pkl",
                        compress=3,
                    )
                    joblib.dump(
                        frenet_frame_counter,
                        f"{self.result_folder}/frenet_frame_counter"
                        + str(final_lineage_counter)
                        + ".pkl",
                        compress=3,
                    )

                    # Lineages
                    if final_lineage_df.empty:
                        final_lineage_df.reset_index(drop=True, inplace=True)
                        self.bagel_loop_data.reset_index(drop=True, inplace=True)
                        final_lineage_df = self.bagel_loop_data.iloc[:, 0:3]

                    else:
                        final_lineage_df.reset_index(drop=True, inplace=True)
                        self.bagel_loop_data.reset_index(drop=True, inplace=True)
                        frames = [final_lineage_df, self.bagel_loop_data.iloc[:, 0:3]]
                        final_lineage_df = pd.concat(
                            frames, axis=1, sort=False
                        )  # Append new lineage data

                    final_lineage_counter = final_lineage_counter + 1

                    if to_be_determinded.empty:
                        print("All lineages detected")
                        all_lineages_detected = True
                        break
                    else:
                        self.bagel_loop_data = to_be_determinded.iloc[
                            :, 0:3
                        ]  # to_be_determinded[to_be_determinded.columns[0:3]] #Select first three columns as new input
                        self.bagel_loop_data = (
                            self.bagel_loop_data.dropna()
                        )  # Drop possible NAN data

                        # Sort data
                        self.bagel_loop_data = self.bagel_loop_data.sort_values(
                            "pseudo_time_normal"
                        )
                        # Provide each cell with a number
                        pt_samples = len(self.bagel_loop_data["pca_1"])
                        cell_id_number = range(pt_samples)
                        # Number of cell
                        self.bagel_loop_data["cell_id_number"] = cell_id_number

                        # Remove lineage from from to_be
                        to_be_determinded = to_be_determinded.iloc[:, 3:]

                        # to_be_determinded.columns = range(
                        #     to_be_determinded.shape[1]
                        # )  # Rename column names
                        # to_be_determinded = to_be_determinded.drop(
                        #     [0, 1, 2], axis=1
                        # )  # Remove new pseudo data lineage from data

            # Dump pkl file
            joblib.dump(
                final_lineage_df,
                f"{self.result_folder}/final_lineage_df.pkl",
                compress=3,
            )
            joblib.dump(
                final_lineage_counter,
                f"{self.result_folder}/final_lineage_counter.pkl",
                compress=3,
            )
            joblib.dump(
                bifurcation_data,
                f"{self.result_folder}/bifurcation_data.pkl",
                compress=3,
            )
            joblib.dump(
                total_bifurcations,
                f"{self.result_folder}/total_bifurcations.pkl",
                compress=3,
            )
            joblib.dump(
                previouse_window_number,
                f"{self.result_folder}/previouse_window_number.pkl",
                compress=3,
            )
            print()

    def plot(self):
        """
        plot
        """
        # TODO update plot script
        plot_sup.results(
            self.bagel_config["plot_config"]["primary_label"],
            self.bagel_config["plot_config"]["secondary_label"],
            self.bagel_config["plot_config"]["output_prefix_label"],
            self.bagel_config["plot_config"]["gene_list"],
            self.bagel_config["plot_config"]["two_dimension_manifold_plot"],
            self.bagel_config["plot_config"]["three_dimension_manifold_plot"],
            self.bagel_config["plot_config"]["gene_expression_plot"],
            self.bagel_config["plot_config"]["bifurcation_plot"],
            self.bagel_config["plot_config"]["one_lineage_plot"],
            self.bagel_config["plot_config"]["all_lineage_plot"],
            self.bagel_config["plot_config"]["frenet_frame_plot"],
            self.bagel_config["plot_config"]["gp_with_data_plot"],
            self.bagel_config["plot_config"]["gp_only_plot"],
            self.bagel_config["plot_config"]["gp_per_lineage_plot"],
            self.result_folder,
        )


def main():
    """
    This function is used to test BAGEL locally
    """


if __name__ == "__main__":
    main()
