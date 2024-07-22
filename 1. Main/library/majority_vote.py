"""
majority voting
"""

import numpy as np
import pandas as pd
from tqdm import tqdm


def estimate_guassians_on_manifold(
    once,
    twice,
    Model_1,
    Model_2,
    once_data,
    twice_data,
    window_3d,
    split,
    final_window,
    window_pseudo_time_interval,
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
):

    # Rectify false positves

    if split is False:
        if (once is True) and (twice is False) and (Model_1 is True):

            # Ascociate data after mis detection
            # Asociate data to lineage
            # once_data_ORIGNAL
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
            ) = association(
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
                once_data,
                once_data_PROJECTED,
                twice_data,
                twice_data_PROJECTED,
                first_time_association,
                previouse_window_L1,
                previouse_window_L2,
                previouse_window_L1_PROJECTED,
                previouse_window_L2_PROJECTED,
                Model_1_Lineage_1_counter,
                Model_1_Lineage_2_counter,
            )
            # Reset Flags
            once = False
            twice = False

        elif (once is True) and (twice is True) and (Model_1 is True):

            # Ascociate data after mis detection
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
            ) = association(
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
                once_data,
                once_data_PROJECTED,
                twice_data,
                twice_data_PROJECTED,
                first_time_association,
                previouse_window_L1,
                previouse_window_L2,
                previouse_window_L1_PROJECTED,
                previouse_window_L2_PROJECTED,
                Model_1_Lineage_1_counter,
                Model_1_Lineage_2_counter,
            )
            # Reset Flags
            once = False
            twice = False
    """
    -----------------------------------------------------------------
    Ensure 3 consequtive splits
    -----------------------------------------------------------------
    """
    if (Model_2 is True) and (once is False) and (split is False):
        once = True  # First split
        once_data = window_3d
        # Once guassian split data
        once_data_ORIGNAL = data_output_gibbs_ORIGNAL
        once_data_PROJECTED = data_output_gibbs_PROJECTED

    elif (Model_2 is True) and (once is True) and (twice is False) and (split is False):
        twice = True  # Second split is true
        twice_data = window_3d
        # Twice gaussian split data
        twice_data_ORIGNAL = data_output_gibbs_ORIGNAL
        twice_data_PROJECTED = data_output_gibbs_PROJECTED

    elif (
        (Model_2 is True)
        and (once is True)
        and (twice is True)
        and (first_time_association is False)
    ):
        split = True  # Third split is true, hence save as split
        first_time_association = True  # Split first time association
        print("Split detected")

    return (
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
    )


def association(
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
):
    # There is no split in data
    if (split is False) and (Model_1 is True) and (once is False) and (twice is False):
        # Lineage 1
        before_split.columns = [
            "pseudo_time_normal",
            "pca_1",
            "pca_2",
        ]  # Set to lineage formating
        frames = [Lineage_1, before_split]
        Lineage_1 = pd.concat(frames, axis=0, ignore_index=False, sort=True)
        # Lineage 2
        frames = [Lineage_2, before_split]
        Lineage_2 = pd.concat(frames, axis=0, ignore_index=False, sort=True)

        before_split.columns = [
            "g1_pseudo_time_normal",
            "g1_pca_1",
            "g1_pca_2",
        ]  # Reset to orignal formating4

    elif (split is False) and (Model_1 is True) and (once is True) and (twice is False):
        # Lineage 1
        # Once data
        frames = [Lineage_1, once_data_ORIGNAL]
        Lineage_1 = pd.concat(frames, axis=0, ignore_index=False, sort=True)
        # Current data
        before_split.columns = [
            "pseudo_time_normal",
            "pca_1",
            "pca_2",
        ]  # Set to lineage formating
        frames = [Lineage_1, before_split]
        Lineage_1 = pd.concat(frames, axis=0, ignore_index=False, sort=True)

        # Lineage 2
        # Once data
        frames = [Lineage_2, once_data_ORIGNAL]
        Lineage_2 = pd.concat(frames, axis=0, ignore_index=False, sort=True)
        # Current data
        frames = [Lineage_2, before_split]
        Lineage_2 = pd.concat(frames, axis=0, ignore_index=False, sort=True)

    elif (split is False) and (Model_1 is True) and (once is True) and (twice is True):
        # Lineage 1
        # Once data
        frames = [Lineage_1, once_data_ORIGNAL]
        Lineage_1 = pd.concat(frames, axis=0, ignore_index=False, sort=True)
        # Twice data
        frames = [Lineage_1, twice_data_ORIGNAL]
        Lineage_1 = pd.concat(frames, axis=0, ignore_index=False, sort=True)
        # Current data
        before_split.columns = [
            "pseudo_time_normal",
            "pca_1",
            "pca_2",
        ]  # Set to lineage formating
        frames = [Lineage_1, before_split]
        Lineage_1 = pd.concat(frames, axis=0, ignore_index=False, sort=True)

        # Lineage 2
        # Once data
        frames = [Lineage_2, once_data_ORIGNAL]
        Lineage_2 = pd.concat(frames, axis=0, ignore_index=False, sort=True)
        # Twice data
        frames = [Lineage_2, twice_data_ORIGNAL]
        Lineage_2 = pd.concat(frames, axis=0, ignore_index=False, sort=True)
        # Current data
        frames = [Lineage_2, before_split]
        Lineage_2 = pd.concat(frames, axis=0, ignore_index=False, sort=True)
        before_split.columns = [
            "g1_pseudo_time_normal",
            "g1_pca_1",
            "g1_pca_2",
        ]  # Reset to orignal formating

    elif split is True:
        if first_time_association is True:
            # Because there is a dely in finalising a split deinition the delyad data has to be assign before continuing
            # The first step is to assign data to lineafge 1 or two 'randomly' because this is the first split data

            # """
            # -----------------------------------------------------------------
            # Once
            # -----------------------------------------------------------------
            # """
            # Data projected onto plane
            d = {
                "pca_1": once_data_PROJECTED["g1_pca_1"],
                "pca_2": once_data_PROJECTED["g1_pca_2"],
            }
            current_PROJECTED_window_g1 = pd.DataFrame(d)
            current_PROJECTED_window_g1 = (
                current_PROJECTED_window_g1.dropna()
            )  # Drop NAN rows

            d = {
                "pca_1": once_data_PROJECTED["g2_pca_1"],
                "pca_2": once_data_PROJECTED["g2_pca_2"],
            }
            current_PROJECTED_window_g2 = pd.DataFrame(d)
            current_PROJECTED_window_g2 = (
                current_PROJECTED_window_g2.dropna()
            )  # Drop NAN rows

            # Data in original 3d window
            d = {
                "pseudo_time_normal": once_data_ORIGNAL["g1_pseudo_time_normal"],
                "pca_1": once_data_ORIGNAL["g1_pca_1"],
                "pca_2": once_data_ORIGNAL["g1_pca_2"],
            }
            current_ORIGINAL_window_g1_3d = pd.DataFrame(d)
            current_ORIGINAL_window_g1_3d = (
                current_ORIGINAL_window_g1_3d.dropna()
            )  # Drop NAN rows

            d = {
                "pseudo_time_normal": once_data_ORIGNAL["g2_pseudo_time_normal"],
                "pca_1": once_data_ORIGNAL["g2_pca_1"],
                "pca_2": once_data_ORIGNAL["g2_pca_2"],
            }
            current_ORIGINAL_window_g2_3d = pd.DataFrame(d)
            current_ORIGINAL_window_g2_3d = (
                current_ORIGINAL_window_g2_3d.dropna()
            )  # Drop NAN rows

            # Data in original 2d window
            d = {
                "pca_1": once_data_ORIGNAL["g1_pca_1"],
                "pca_2": once_data_ORIGNAL["g1_pca_2"],
            }
            current_ORIGINAL_window_g1_2d = pd.DataFrame(d)
            current_ORIGINAL_window_g1_2d = (
                current_ORIGINAL_window_g1_2d.dropna()
            )  # Drop NAN rows

            d = {
                "pca_1": once_data_ORIGNAL["g2_pca_1"],
                "pca_2": once_data_ORIGNAL["g2_pca_2"],
            }
            current_ORIGINAL_window_g2_2d = pd.DataFrame(d)
            current_ORIGINAL_window_g2_2d = (
                current_ORIGINAL_window_g2_2d.dropna()
            )  # Drop NAN rows

            # Assign NEW previouse windows
            previouse_window_L1 = (
                current_ORIGINAL_window_g1_3d  # current_PROJECTED_window_g1
            )
            previouse_window_L2 = (
                current_ORIGINAL_window_g2_3d  # current_PROJECTED_window_g2
            )

            previouse_window_L1_PROJECTED = current_ORIGINAL_window_g1_3d
            previouse_window_L2_PROJECTED = current_ORIGINAL_window_g2_3d

            # Add data to lineage
            if once is True:
                frames = [Lineage_1, current_ORIGINAL_window_g1_3d]
                Lineage_1 = pd.concat(frames, axis=0, ignore_index=False, sort=True)

                frames = [Lineage_2, current_ORIGINAL_window_g2_3d]
                Lineage_2 = pd.concat(frames, axis=0, ignore_index=False, sort=True)

            # """
            # -----------------------------------------------------------------
            # Twice
            # -----------------------------------------------------------------
            # """
            if twice is True:
                (
                    Lineage_1,
                    Lineage_2,
                    previouse_window_L1,
                    previouse_window_L2,
                    previouse_window_L1_PROJECTED,
                    previouse_window_L2_PROJECTED,
                    Model_1_Lineage_1_counter,
                    Model_1_Lineage_2_counter,
                ) = euclidean_dist_association(
                    previouse_window_L1,
                    previouse_window_L2,
                    previouse_window_L1_PROJECTED,
                    previouse_window_L2_PROJECTED,
                    twice_data_ORIGNAL,
                    twice_data_PROJECTED,
                    Lineage_1,
                    Lineage_2,
                    Model_1,
                    Model_2,
                    Model_1_Lineage_1_counter,
                    Model_1_Lineage_2_counter,
                )
            # """
            # -----------------------------------------------------------------
            # Current
            # -----------------------------------------------------------------
            # """
            if (once is True) and (twice is True):
                (
                    Lineage_1,
                    Lineage_2,
                    previouse_window_L1,
                    previouse_window_L2,
                    previouse_window_L1_PROJECTED,
                    previouse_window_L2_PROJECTED,
                    Model_1_Lineage_1_counter,
                    Model_1_Lineage_2_counter,
                ) = euclidean_dist_association(
                    previouse_window_L1,
                    previouse_window_L2,
                    previouse_window_L1_PROJECTED,
                    previouse_window_L2_PROJECTED,
                    data_output_gibbs_ORIGNAL,
                    data_output_gibbs_PROJECTED,
                    Lineage_1,
                    Lineage_2,
                    Model_1,
                    Model_2,
                    Model_1_Lineage_1_counter,
                    Model_1_Lineage_2_counter,
                )

            first_time_association = False  # Ensure only do this step once
            # Reste flags to ensure this asociation only happens once
            once = False
            twice = False

    return (
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
    )


def after_split_euclidean_dist_association(Lineage_1, Lineage_2, window_3d):

    window_3d = window_3d.iloc[:, 0:3]

    # Calculate euclidean distance till each mean
    # Set cell ID number to each cell to enable singel cell selection
    pt_samples = len(window_3d.index)
    cell_id_number = range(pt_samples)
    label = window_3d.index
    d = {"cell_id_number": cell_id_number}
    cell_id_df = pd.DataFrame(d)

    window_3d.reset_index(drop=True, inplace=True)
    cell_id_df.reset_index(drop=True, inplace=True)

    # Append column
    Frames = [window_3d, cell_id_df]
    con = pd.concat(Frames, axis=1, sort=False)
    temp_ID_WINDOW = pd.DataFrame(con.values, index=label, columns=con.columns)

    L1_counter = 0
    L2_counter = 0
    for _ in tqdm(
        range(len(temp_ID_WINDOW.index)), desc="Data association after split."
    ):
        # TODO update paper to use mean
        # TODO uncomment for mean average of cells instead of single cell
        if len(Lineage_1.index) <= 50:
            L1_mean = Lineage_1.tail(len(Lineage_1.index)).mean(axis=0)
            # L1_mean = Lineage_1.tail(1)
        else:
            L1_mean = Lineage_1.tail(50).mean(axis=0)
            # L1_mean = Lineage_1.tail(1)

        if len(Lineage_2.index) <= 50:
            L2_mean = Lineage_2.tail(len(Lineage_2.index)).mean(axis=0)
            # L2_mean = Lineage_2.tail(1)
        else:
            L2_mean = Lineage_2.tail(50).mean(axis=0)
            # L2_mean = Lineage_2.tail(1)

        L1_mean.columns = ["pseudo_time_normal", "pca_1", "pca_2"]
        L2_mean.columns = ["pseudo_time_normal", "pca_1", "pca_2"]

        # Select one cell
        select_one = temp_ID_WINDOW.head(1)
        # Remove selected cell from data
        temp_ID_WINDOW = temp_ID_WINDOW[
            ~temp_ID_WINDOW["cell_id_number"].isin(select_one["cell_id_number"].values)
        ]

        # Delete cell ID column
        del select_one["cell_id_number"]
        # Calculate euclidean distance
        L1_Data_euclidean_distance = np.linalg.norm(L1_mean.values - select_one.values)
        L2_Data_euclidean_distance = np.linalg.norm(L2_mean.values - select_one.values)

        if L1_Data_euclidean_distance <= L2_Data_euclidean_distance:
            Frames = [Lineage_1, select_one]
            Lineage_1 = pd.concat(Frames, axis=0, sort=False)
            L1_counter = L1_counter + 1
        else:
            Frames = [Lineage_2, select_one]
            Lineage_2 = pd.concat(Frames, axis=0, sort=False)
            L2_counter = L2_counter + 1

    previouse_window_L1 = Lineage_1.tail(L1_counter)
    previouse_window_L2 = Lineage_2.tail(L2_counter)

    return Lineage_1, Lineage_2, previouse_window_L1, previouse_window_L2


def euclidean_dist_association(
    previouse_window_L1,
    previouse_window_L2,
    previouse_window_L1_PROJECTED,
    previouse_window_L2_PROJECTED,
    data_output_gibbs_ORIGNAL,
    data_output_gibbs_PROJECTED,
    Lineage_1,
    Lineage_2,
    Model_1,
    Model_2,
    Model_1_Lineage_1_counter,
    Model_1_Lineage_2_counter,
):

    # Reset model 1 count
    Model_1_Lineage_1_counter = 0
    Model_1_Lineage_2_counter = 0

    # Data projected onto plane
    d = {
        "pca_1": data_output_gibbs_PROJECTED["g1_pca_1"],
        "pca_2": data_output_gibbs_PROJECTED["g1_pca_2"],
    }
    current_PROJECTED_window_g1 = pd.DataFrame(d)
    current_PROJECTED_window_g1 = current_PROJECTED_window_g1.dropna()  # Drop NAN rows

    d = {
        "pca_1": data_output_gibbs_PROJECTED["g2_pca_1"],
        "pca_2": data_output_gibbs_PROJECTED["g2_pca_2"],
    }
    current_PROJECTED_window_g2 = pd.DataFrame(d)
    current_PROJECTED_window_g2 = current_PROJECTED_window_g2.dropna()  # Drop NAN rows

    # Data in original 3d window
    d = {
        "pseudo_time_normal": data_output_gibbs_ORIGNAL["g1_pseudo_time_normal"],
        "pca_1": data_output_gibbs_ORIGNAL["g1_pca_1"],
        "pca_2": data_output_gibbs_ORIGNAL["g1_pca_2"],
    }
    current_ORIGINAL_window_g1_3d = pd.DataFrame(d)
    current_ORIGINAL_window_g1_3d = (
        current_ORIGINAL_window_g1_3d.dropna()
    )  # Drop NAN rows

    d = {
        "pseudo_time_normal": data_output_gibbs_ORIGNAL["g2_pseudo_time_normal"],
        "pca_1": data_output_gibbs_ORIGNAL["g2_pca_1"],
        "pca_2": data_output_gibbs_ORIGNAL["g2_pca_2"],
    }
    current_ORIGINAL_window_g2_3d = pd.DataFrame(d)
    current_ORIGINAL_window_g2_3d = (
        current_ORIGINAL_window_g2_3d.dropna()
    )  # Drop NAN rows

    d = {
        "pca_1": data_output_gibbs_ORIGNAL["g1_pca_1"],
        "pca_2": data_output_gibbs_ORIGNAL["g1_pca_2"],
    }
    current_ORIGINAL_window_g1 = pd.DataFrame(d)
    current_ORIGINAL_window_g1_2d = current_ORIGINAL_window_g1.dropna()  # Drop NAN rows

    d = {
        "pca_1": data_output_gibbs_ORIGNAL["g2_pca_1"],
        "pca_2": data_output_gibbs_ORIGNAL["g2_pca_2"],
    }
    current_ORIGINAL_window_g2_2d = pd.DataFrame(d)
    current_ORIGINAL_window_g2_2d = (
        current_ORIGINAL_window_g2_2d.dropna()
    )  # Drop NAN rows

    if len(previouse_window_L1.index) > len(previouse_window_L2.index):
        previouse_window_L1 = previouse_window_L1.sample(
            n=len(previouse_window_L2.index)
        )
    else:
        previouse_window_L2 = previouse_window_L2.sample(
            n=len(previouse_window_L1.index)
        )

    # The prevouse data is known as the image and the current data as the template
    # Data frame of mean meseure distance to mean
    mean = previouse_window_L1.mean(axis=0)
    mean.columns = ["pca_1", "pca_2"]
    # Calculate euclidean distance
    euclidean_distance_L1_g1 = (
        (current_ORIGINAL_window_g1_3d - mean).pow(2).sum(1).pow(0.5)
    )
    euclidean_distance_L1_g1 = euclidean_distance_L1_g1.mean(axis=0)

    mean = previouse_window_L2.mean(axis=0)
    mean.columns = ["pca_1", "pca_2"]
    euclidean_distance_L2_g1 = (
        (current_ORIGINAL_window_g1_3d - mean).pow(2).sum(1).pow(0.5)
    )
    euclidean_distance_L2_g1 = euclidean_distance_L2_g1.mean(axis=0)

    # Bigger distance is bad!
    if euclidean_distance_L1_g1 > euclidean_distance_L2_g1:
        NEW_previouse_window_L1 = current_ORIGINAL_window_g2_3d  # Assign the current window as future previose window
        NEW_previouse_window_L2 = current_ORIGINAL_window_g1_3d  # Assign the current window as future previose window

        NEW_previouse_window_L1_PROJECTED = current_ORIGINAL_window_g2_3d
        NEW_previouse_window_L2_PROJECTED = current_ORIGINAL_window_g1_3d

        # Correctly append asociated data to new lineage

        frames = [Lineage_1, current_ORIGINAL_window_g2_3d]
        Lineage_1 = pd.concat(frames, axis=0, ignore_index=False, sort=True)

        frames = [Lineage_2, current_ORIGINAL_window_g1_3d]
        Lineage_2 = pd.concat(frames, axis=0, ignore_index=False, sort=True)

    else:
        NEW_previouse_window_L1 = current_ORIGINAL_window_g1_3d  # Assign the current window as future previose window
        NEW_previouse_window_L2 = current_ORIGINAL_window_g2_3d  # Assign the current window as future previose window

        NEW_previouse_window_L1_PROJECTED = current_ORIGINAL_window_g1_3d
        NEW_previouse_window_L2_PROJECTED = current_ORIGINAL_window_g2_3d

        frames = [Lineage_1, current_ORIGINAL_window_g1_3d]
        Lineage_1 = pd.concat(frames, axis=0, ignore_index=False, sort=True)

        frames = [Lineage_2, current_ORIGINAL_window_g2_3d]
        Lineage_2 = pd.concat(frames, axis=0, ignore_index=False, sort=True)

    # Associate lineage_1 and lineage_2 with before_split and after_split due to window and step size mis match

    return (
        Lineage_1,
        Lineage_2,
        NEW_previouse_window_L1,
        NEW_previouse_window_L2,
        NEW_previouse_window_L1_PROJECTED,
        NEW_previouse_window_L2_PROJECTED,
        Model_1_Lineage_1_counter,
        Model_1_Lineage_2_counter,
    )


def bifurcation_points(
    split, bifurcation_data, bifurcate_once, total_bifurcations, once_data
):
    if (split is True) and (bifurcate_once is False):
        bifurcate_once = True  # Ensure one bifurcation data point per run
        if bifurcation_data.empty:
            bifurcation_data = once_data
            total_bifurcations = total_bifurcations + 1
        else:
            once_data.reset_index(drop=True, inplace=True)
            bifurcation_data.reset_index(drop=True, inplace=True)

            frames = [bifurcation_data, once_data]
            bifurcation_data = pd.concat(
                frames, axis=1, sort=False
            )  # Append new bifurcation data
            total_bifurcations = total_bifurcations + 1

    return bifurcation_data, bifurcate_once, total_bifurcations
