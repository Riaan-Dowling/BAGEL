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
    data_output_gibbs_ORIGNAl,
    data_output_gibbs_PROJECTED,
    first_time_association,
    once_data_ORIGNAl,
    once_data_PROJECTED,
    twice_data_ORIGNAl,
    twice_data_PROJECTED,
    lineage_1,
    lineage_2,
    previouse_window_l1,
    previouse_window_l2,
    previouse_window_l1_PROJECTED,
    previouse_window_l2_PROJECTED,
    Model_1_lineage_1_counter,
    Model_1_lineage_2_counter,
):

    # Rectify false positves

    if split is False:
        if (once is True) and (twice is False) and (Model_1 is True):

            # Ascociate data after mis detection
            # Asociate data to lineage
            # once_data_ORIGNAl
            (
                once,
                twice,
                lineage_1,
                lineage_2,
                previouse_window_l1,
                previouse_window_l2,
                previouse_window_l1_PROJECTED,
                previouse_window_l2_PROJECTED,
                first_time_association,
                Model_1_lineage_1_counter,
                Model_1_lineage_2_counter,
            ) = association(
                once,
                twice,
                split,
                window_3d,
                Model_1,
                Model_2,
                lineage_1,
                lineage_2,
                data_output_gibbs_ORIGNAl,
                data_output_gibbs_PROJECTED,
                once_data,
                once_data_PROJECTED,
                twice_data,
                twice_data_PROJECTED,
                first_time_association,
                previouse_window_l1,
                previouse_window_l2,
                previouse_window_l1_PROJECTED,
                previouse_window_l2_PROJECTED,
                Model_1_lineage_1_counter,
                Model_1_lineage_2_counter,
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
                lineage_1,
                lineage_2,
                previouse_window_l1,
                previouse_window_l2,
                previouse_window_l1_PROJECTED,
                previouse_window_l2_PROJECTED,
                first_time_association,
                Model_1_lineage_1_counter,
                Model_1_lineage_2_counter,
            ) = association(
                once,
                twice,
                split,
                window_3d,
                Model_1,
                Model_2,
                lineage_1,
                lineage_2,
                data_output_gibbs_ORIGNAl,
                data_output_gibbs_PROJECTED,
                once_data,
                once_data_PROJECTED,
                twice_data,
                twice_data_PROJECTED,
                first_time_association,
                previouse_window_l1,
                previouse_window_l2,
                previouse_window_l1_PROJECTED,
                previouse_window_l2_PROJECTED,
                Model_1_lineage_1_counter,
                Model_1_lineage_2_counter,
            )
            # Reset Flags
            once = False
            twice = False
    # """
    # -----------------------------------------------------------------
    # Ensure 3 consequtive splits
    # -----------------------------------------------------------------
    # """
    if (Model_2 is True) and (once is False) and (split is False):
        once = True  # First split
        once_data = window_3d
        # Once guassian split data
        once_data_ORIGNAl = data_output_gibbs_ORIGNAl
        once_data_PROJECTED = data_output_gibbs_PROJECTED

    elif (Model_2 is True) and (once is True) and (twice is False) and (split is False):
        twice = True  # Second split is true
        twice_data = window_3d
        # Twice gaussian split data
        twice_data_ORIGNAl = data_output_gibbs_ORIGNAl
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
        once_data_ORIGNAl,
        once_data_PROJECTED,
        twice_data_ORIGNAl,
        twice_data_PROJECTED,
        first_time_association,
        lineage_1,
        lineage_2,
    )


def association(
    once,
    twice,
    split,
    before_split,
    Model_1,
    Model_2,
    lineage_1,
    lineage_2,
    data_output_gibbs_ORIGNAl,
    data_output_gibbs_PROJECTED,
    once_data_ORIGNAl,
    once_data_PROJECTED,
    twice_data_ORIGNAl,
    twice_data_PROJECTED,
    first_time_association,
    previouse_window_l1,
    previouse_window_l2,
    previouse_window_l1_PROJECTED,
    previouse_window_l2_PROJECTED,
    Model_1_lineage_1_counter,
    Model_1_lineage_2_counter,
):
    # There is no split in data
    if (split is False) and (Model_1 is True) and (once is False) and (twice is False):
        # lineage 1
        before_split.columns = [
            "pseudo_time_normal",
            "pca_1",
            "pca_2",
        ]  # Set to lineage formating
        frames = [lineage_1, before_split]
        lineage_1 = pd.concat(frames, axis=0, ignore_index=False, sort=True)
        # lineage 2
        frames = [lineage_2, before_split]
        lineage_2 = pd.concat(frames, axis=0, ignore_index=False, sort=True)

        before_split.columns = [
            "g1_pseudo_time_normal",
            "g1_pca_1",
            "g1_pca_2",
        ]  # Reset to orignal formating4

    elif (split is False) and (Model_1 is True) and (once is True) and (twice is False):
        # lineage 1
        # Once data
        frames = [lineage_1, once_data_ORIGNAl]
        lineage_1 = pd.concat(frames, axis=0, ignore_index=False, sort=True)
        # Current data
        before_split.columns = [
            "pseudo_time_normal",
            "pca_1",
            "pca_2",
        ]  # Set to lineage formating
        frames = [lineage_1, before_split]
        lineage_1 = pd.concat(frames, axis=0, ignore_index=False, sort=True)

        # lineage 2
        # Once data
        frames = [lineage_2, once_data_ORIGNAl]
        lineage_2 = pd.concat(frames, axis=0, ignore_index=False, sort=True)
        # Current data
        frames = [lineage_2, before_split]
        lineage_2 = pd.concat(frames, axis=0, ignore_index=False, sort=True)

    elif (split is False) and (Model_1 is True) and (once is True) and (twice is True):
        # lineage 1
        # Once data
        frames = [lineage_1, once_data_ORIGNAl]
        lineage_1 = pd.concat(frames, axis=0, ignore_index=False, sort=True)
        # Twice data
        frames = [lineage_1, twice_data_ORIGNAl]
        lineage_1 = pd.concat(frames, axis=0, ignore_index=False, sort=True)
        # Current data
        before_split.columns = [
            "pseudo_time_normal",
            "pca_1",
            "pca_2",
        ]  # Set to lineage formating
        frames = [lineage_1, before_split]
        lineage_1 = pd.concat(frames, axis=0, ignore_index=False, sort=True)

        # lineage 2
        # Once data
        frames = [lineage_2, once_data_ORIGNAl]
        lineage_2 = pd.concat(frames, axis=0, ignore_index=False, sort=True)
        # Twice data
        frames = [lineage_2, twice_data_ORIGNAl]
        lineage_2 = pd.concat(frames, axis=0, ignore_index=False, sort=True)
        # Current data
        frames = [lineage_2, before_split]
        lineage_2 = pd.concat(frames, axis=0, ignore_index=False, sort=True)
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
                "pseudo_time_normal": once_data_ORIGNAl["g1_pseudo_time_normal"],
                "pca_1": once_data_ORIGNAl["g1_pca_1"],
                "pca_2": once_data_ORIGNAl["g1_pca_2"],
            }
            current_ORIGINAl_window_g1_3d = pd.DataFrame(d)
            current_ORIGINAl_window_g1_3d = (
                current_ORIGINAl_window_g1_3d.dropna()
            )  # Drop NAN rows

            d = {
                "pseudo_time_normal": once_data_ORIGNAl["g2_pseudo_time_normal"],
                "pca_1": once_data_ORIGNAl["g2_pca_1"],
                "pca_2": once_data_ORIGNAl["g2_pca_2"],
            }
            current_ORIGINAl_window_g2_3d = pd.DataFrame(d)
            current_ORIGINAl_window_g2_3d = (
                current_ORIGINAl_window_g2_3d.dropna()
            )  # Drop NAN rows

            # Data in original 2d window
            d = {
                "pca_1": once_data_ORIGNAl["g1_pca_1"],
                "pca_2": once_data_ORIGNAl["g1_pca_2"],
            }
            current_ORIGINAl_window_g1_2d = pd.DataFrame(d)
            current_ORIGINAl_window_g1_2d = (
                current_ORIGINAl_window_g1_2d.dropna()
            )  # Drop NAN rows

            d = {
                "pca_1": once_data_ORIGNAl["g2_pca_1"],
                "pca_2": once_data_ORIGNAl["g2_pca_2"],
            }
            current_ORIGINAl_window_g2_2d = pd.DataFrame(d)
            current_ORIGINAl_window_g2_2d = (
                current_ORIGINAl_window_g2_2d.dropna()
            )  # Drop NAN rows

            # Assign NEW previouse windows
            previouse_window_l1 = (
                current_ORIGINAl_window_g1_3d  # current_PROJECTED_window_g1
            )
            previouse_window_l2 = (
                current_ORIGINAl_window_g2_3d  # current_PROJECTED_window_g2
            )

            previouse_window_l1_PROJECTED = current_ORIGINAl_window_g1_3d
            previouse_window_l2_PROJECTED = current_ORIGINAl_window_g2_3d

            # Add data to lineage
            if once is True:
                frames = [lineage_1, current_ORIGINAl_window_g1_3d]
                lineage_1 = pd.concat(frames, axis=0, ignore_index=False, sort=True)

                frames = [lineage_2, current_ORIGINAl_window_g2_3d]
                lineage_2 = pd.concat(frames, axis=0, ignore_index=False, sort=True)

            # """
            # -----------------------------------------------------------------
            # Twice
            # -----------------------------------------------------------------
            # """
            if twice is True:
                (
                    lineage_1,
                    lineage_2,
                    previouse_window_l1,
                    previouse_window_l2,
                    previouse_window_l1_PROJECTED,
                    previouse_window_l2_PROJECTED,
                    Model_1_lineage_1_counter,
                    Model_1_lineage_2_counter,
                ) = euclidean_dist_association(
                    previouse_window_l1,
                    previouse_window_l2,
                    previouse_window_l1_PROJECTED,
                    previouse_window_l2_PROJECTED,
                    twice_data_ORIGNAl,
                    twice_data_PROJECTED,
                    lineage_1,
                    lineage_2,
                    Model_1,
                    Model_2,
                    Model_1_lineage_1_counter,
                    Model_1_lineage_2_counter,
                )
            # """
            # -----------------------------------------------------------------
            # Current
            # -----------------------------------------------------------------
            # """
            if (once is True) and (twice is True):
                (
                    lineage_1,
                    lineage_2,
                    previouse_window_l1,
                    previouse_window_l2,
                    previouse_window_l1_PROJECTED,
                    previouse_window_l2_PROJECTED,
                    Model_1_lineage_1_counter,
                    Model_1_lineage_2_counter,
                ) = euclidean_dist_association(
                    previouse_window_l1,
                    previouse_window_l2,
                    previouse_window_l1_PROJECTED,
                    previouse_window_l2_PROJECTED,
                    data_output_gibbs_ORIGNAl,
                    data_output_gibbs_PROJECTED,
                    lineage_1,
                    lineage_2,
                    Model_1,
                    Model_2,
                    Model_1_lineage_1_counter,
                    Model_1_lineage_2_counter,
                )

            first_time_association = False  # Ensure only do this step once
            # Reste flags to ensure this asociation only happens once
            once = False
            twice = False

    return (
        once,
        twice,
        lineage_1,
        lineage_2,
        previouse_window_l1,
        previouse_window_l2,
        previouse_window_l1_PROJECTED,
        previouse_window_l2_PROJECTED,
        first_time_association,
        Model_1_lineage_1_counter,
        Model_1_lineage_2_counter,
    )


def after_split_euclidean_dist_association(lineage_1, lineage_2, window_3d):
    """
    after_split_euclidean_dist_association
    """

    lineage_1 = lineage_1.reset_index(drop=True)
    lineage_2 = lineage_2.reset_index(drop=True)

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

    con = pd.concat([window_3d, cell_id_df], axis=1, sort=False)
    temp_id_window = pd.DataFrame(con.values, index=label, columns=con.columns)

    l1_counter = 0
    l2_counter = 0
    counter = 0
    for _ in tqdm(
        range(len(temp_id_window.index)), desc="Data association after split."
    ):
        # NOTE possibel uncomment for mean average of cells instead of single cell
        if len(lineage_1.index) <= 50:
            # l1_mean = lineage_1.tail(len(lineage_1.index)).mean(axis=0)
            l1_mean = lineage_1.tail(1)
        else:
            # l1_mean = lineage_1.tail(50).mean(axis=0)
            l1_mean = lineage_1.tail(1)

        if len(lineage_2.index) <= 50:
            # l2_mean = lineage_2.tail(len(lineage_2.index)).mean(axis=0)
            l2_mean = lineage_2.tail(1)
        else:
            # l2_mean = lineage_2.tail(50).mean(axis=0)
            l2_mean = lineage_2.tail(1)

        l1_mean.columns = lineage_1.columns
        l2_mean.columns = lineage_2.columns

        # Select one cell
        select_one = temp_id_window.head(1)
        # Remove selected cell from data
        temp_id_window = temp_id_window[
            ~temp_id_window["cell_id_number"].isin(select_one["cell_id_number"].values)
        ]

        # Delete cell ID column
        del select_one["cell_id_number"]
        # Calculate euclidean distance
        # fmt:off
        l1_data_euclidean_distance = np.linalg.norm(l1_mean[['pseudo_time_normal', 'pca_1', 'pca_2']].values - select_one[['pseudo_time_normal', 'pca_1', 'pca_2']].values)
        l2_data_euclidean_distance = np.linalg.norm(l2_mean[['pseudo_time_normal', 'pca_1', 'pca_2']].values - select_one[['pseudo_time_normal', 'pca_1', 'pca_2']].values)
        # fmt:on
        # print(f"l1_data_euclidean_distance - {l1_data_euclidean_distance}")
        # print(f"l2_data_euclidean_distance - {l2_data_euclidean_distance}")
        # print()

        if l1_data_euclidean_distance < l2_data_euclidean_distance:
            lineage_1 = pd.concat([lineage_1, select_one], axis=0).reset_index(
                drop=True
            )
            lineage_1 = lineage_1.sort_values(by="pseudo_time_normal", ascending=True)
            l1_counter = l1_counter + 1
        else:
            lineage_2 = pd.concat([lineage_2, select_one], axis=0).reset_index(
                drop=True
            )
            lineage_2 = lineage_2.sort_values(by="pseudo_time_normal", ascending=True)
            l2_counter = l2_counter + 1
        counter = counter + 1
        # if counter > 300:
        #     print()

    previouse_window_l1 = lineage_1.tail(l1_counter)
    previouse_window_l2 = lineage_2.tail(l2_counter)

    return lineage_1, lineage_2, previouse_window_l1, previouse_window_l2


def euclidean_dist_association(
    previouse_window_l1,
    previouse_window_l2,
    previouse_window_l1_PROJECTED,
    previouse_window_l2_PROJECTED,
    data_output_gibbs_ORIGNAl,
    data_output_gibbs_PROJECTED,
    lineage_1,
    lineage_2,
    Model_1,
    Model_2,
    Model_1_lineage_1_counter,
    Model_1_lineage_2_counter,
):

    # Reset model 1 count
    Model_1_lineage_1_counter = 0
    Model_1_lineage_2_counter = 0

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
        "pseudo_time_normal": data_output_gibbs_ORIGNAl["g1_pseudo_time_normal"],
        "pca_1": data_output_gibbs_ORIGNAl["g1_pca_1"],
        "pca_2": data_output_gibbs_ORIGNAl["g1_pca_2"],
    }
    current_ORIGINAl_window_g1_3d = pd.DataFrame(d)
    current_ORIGINAl_window_g1_3d = (
        current_ORIGINAl_window_g1_3d.dropna()
    )  # Drop NAN rows

    d = {
        "pseudo_time_normal": data_output_gibbs_ORIGNAl["g2_pseudo_time_normal"],
        "pca_1": data_output_gibbs_ORIGNAl["g2_pca_1"],
        "pca_2": data_output_gibbs_ORIGNAl["g2_pca_2"],
    }
    current_ORIGINAl_window_g2_3d = pd.DataFrame(d)
    current_ORIGINAl_window_g2_3d = (
        current_ORIGINAl_window_g2_3d.dropna()
    )  # Drop NAN rows

    d = {
        "pca_1": data_output_gibbs_ORIGNAl["g1_pca_1"],
        "pca_2": data_output_gibbs_ORIGNAl["g1_pca_2"],
    }
    current_ORIGINAl_window_g1 = pd.DataFrame(d)
    current_ORIGINAl_window_g1_2d = current_ORIGINAl_window_g1.dropna()  # Drop NAN rows

    d = {
        "pca_1": data_output_gibbs_ORIGNAl["g2_pca_1"],
        "pca_2": data_output_gibbs_ORIGNAl["g2_pca_2"],
    }
    current_ORIGINAl_window_g2_2d = pd.DataFrame(d)
    current_ORIGINAl_window_g2_2d = (
        current_ORIGINAl_window_g2_2d.dropna()
    )  # Drop NAN rows

    if len(previouse_window_l1.index) > len(previouse_window_l2.index):
        previouse_window_l1 = previouse_window_l1.sample(
            n=len(previouse_window_l2.index)
        )
    else:
        previouse_window_l2 = previouse_window_l2.sample(
            n=len(previouse_window_l1.index)
        )

    # The prevouse data is known as the image and the current data as the template
    # Data frame of mean meseure distance to mean
    mean = previouse_window_l1.mean(axis=0)
    mean.columns = ["pca_1", "pca_2"]
    # Calculate euclidean distance
    euclidean_distance_l1_g1 = (
        (current_ORIGINAl_window_g1_3d - mean).pow(2).sum(1).pow(0.5)
    )
    euclidean_distance_l1_g1 = euclidean_distance_l1_g1.mean(axis=0)

    mean = previouse_window_l2.mean(axis=0)
    mean.columns = ["pca_1", "pca_2"]
    euclidean_distance_l2_g1 = (
        (current_ORIGINAl_window_g1_3d - mean).pow(2).sum(1).pow(0.5)
    )
    euclidean_distance_l2_g1 = euclidean_distance_l2_g1.mean(axis=0)

    # Bigger distance is bad!
    if euclidean_distance_l1_g1 > euclidean_distance_l2_g1:
        NEW_previouse_window_l1 = current_ORIGINAl_window_g2_3d  # Assign the current window as future previose window
        NEW_previouse_window_l2 = current_ORIGINAl_window_g1_3d  # Assign the current window as future previose window

        NEW_previouse_window_l1_PROJECTED = current_ORIGINAl_window_g2_3d
        NEW_previouse_window_l2_PROJECTED = current_ORIGINAl_window_g1_3d

        # Correctly append asociated data to new lineage

        frames = [lineage_1, current_ORIGINAl_window_g2_3d]
        lineage_1 = pd.concat(frames, axis=0, ignore_index=False, sort=True)

        frames = [lineage_2, current_ORIGINAl_window_g1_3d]
        lineage_2 = pd.concat(frames, axis=0, ignore_index=False, sort=True)

    else:
        NEW_previouse_window_l1 = current_ORIGINAl_window_g1_3d  # Assign the current window as future previose window
        NEW_previouse_window_l2 = current_ORIGINAl_window_g2_3d  # Assign the current window as future previose window

        NEW_previouse_window_l1_PROJECTED = current_ORIGINAl_window_g1_3d
        NEW_previouse_window_l2_PROJECTED = current_ORIGINAl_window_g2_3d

        frames = [lineage_1, current_ORIGINAl_window_g1_3d]
        lineage_1 = pd.concat(frames, axis=0, ignore_index=False, sort=True)

        frames = [lineage_2, current_ORIGINAl_window_g2_3d]
        lineage_2 = pd.concat(frames, axis=0, ignore_index=False, sort=True)

    # Associate lineage_1 and lineage_2 with before_split and after_split due to window and step size mis match

    return (
        lineage_1,
        lineage_2,
        NEW_previouse_window_l1,
        NEW_previouse_window_l2,
        NEW_previouse_window_l1_PROJECTED,
        NEW_previouse_window_l2_PROJECTED,
        Model_1_lineage_1_counter,
        Model_1_lineage_2_counter,
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


# import matplotlib.pyplot as plt
# import time
# for step in range(20):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection="3d")
#     ax.view_init(elev=10, azim=10 + 20*step)
#     ax.scatter(
#         lineage_2["pseudo_time_normal"],
#         lineage_2["pca_1"],
#         lineage_2["pca_2"],
#         c="c",
#         marker="X",
#         s=50,
#         alpha=0.1,
#     )
#     ax.scatter(
#         lineage_1["pseudo_time_normal"],
#         lineage_1["pca_1"],
#         lineage_1["pca_2"],
#         c="m",
#         marker="X",
#         s=50,
#         alpha=0.1,
#     )
#     ax.scatter(
#         select_one["pseudo_time_normal"],
#         select_one["pca_1"],
#         select_one["pca_2"],
#         c="r",
#         marker="o",
#         s=70,
#     )
#     ax.scatter(
#         l1_mean["pseudo_time_normal"],
#         l1_mean["pca_1"],
#         l1_mean["pca_2"],
#         c="orange",
#         marker="o",
#         s=70,
#     )
#     ax.scatter(
#         l2_mean["pseudo_time_normal"],
#         l2_mean["pca_1"],
#         l2_mean["pca_2"],
#         c="blue",
#         marker="o",
#         s=70,
#     )
#     plt.savefig("temp.png")
#     plt.show()
#     time.sleep(0.5)
