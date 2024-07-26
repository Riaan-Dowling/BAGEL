"""
This scirpt is the main function
"""

# import joblib
# import pseudo_time
# import lineages
# import D_resutls
# import shutil
import json
import os
import library.bagel_class as bagel_class_script


def run_bagel():
    """
    run_bagel
    """


def main():
    """
    main
    """

    print("BAGEL -- START.")

    parent_dir = os.path.dirname(os.path.realpath(__file__))
    files_body = open(f"{parent_dir}/config/bagel_config.json", "rb")
    bagel_config = json.load(files_body)

    # """
    # -----------------------------------------------------------------
    # BAGEL options
    # -----------------------------------------------------------------
    # """
    load_old_manifold = bagel_config["phenotypic_manifold_config"][
        "load_old_manifold"
    ]  # Start with a new manifold? This flag should only be set to True the first time that BAGEL obtains the phenotypic manifold of the current dataset/s. After this initial run, BAGEL stored the current phenotypic manifold and this flag can be set to False.
    results_only = bagel_config["output"][
        "results_only"
    ]  # Only display results? Can be set to true after BAGEL's initial run, as all of the modelling data has been saved locally.

    # """
    # -----------------------------------------------------------------
    # Input
    # -----------------------------------------------------------------
    # """
    main_data_file = bagel_config["phenotypic_manifold_config"]["main_data_file"]
    secondary_data_file = bagel_config["phenotypic_manifold_config"][
        "secondary_data_file"
    ]
    early_cell = bagel_config["phenotypic_manifold_config"]["early_cell"]

    # """
    # -----------------------------------------------------------------
    # START BAGEL
    # -----------------------------------------------------------------
    # """

    output_version_no = bagel_config["output"]["output_version_no"]
    bagel_object = bagel_class_script.BAGEL(
        bagel_config,
        load_old_manifold,
        results_only,
        main_data_file,
        secondary_data_file,
        early_cell,
        output_version_no,
    )
    bagel_object.create_processing_dir()
    bagel_object.load_datasets()

    if results_only is False:
        bagel_object.load_phenotypic_manifold()
        bagel_object.bagel_loop()
    bagel_object.plot()
    print("BAGEL -- END.")


if __name__ == "__main__":
    main()


# TODO remove below
#  # """
#     # -----------------------------------------------------------------
#     # Example inputs
#     # -----------------------------------------------------------------
#     # """

#     # 1 Results: Mouse bone marrow
#     # main_data_file = "data/sample_scseq_data.csv"  # Mouse
#     # early_cell = "W30258"
#     # secondary_data_file = ""

#     # 2 Results: Human bone marrow
#     # main_data_file = 'data/marrow_sample_scseq_counts.csv.gz' #Human 1
#     # secondary_data_file = ""
#     # early_cell = 'Run5_164698952452459'

#     # 3 Results: Projection of human UCB onto human bone marrow
#     main_data_file = "data/marrow_sample_scseq_counts.csv.gz"  # Human 1
#     secondary_data_file = "data/human_UCB.csv"  # Human 2
#     early_cell = "Run5_164698952452459"

#     # 1 Results: Projection of human UCB onto mouse bone marrow
#     # main_data_file = 'data/sample_scseq_data.csv' #Mouse
#     # secondary_data_file = 'data/human_UCB.csv' #Human 2
#     # early_cell = 'W30258'

# """
# -----------------------------------------------------------------
# OUTPUT
# -----------------------------------------------------------------
# """

# # human_bone_marrow_and_human_UCB
# # TODO move to config
# primary_label = "Human bone marrow"  # Primary data label
# secondary_label = "Human UCB"  # Secondary data label
# output_prefix_label = "human_bone_marrow_and_human_UCB"  # Label of saved figure

# # Display settings
# genelist = ["CD34", "GATA1", "MPO"]  # Genes to be displayed (must be equal to 3)

# two_dimension_manifold_plot = True  # Two dimensional phenotypic manifold plot
# three_dimension_manifold_plot = True  # Three dimensional phenotypic manifold plot
# gene_expression_plot = (
#     True  # Gene expressions of two dimensional phenotypic manifold plot
# )
# bifurcation_plot = True  # Detected bifurcation points plot
# one_lineage_plot = True  # Plot one detected lineage at a time
# all_lineage_plot = True  # Plot all detected lineage
# frenet_frame_plot = True  # Plot Frenet frame representation
# GP_with_data_plot = True  # Gaussian process with data plot
# GP_only_plot = True  # Gaussian process only plot
# GP_per_lineage_plot = (
#     True  # Plot one detected lineage with Gaussian process at a time
# )
