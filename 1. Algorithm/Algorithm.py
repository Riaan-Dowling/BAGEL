import joblib
import pseudo_time
import lineages
import D_resutls
import shutil
import os


def main():
    print("BAGEL -- START.")

    """
    -----------------------------------------------------------------
    Model options
    -----------------------------------------------------------------
    """
    new_manifold_FLAG = True  # Start with a new manifold? This flag should only be set to True the first time that BAGEL obtains the phenotypic manifold of the current dataset/s. After this initial run, BAGEL stored the current phenotypic manifold and this flag can be set to False.
    two_data_set_FLAG = False  # Is two diferent datasets used? Is there one (Flase) or Two (True) datasets used as input to BAGEL
    results_only = False  # Only display results? Can be set to true after BAGEL's initial run, as all of the modelling data has been saved locally.

    """
    -----------------------------------------------------------------
    Data sets
    -----------------------------------------------------------------
    """

    # 1 Results: Mouse bone marrow
    Main_data_file = "sample_scseq_data.csv"  # Mouse
    early_cell = "W30258"

    # 2 Results: Human bone marrow
    # Main_data_file = 'marrow_sample_scseq_counts.csv.gz' #Human 1
    # early_cell = 'Run5_164698952452459'

    # 3 Results: Projection of human UCB onto human bone marrow
    # Main_data_file = 'marrow_sample_scseq_counts.csv.gz' #Human 1
    # Secondary_data_file = 'human_UCB.csv' #Human 2
    # early_cell = 'Run5_164698952452459'

    # 1 Results: Projection of human UCB onto mouse bone marrow
    # Main_data_file = 'sample_scseq_data.csv' #Mouse
    # Secondary_data_file = 'human_UCB.csv' #Human 2
    # early_cell = 'W30258'

    """
    -----------------------------------------------------------------
    Results
    -----------------------------------------------------------------
    """

    # human_bone_marrow_and_human_UCB

    Primary_label = "Human bone marrow"  # Primary data label
    Secondary_label = "Human UCB"  # Secondary data label
    Picture_name = "human_bone_marrow_and_human_UCB"  # Label of saved figure

    # Display settings
    genelist = ["CD34", "GATA1", "MPO"]  # Genes to be displayed (must be equal to 3)

    two_dimension_manifold_plot = True  # Two dimensional phenotypic manifold plot
    three_dimension_manifold_plot = True  # Three dimensional phenotypic manifold plot
    gene_expression_plot = (
        True  # Gene expressions of two dimensional phenotypic manifold plot
    )
    bifurcation_plot = True  # Detected bifurcation points plot
    one_lineage_plot = True  # Plot one detected lineage at a time
    all_lineage_plot = True  # Plot all detected lineage
    frenet_frame_plot = True  # Plot Frenet frame representation
    GP_with_data_plot = True  # Gaussian process with data plot
    GP_only_plot = True  # Gaussian process only plot
    GP_per_lineage_plot = (
        True  # Plot one detected lineage with Gaussian process at a time
    )

    """
    -----------------------------------------------------------------
    Start modelling
    -----------------------------------------------------------------
    """
    if new_manifold_FLAG == True:
        joblib.dump(early_cell, "early_cell.pkl", compress=3)
        # Delete video folder
        parent_dir = os.path.dirname(os.path.realpath(__file__))
        resultPath = os.path.join(parent_dir, "VideoResultPictures")
        # delete old video folder if possible
        try:
            shutil.rmtree(resultPath)  # Delete
            if not os.path.exists(resultPath):
                os.makedirs(resultPath)
        except:
            if not os.path.exists(resultPath):
                os.makedirs(resultPath)

    if two_data_set_FLAG == False:
        Secondary_data_file = Main_data_file

    # Diffusion components of palantri algorithm
    diffusion_components = 5

    if results_only == False:
        # Dimensionality reduction and pseuod-time
        pseudo_time.palantir_pseudo_time(
            early_cell,
            diffusion_components,
            new_manifold_FLAG,
            Main_data_file,
            Secondary_data_file,
        )

        # Trajectory infernce
        # _, _ = lineages.lineages_estimater()
        lineages.lineages_estimater()

    # Plot results
    print("Results start.")
    D_resutls.results(
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
    )
    print("BAGEL -- END.")


if __name__ == "__main__":
    main()
