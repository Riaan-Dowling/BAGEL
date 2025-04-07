# BAGEL: Bayesian Analysis of Gene Expression Lineages

BAGEL can be run by opening the *1. Main* directory and running the *run.py* file. This run will produce BAGEL's human bone marrow dataset results by default. Note: BAGEL can be controlled by adjusting it's parameters in the *config/bagel_config.json* file as seen below.

# 1.) MAIN/config/bagel_config.json

## interactive_mode_config
- turn_interactive_mode_on_flag : Should BAGEL run in interactive mode? (aka require human input)
- interactive_mode_on_config
    - total_simulations_before_assuming_continuous_manifold : How many phenotypic manifolds should be created before a continuous manifold is assumed. 

## output
- results_only : Only plot the results and do not run BAGEL. NOTE, this flag assumes that bagel has been previously run and the previous run's results is in the provided *output_name*.
- output_name : Directory prefix for BAGEL's run.

## phenotypic_manifold_config
- load_old_manifold : Should a previous manifold be loaded? True to create new manifold.
- data_cofig
    - main_data_config
        - main_data_file : Path to main data file.
        - cell_index_prefix : Prefix used to extract cell index from main data file e.g. Run
    - secondary_data_config
        - secondary_data_file: Path to secondary data file.
        - cell_index_prefix: Prefix used to extract cell index from secondary data file e.g. Run
- early_cell : User define early cell of *MAIN* dataset.
- palantir : Palantir algorithim parameters. NOTE, the default settings correspond to - Setty, M. et al. Characterization of cell fate probabilities in single-cell data with palantir. Nat. biotechnology 37, 451–465 (2019).
    - diffusion_components
    - _cell_min_molecules 
    - _genes_min_cells
    - n_components_pca
    - n_waypoints
    - pseudotime_knn

# bagel_loop_config
- load_old_bagel_loop : Should a previous BAGEL'S run be loaded? True to create new BAGEL'S run.
- window_interval_extra_samples : How many samples should be used to compute window interval.
- window_interval_actual_samples : Actual window interval size.
- gibbs_samples : Total gibbs samples.
- gibbs_burn_in_period : Gibbs sample burn-in-period.
- initialize_cells_offset

# plot_config
- primary_label : Lable to use for primary/ MAIN dataset plots.
- secondary_label : Lable to use for SECONDARY dataset plots.
- output_prefix_label : Plot save name prefix.
- gene_list : List of genes. 
- two_dimension_manifold_plot : True to create two dimensional manifold plots.
- three_dimension_manifold_plot : True to create three dimensional manifold plots.
- gene_expression_plot : True to create gene expression plots.
- bifurcation_plot : True to create bifurcation plots.
- one_lineage_plot : True to create per lineage plots (data only).
- all_lineage_plot : True to create all lineages plots.
- frenet_frame_plot : True to create Frenet frame plots.
- gp_with_data_plot : True to create GP with data plots.
- gp_only_plot : True to create GP only plots.
- gp_per_lineage_plot : True to create GP per lineage plots.

# 2. Data sets
Uploaded datasets
- human_UCB.csv : Umbilical cord blood (UCB) data.
- marrow_sample_scseq_counts.csv : Human bone marrow dataset: publicly available (Setty, M. et al. Characterization of cell fate probabilities in single-cell data with palantir. Nat. biotechnology 37, 451–465 (2019)).
- sample_scseq_data.csv : Mouse bone marrow dataset: publicly available (Paul, F. et al. Transcriptional heterogeneity and lineage commitment in myeloid progenitors. Cell 163, 1663–1677 (2015)).


# 3. Results - Figures
This folder contains all of the figures used in the paper.

