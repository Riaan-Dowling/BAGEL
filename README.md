# BAGEL: Bayesian Analysis of Gene Expression Lineages

BAGEL can be run by opening the 1. Algorithm directory and running the Algorithm.py file. This will produce BAGELS mouse bone marrow dataset results. To change the output of BAGEL / how the modelling is done the following parameters can be set:

# Model parameters:
## 1.) Algorithm/Algorithm.py
### Moddeling options
- new_manifold_FLAG = True  # Start with a new manifold? This flag should only be set to True the first time that BAGEL obtains the phenotypic manifold of the current dataset/s. After this initial run, BAGEL stored the current phenotypic manifold and this flag can be set to False.
- two_data_set_FLAG = False  # Is two diferent datasets used? Is there one (Flase) or Two (True) datasets used as input to BAGEL
- results_only = False  # Only display results? Can be set to true after BAGEL's initial run, as all of the modelling data has been saved locally.

### Dataset options
#### 1) Results: Mouse bone marrow
- Main_data_file = "sample_scseq_data.csv"  # Mouse
- early_cell = "W30258"

#### 2)  Results: Human bone marrow
- Main_data_file = 'marrow_sample_scseq_counts.csv.gz' #Human 1
- early_cell = 'Run5_164698952452459'

#### 3)  Results: Projection of human UCB onto human bone marrow
- Main_data_file = 'marrow_sample_scseq_counts.csv.gz' #Human 1
- Secondary_data_file = 'human_UCB.csv' #Human 2
- early_cell = 'Run5_164698952452459'

#### 4)  Results: Projection of human UCB onto mouse bone marrow
- Main_data_file = 'sample_scseq_data.csv' #Mouse
- Secondary_data_file = 'human_UCB.csv' #Human 2
- early_cell = 'W30258'

### Results
- Primary_label = "Human bone marrow"  # Primary data label
- Secondary_label = "Human UCB"  # Secondary data label
- Picture_name = "human_bone_marrow_and_human_UCB"  # Prefix of results
- genelist = ["CD34", "GATA1", "MPO"]  # Genes to be displayed (must be equal to 3)


## 1.) Algorithm/lineage_parameters
### Window parameters                                   
- pseudo_time_interval = 200  # How much cells per step (pseudo_time_interval)  
- window_interval = 150  # How many cells in window  (window_interval)

### Gibbs sampler parameters
- itterations = 2000  # Total itterations
- burn_period = 500  # Total itterations for "burn in period"

