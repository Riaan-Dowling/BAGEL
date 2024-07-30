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
    main_data_file = bagel_config["phenotypic_manifold_config"]["data_cofig"][
        "main_data_config"
    ]["main_data_file"]
    main_cell_index_prefix = bagel_config["phenotypic_manifold_config"]["data_cofig"][
        "main_data_config"
    ]["cell_index_prefix"]

    secondary_data_file = bagel_config["phenotypic_manifold_config"]["data_cofig"][
        "secondary_data_config"
    ]["secondary_data_file"]
    secondary_cell_index_prefix = bagel_config["phenotypic_manifold_config"][
        "data_cofig"
    ]["secondary_data_config"]["cell_index_prefix"]

    early_cell = bagel_config["phenotypic_manifold_config"]["early_cell"]

    # """
    # -----------------------------------------------------------------
    # START BAGEL
    # -----------------------------------------------------------------
    # """

    output_version_no = bagel_config["output"]["output_name"]
    bagel_object = bagel_class_script.BAGEL(
        bagel_config,
        load_old_manifold,
        results_only,
        main_data_file,
        secondary_data_file,
        early_cell,
        output_version_no,
        main_cell_index_prefix,
        secondary_cell_index_prefix,
    )
    bagel_object.create_processing_dir()
    bagel_object.load_datasets()

    if results_only is False:
        bagel_object.load_phenotypic_manifold()
        bagel_object.bagel_loop()
    bagel_object.plot()
    print("BAGEL -- END.")


if __name__ == "__main__":
    run_bagel()
