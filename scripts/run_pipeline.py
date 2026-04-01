import yaml
import h5py

from model_alpha_pipeline.pipeline.parent_pipeline import simulation_parent


def main():
    # Load config file
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Unpack run settings
    simulations_per_permutation = config["run"]["simulations_per_permutation"]
    instruments = config["run"]["instruments"]
    dm_types = config["run"]["dm_types"]
    output_dir = config["run"]["output_dir"]

    # Unpack data paths
    hsc_catalog_path = config["data"]["hsc_catalog"]
    
    # Open observational data
    with h5py.File(hsc_catalog_path, "r") as observational_data:
        simulation_parent(
             DM_Types=dm_types,
            instruments=instruments,
            sim_number_per_permutation=simulations_per_permutation,
            observational_data_file=observational_data,
            output_dir=output_dir
        )


if __name__ == "__main__":
    main()
