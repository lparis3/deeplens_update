#%%
import yaml
import h5py

from model_alpha_pipeline_mp.pipeline.parent_pipeline import simulation_parent
from model_alpha_pipeline_mp.paths.paths import PROJECT_ROOT

def main():
    # Load config file
    with open(PROJECT_ROOT/"configs"/"default.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Unpack run settings
    simulations_per_permutation = config["run"]["simulations_per_permutation"]
    n_workers = config["run"]["n_workers"]
    instruments = config["run"]["instruments"]
    dm_types = config["run"]["dm_types"]
    light_profile = config["run"]["light_profile"]
    output_dir = config["run"]["output_dir"] 

    # Unpack data paths
    hsc_catalog_path = PROJECT_ROOT/config["data"]["hsc_catalog"]
    
    # Open observational data
    simulation_parent(
            DM_Types=dm_types,
        instruments=instruments,
        sim_number_per_permutation=simulations_per_permutation,
        light_profile=light_profile,
        observational_data_path=hsc_catalog_path,
        output_dir=output_dir,
        n_workers=n_workers
    )


if __name__ == "__main__":
    main()

# %%
