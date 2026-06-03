import numpy as np
import time
import traceback
from model_alpha_pipeline.sampling.sampler import sampler_master_function
from model_alpha_pipeline.physics.lens_builder import dlu_1
from model_alpha_pipeline.observations.observation_builder import make_bins
from model_alpha_pipeline.observations.observation_builder import dlu_2
from model_alpha_pipeline.simulation.simulation_builder import dlu_3



def simulation_child(DM_type,instrument,i,light_profile,observational_data_file,timestamp,results_h5_file):
    '''Runs all steps of a single simulation.'''
    sim_id = f"{DM_type}_{instrument}_{i}"
    while True:
        sampled_vals = sampler_master_function()

        try: 
            start0=time.time()           
            dlu_1_results,redshifts = dlu_1(DM_type,sampled_vals)
            
            sampled_vals.redshifts = redshifts #in case sample redshifts had to be altered slightly for dlu_1() to suceed

            hsc_redshifts = np.array([observational_data_file['specz_redshift']])[0]
            redshift_bin_edges = make_bins(hsc_redshifts)

            dlu_2_results = dlu_2(instrument,observational_data_file,sampled_vals.redshifts,redshift_bin_edges,light_profile=light_profile)
            
            dlu_3(i,DM_type,instrument,results_h5_file,timestamp,sampled_vals,dlu_1_results,dlu_2_results,light_profile=light_profile)

            end0=time.time()
            print(f'Sim took {end0-start0} secs')
            break

        except Exception as e:
            print(f"REDSHIFT z_scr={sampled_vals.redshifts[1]}, z_dfr={sampled_vals.redshifts[0]}, Δz={sampled_vals.redshifts[1] - sampled_vals.redshifts[0]}")
            print(f"ERROR in simulation {sim_id}")
            print("Exception repr:", repr(e))
            print("Exception str :", str(e))
            traceback.print_exc()
            del results_h5_file[f'images/strong_lens_{i}']
            results_h5_file.create_group(f'images/strong_lens_{i}')
            print("Trying a new simulation due to error")
            continue

