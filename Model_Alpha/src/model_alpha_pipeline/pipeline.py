import h5py
from model_alpha_pipeline.sampling.sampler import sampler_master_function
from model_alpha_pipeline.physics.lens_builder import dlu_1
from model_alpha_pipeline.observations.observation_builder import dlu_2
from model_alpha_pipeline.simulation.simulation_builder import dlu_3



def simulation_child(DM_Type,instrument,i,observational_data_file,time_stamp,results_h5_file):
    '''Runs all steps of a single simulation.'''
    sim_id = f"{DM_Type}_{instrument}_{i}"
    start_time = time.time()

    while True:
        sampled_vals = sampler_master_function()

        try:            
            dlu_1_results,redshifts = dlu_1(DM_Type,sampled_vals)
            
            sampled_vals.redshifts = redshifts #in case sample redshifts had to be altered slightly for dlu_1() to suceed

            hsc_redshifts = np.array([observational_data_file['specz_redshift']])[0]
            redshift_bin_edges = make_bins(hsc_redshifts)

            dlu_2_results = dlu_2(instrument,observational_data_file,sampled_vals.redshifts,redshift_bin_edges)
            
            dlu_3(DM_Type,instrument,sampled_vals,dlu_1_results,dlu_2_results,start_time,time_stamp,i,results_h5_file)

            break

        except Exception as e:
            print(f"REDSHIFT z_scr={sampled_vals.redshifts[1]}, z_dfr={sampled_vals.redshifts[0]}, Δz={sampled_vals.redshifts[1] - sampled_vals.redshifts[0]}")
            print(f"ERROR in simulation {sim_id}: {e}")
            print("Trying a new simulation due to error")
            del results_h5_file[f'images/strong_lens_{i}']
            results_h5_file.create_group(f'images/strong_lens_{i}')
            continue
