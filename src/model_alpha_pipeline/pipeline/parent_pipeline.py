import h5py
import os
import datetime
from model_alpha_pipeline.pipeline.child_pipeline import simulation_child


def simulation_parent(DM_Types,instruments,sim_number_per_permutation,observational_data_file,output_dir):
    '''Runs all simulations for a single run.'''

    #Create timestamp for when this batch of sims is called
    timestamp = datetime.datetime.now().strftime("[%Y-%m-%d]")

    #Create directory to store result data
    if not os.path.exists(f'./{output_dir}/model_alpha_{timestamp}'):
        os.makedirs(f'./{output_dir}/model_alpha_{timestamp}')

    for type in DM_Types:
        for instrument in instruments:
            output_file_name = f"model_alpha_{type}_{instrument}_{timestamp}.h5" 
            output_path = f'./{output_dir}/model_alpha_{timestamp}/{output_file_name}' #Create file in new directory corresponding to a particular permutation of (DM_Type, instrument)
            if os.path.exists(output_path):
                with h5py.File(f'./{output_dir}/model_alpha_{timestamp}/{output_file_name}','r+') as results_h5_file:
                    for i in range(1,sim_number_per_permutation+1):
                        if f'images/strong_lens_{i}' not in results_h5_file:
                            results_h5_file.create_group(f'images/strong_lens_{i}')
                            args=[type,instrument,i,observational_data_file,timestamp,results_h5_file]
                            simulation_child(*args)
            else:
                with h5py.File(f'./{output_dir}/model_alpha_{timestamp}/{output_file_name}','w') as results_h5_file:
                    results_h5_file.create_group('images')
                    for i in range(1,sim_number_per_permutation+1):
                        results_h5_file.create_group(f'images/strong_lens_{i}')
                        args=[type,instrument,i,observational_data_file,timestamp,results_h5_file]
                        simulation_child(*args)



