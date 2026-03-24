import datetime, os, time, h5py
from lensing_sim.sampling.sampler import sampler_master_function
from lensing_sim.utils.math_utils import make_bins
from lensing_sim.observation.dataset import dlu_2
from lensing_sim.physics.halo_construction import dlu_1
from lensing_sim.simulation.image_sim import dlu_3

def simulation_child(DM_Type, instrument, i, observational_data_file, time_stamp, results_h5_file, cfg):
    start_time = time.time()

    while True:
        sampled_vals = sampler_master_function(cfg['lenspop'], cfg['camels'])

        try:
            dlu_1_results, redshifts = dlu_1(DM_Type, sampled_vals)
            sampled_vals['redshifts'] = redshifts

            hsc_redshifts = observational_data_file['specz_redshift'][:]
            redshift_bin_edges = make_bins(hsc_redshifts)

            dlu_2_results = dlu_2(instrument, observational_data_file, sampled_vals['redshifts'], redshift_bin_edges)

            dlu_3(DM_Type, instrument, sampled_vals, dlu_1_results, dlu_2_results, start_time, time_stamp, i, results_h5_file)
            break

        except Exception as e:
            print(f"Retrying simulation {i}: {e}")
            del results_h5_file[f'images/strong_lens_{i}']
            results_h5_file.create_group(f'images/strong_lens_{i}')


def simulation_parent(DM_Types, instruments, sim_number_per_permutation, observational_data_file, cfg):
    timestamp = datetime.datetime.now().strftime("[%Y-%m-%d]")

    if not os.path.exists(f'model_alpha_{timestamp}'):
        os.mkdir(f'model_alpha_{timestamp}')

    for type_ in DM_Types:
        for instrument in instruments:
            output_file = f'model_alpha_{type_}_{instrument}_{timestamp}.h5'
            path = f'./model_alpha_{timestamp}/{output_file}'

            with h5py.File(path, 'w') as hf:
                hf.create_group('images')
                for i in range(1, sim_number_per_permutation + 1):
                    hf.create_group(f'images/strong_lens_{i}')
                    simulation_child(type_, instrument, i, observational_data_file, timestamp, hf, cfg)

