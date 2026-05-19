import os
import numpy as np
import time
import traceback
from model_alpha_pipeline.sampling.sampler import sampler_master_function
from model_alpha_pipeline.physics.lens_builder import dlu_1
from model_alpha_pipeline.observations.observation_builder import make_bins
from model_alpha_pipeline.observations.observation_builder import dlu_2
from model_alpha_pipeline.simulation.simulation_builder_mp import dlu_3_collect
from model_alpha_pipeline.simulation.output_writer_mp import collect_simulation_output


# Module-level handle, populated once per worker process by worker_init().
# Each worker opens its own read-only handle to the observational data file.
# Reading from HDF5 in parallel across separate processes is safe as long as
# no one is writing to the file.
_OBS_FILE = None


def worker_init(observational_data_path):
    """
    Pool initializer: runs once when each worker process starts.

    Opens the observational data file in read-only mode and stashes the handle
    in a module-level global so every simulation this worker runs can reuse it
    without re-opening.
    """
    global _OBS_FILE
    import h5py
    _OBS_FILE = h5py.File(observational_data_path, 'r')

    # Re-seed numpy RNG once per worker. Without this, fork-based workers all
    # inherit identical numpy random state from the parent.
    np.random.seed(os.getpid() % (2**32 - 1))


def simulation_child(DM_type, instrument, i, timestamp):
    """
    Runs all steps of a single simulation in a worker process.

    Uses the module-global _OBS_FILE handle opened by worker_init.
    Returns a 'collected' dict ready to be written to HDF5 by the parent.
    """
    global _OBS_FILE
    if _OBS_FILE is None:
        raise RuntimeError(
            "simulation_child called without worker_init having opened "
            "_OBS_FILE. If you're calling this outside a Pool, open the "
            "observational file and assign it to this module's _OBS_FILE "
            "global first."
        )

    sim_id = f"{DM_type}_{instrument}_{i}"

    # Mix sim index into RNG state so different sims on the same worker
    # don't repeat identical streams.
    np.random.seed((os.getpid() * 100003 + i) % (2**32 - 1))

    while True:
        sampled_vals = sampler_master_function()

        try:
            start0 = time.time()
            dlu_1_results, redshifts = dlu_1(DM_type, sampled_vals)
            sampled_vals.redshifts = redshifts

            hsc_redshifts = np.array(_OBS_FILE['specz_redshift'])
            redshift_bin_edges = make_bins(hsc_redshifts)

            # dlu_2's original signature, unchanged — it takes the open file.
            dlu_2_results = dlu_2(instrument, _OBS_FILE,
                                  sampled_vals.redshifts, redshift_bin_edges)

            # dlu_3 split: computation only, no h5 writes. See note below.
            setup_results, image_results = dlu_3_collect(
                DM_type, instrument, sampled_vals, dlu_1_results, dlu_2_results
            )

            collected = collect_simulation_output(
                i, DM_type, instrument, timestamp,
                sampled_vals, dlu_1_results, dlu_2_results,
                setup_results, image_results,
            )

            end0 = time.time()
            print(f'Sim {sim_id} took {end0 - start0:.1f} secs (pid={os.getpid()})')
            return collected

        except Exception as e:
            print(f"REDSHIFT z_scr={sampled_vals.redshifts[1]}, "
                  f"z_dfr={sampled_vals.redshifts[0]}, "
                  f"\u0394z={sampled_vals.redshifts[1] - sampled_vals.redshifts[0]}")
            print(f"ERROR in simulation {sim_id}")
            print("Exception repr:", repr(e))
            print("Exception str :", str(e))
            traceback.print_exc()
            print("Trying a new simulation due to error")
            # Nothing was written to disk, just retry in-memory.
            continue