import os
import datetime
import multiprocessing as mp
from functools import partial

import h5py

from new_pipeline_mp.pipeline.child_pipeline import (
    simulation_child,
    worker_init,
)
from new_pipeline_mp.simulation.output_writer import write_simulation_output


def _worker(args, timestamp, nss):
    """
    Top-level worker entrypoint. Unpacks the work tuple and runs one
    simulation. The observational data file handle is set up once per
    worker process by worker_init (see child_pipeline.py).
    """
    dm_type, instrument, light_profile,sim_index = args
    collected = simulation_child(dm_type, instrument, light_profile,sim_index, timestamp, nss=nss)
    return dm_type, instrument,sim_index, collected


def simulation_parent(DM_Types, instruments, sim_number_per_permutation,light_profile,
                      observational_data_path, output_dir, n_workers=None, nss=True):
    """
    Runs all simulations for a single run, parallelized across CPU cores
    with multiprocessing.Pool.

    Parameters
    ----------
    observational_data_path : str
        FILE PATH to the observational HDF5 file (not an open handle).
        Each worker process opens its own read-only handle to this file via
        the Pool initializer. The parent does not need to open it at all.
    n_workers : int, optional
        Defaults to cpu_count() - 1.
    """
    # Validate the path up front so we fail fast rather than inside every worker.
    if not os.path.exists(observational_data_path):
        raise FileNotFoundError(
            f"observational_data_path does not exist: {observational_data_path}"
        )

    timestamp = datetime.datetime.now().strftime("[%Y-%m-%d]")

    run_dir = f'./{output_dir}/model_alpha_{timestamp}'
    os.makedirs(run_dir, exist_ok=True)

    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)

    for dm_type in DM_Types:
        for instrument in instruments:
            output_file_name = f"model_alpha_{dm_type}_{instrument}_{timestamp}.h5"
            output_path = f'{run_dir}/{output_file_name}'

            mode = 'r+' if os.path.exists(output_path) else 'w'
            with h5py.File(output_path, mode) as results_h5_file:
                if 'images' not in results_h5_file:
                    results_h5_file.create_group('images')

                pending = [
                    i for i in range(1, sim_number_per_permutation + 1)
                    if f'images/strong_lens_{i}' not in results_h5_file
                ]
                if not pending:
                    print(f"All sims already complete for {dm_type}/{instrument}, skipping.")
                    continue

                print(f"Running {len(pending)} sims for {dm_type}/{instrument} "
                      f"on {n_workers} workers...")

                work_items = [(dm_type, instrument,light_profile, i) for i in pending]
                worker_fn = partial(_worker, timestamp=timestamp, nss=nss)

                # initializer runs once when each worker process starts,
                # opening the observational HDF5 file in read-only mode.
                # initargs is the tuple passed to that initializer.
                with mp.Pool(
                    processes=n_workers,
                    initializer=worker_init,
                    initargs=(observational_data_path,),
                ) as pool:
                    for dm_t, inst, sim_i, collected in pool.imap_unordered(
                            worker_fn, work_items, chunksize=1):
                        results_h5_file.create_group(f'images/strong_lens_{sim_i}')
                        write_simulation_output(results_h5_file, sim_i, collected)
                        results_h5_file.flush()
                        print(f"Wrote strong_lens_{sim_i} for {dm_t}/{inst}")