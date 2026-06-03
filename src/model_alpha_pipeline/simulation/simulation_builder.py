# src/model_alpha_pipeline_mp/simulation/simulation_builder.py
import time

from model_alpha_pipeline.simulation.lensing_setup import build_lensing_setup
from model_alpha_pipeline.simulation.image_generation import generate_images
from model_alpha_pipeline.simulation.output_writer import write_simulation_output


def dlu_3_collect(DM_type, Instrument, sampled_vals, dlu_1_results, dlu_2_results,
                  light_profile='INTERPOL'):
    """
    Computation half of the original dlu_3: builds the lensing setup and
    renders images, but does NOT write anything to HDF5.

    Parameters
    ----------
    light_profile : str
        'INTERPOL' (default) or 'SERSIC'. Passed through to
        build_lensing_setup so the lens/source light model lists and
        kwargs match the chosen profile.

    Returns
    -------
    setup_results : object
        Output of build_lensing_setup; consumed by collect_simulation_output
        for things like source_x, source_y.
    image_results : object
        Output of generate_images; consumed by collect_simulation_output for
        the rendered image arrays, exposure times, SNR, etc.

    This is what worker processes call. Designed to be safe across processes:
    no shared file handles, no global state mutations, just pure computation
    given the inputs.
    """
    start3 = time.time()

    setup_results = build_lensing_setup(
        sampled_vals=sampled_vals,
        dlu_1_results=dlu_1_results,
        dlu_2_results=dlu_2_results,
        light_profile=light_profile,
    )

    image_results = generate_images(Instrument,
        setup_results=setup_results,
        dlu_1_results=dlu_1_results,
        dlu_2_results=dlu_2_results,
    )

    end3 = time.time()
    #print(f'Step 3 took {end3 - start3:.2f} secs')

    return setup_results, image_results


def dlu_3(i, DM_type, Instrument, hf, timestamp,
          sampled_vals, dlu_1_results, dlu_2_results,
          light_profile='INTERPOL'):
    """
    Legacy single-process entrypoint, preserved so any non-parallel callers
    of dlu_3 still work.

    Runs the computation via dlu_3_collect, then writes results to the given
    open HDF5 file handle using the new collect+write split in output_writer.
    """
    from model_alpha_pipeline_mp.simulation.output_writer import collect_simulation_output

    setup_results, image_results = dlu_3_collect(
        DM_type=DM_type,
        Instrument=Instrument,
        sampled_vals=sampled_vals,
        dlu_1_results=dlu_1_results,
        dlu_2_results=dlu_2_results,
        light_profile=light_profile,
    )

    collected = collect_simulation_output(
        i=i,
        DM_type=DM_type,
        Instrument=Instrument,
        timestamp=timestamp,
        sampled_vals=sampled_vals,
        dlu_1_results=dlu_1_results,
        dlu_2_results=dlu_2_results,
        setup_results=setup_results,
        image_results=image_results,
    )

    write_simulation_output(hf, i, collected)

    return image_results