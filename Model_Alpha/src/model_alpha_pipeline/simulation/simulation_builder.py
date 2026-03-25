# src/model_alpha_pipeline/simulation/simulation_builder.py

from model_alpha_pipeline.simulation.lensing_setup import build_lensing_setup
from model_alpha_pipeline.simulation.image_generation import generate_images
from model_alpha_pipeline.simulation.output_writer import write_simulation_output

def dlu_3(i, Instrument, hf, sampled_vals, dlu_1_results, dlu_2_results):
    # 1. prepare lensing kwargs and setup
    setup_results = build_lensing_setup(
        sampled_vals=sampled_vals,
        dlu_1_results=dlu_1_results,
        dlu_2_results=dlu_2_results,
    )

    # 2. render images
    image_results = generate_images(
        setup_results=setup_results,
        dlu_2_results=dlu_2_results,
    )

    # 3. write outputs to file
    write_simulation_output(
        hf=hf,
        i=i,
        Instrument=Instrument,
        sampled_vals=sampled_vals,
        dlu_1_results=dlu_1_results,
        dlu_2_results=dlu_2_results,
        image_results=image_results,
    )

    return image_results
