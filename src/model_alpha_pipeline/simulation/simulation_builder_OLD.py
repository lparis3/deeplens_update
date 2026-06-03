# src/model_alpha_pipeline/simulation/simulation_builder.py
import time

from model_alpha_pipeline.simulation.lensing_setup import build_lensing_setup
from model_alpha_pipeline.simulation.image_generation import generate_images
from model_alpha_pipeline.simulation.output_writer import write_simulation_output

def dlu_3(i, DM_type,Instrument,hf, timestamp,sampled_vals, dlu_1_results, dlu_2_results):
    
    start3 = time.time()
    # 1. prepare lensing kwargs and setup
    setup_results = build_lensing_setup(
        sampled_vals=sampled_vals,
        dlu_1_results=dlu_1_results,
        dlu_2_results=dlu_2_results,
    )

    # 2. render images
    image_results = generate_images(
        setup_results=setup_results,
        dlu_1_results=dlu_1_results,
        dlu_2_results=dlu_2_results,
    )

    # 3. write outputs to file
    write_simulation_output(
        i=i,
        DM_type = DM_type,
        Instrument=Instrument,
        hf=hf,
        timestamp = timestamp,
        sampled_vals=sampled_vals,
        dlu_1_results=dlu_1_results,
        dlu_2_results=dlu_2_results,
        setup_results = setup_results,
        image_results=image_results,
    )
    end3=time.time()
    print(f'Step 3 took {end3-start3} secs')

    return image_results
