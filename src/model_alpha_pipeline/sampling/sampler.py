import numpy as np

from model_alpha_pipeline.structures.dataclasses import sampled_values 
from model_alpha_pipeline.sampling.lenspop_sampler import (
    load_lenspop_flow,
    sample_lenspop
)
from model_alpha_pipeline.sampling.camels_sampler import (
    load_camels_flow,
    sample_camels
)
from model_alpha_pipeline.physics.mass_conversion import lensing_params_to_gnfw_m200


def sampler_master_function():
    """
    Sample all lensing parameters needed for one simulation.
    """
    lenspop_flow, lenspop_mean, lenspop_std = load_lenspop_flow()
    camels_flow, camels_condition_mean, camels_condition_std, camels_dependent_mean, camels_dependent_std = load_camels_flow()

    good = False
    while not good:
        lenspop_sample = sample_lenspop(lenspop_flow, lenspop_mean, lenspop_std, 1)
        redshift_pairs = np.round(lenspop_sample[0, 0:2], 2)

        if redshift_pairs[1] - redshift_pairs[0] >= 0.25 and redshift_pairs[1] <= 4:
            good = True

    theta_e_sample = np.round(lenspop_sample[0, -1], 2)
    m_host_sample = lensing_params_to_gnfw_m200(lenspop_sample[0])
    log10_m_host = np.log10(m_host_sample / 10**10)

    good = False
    while not good:
        m_sub_max_sample = (
            sample_camels(
                camels_flow,
                camels_condition_mean,
                camels_condition_std,
                camels_dependent_mean,
                camels_dependent_std,
                log10_m_host,
                1,
            ).flatten()[0]
            * 10**10
        )

        if np.log10(m_sub_max_sample) > 6 and np.log10(m_sub_max_sample) < np.log10(m_host_sample):
            good = True

    return sampled_values(
        redshifts=redshift_pairs,
        host_theta_E_arcsecond=theta_e_sample,
        M_host=m_host_sample,
        max_subhalo_mass=m_sub_max_sample,
    )
