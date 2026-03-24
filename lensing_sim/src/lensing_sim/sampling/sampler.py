import numpy as np
from lensing_sim.sampling.lenspop import load_lenspop_flow, sample_lenspop
from lensing_sim.sampling.camels import load_camels_flow, sample_camels
from lensing_sim.physics.mass_conversion import lensing_params_to_gnfw_M200

def sampler_master_function(lenspop_ckpt, camels_ckpt):
    Lenspop_flow, Lenspop_mean, Lenspop_std = load_lenspop_flow(lenspop_ckpt)
    Camels_flow, camels_ckpt_data = load_camels_flow(camels_ckpt)

    good = False
    while not good:
        Lenspop_sample = sample_lenspop(Lenspop_flow, Lenspop_mean, Lenspop_std, 1)
        redshift_pairs = np.round(Lenspop_sample[0, 0:2], 2)
        if redshift_pairs[1] - redshift_pairs[0] >= 0.25 and redshift_pairs[1] <= 4:
            good = True

    theta_E_sample = np.round(Lenspop_sample[0, -1], 2)
    M_host_sample = lensing_params_to_gnfw_M200(Lenspop_sample[0])
    log10_M_host = np.log10(M_host_sample / 10**10)

    good = False
    while not good:
        m_sub_max_sample = sample_camels(Camels_flow, camels_ckpt_data, log10_M_host, 1).flatten()[0] * 10**10
        if np.log10(m_sub_max_sample) > 6 and np.log10(m_sub_max_sample) < np.log10(M_host_sample):
            good = True

    return dict(
        redshifts=redshift_pairs,
        host_theta_E_arcsecond=theta_E_sample,
        M_host=M_host_sample,
        max_subhalo_mass=m_sub_max_sample
    )
