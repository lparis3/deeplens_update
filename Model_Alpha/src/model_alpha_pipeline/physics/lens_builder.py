import time
import numpy as np

from model_alpha_pipeline.structures.dataclasses import dlu_1_output
from model_alpha_pipeline.physics.halo_constructors import (
    cdm_constructor,
    wdm_constructor,
    sidm_constructor,
    axion_constructor,
)


def build_lens_system(dm_type, sampled_vals, interlopers=True):
    """
    Construct host halo, subhalos, and field halos realization using pyHalo.
    Refactored from dlu_1 without changing behavior.
    """
    start1 = time.time()

    # Unpack sampled_vals
    redshifts = sampled_vals.redshifts
    host_theta_e_arcsec = sampled_vals.host_theta_E_arcsecond
    m_host = sampled_vals.M_host
    max_subhalo_mass = sampled_vals.max_subhalo_mass

    if interlopers is True:
        los = 1.0
    else:
        los = 0.0

    def host_slope(gammaL=1.9, gammaH=2.2):
        """
        Samples host halo log slope between lower and upper bounds
        from Gilman et al. 2022.
        """
        return np.random.uniform(gammaL, gammaH, None)

    slope_host = host_slope()
    attempt = 1
    max_attempts = 10
    good = False

    while good is False and attempt <= max_attempts:
        try:
            print(
                f"Attempt:{attempt} with zsrc = {redshifts[1]}, "
                f"zdfr = {redshifts[0]},"
                f"log10_M_host = {np.log10(m_host)},"
                f"log10_M_submax = {np.log10(max_subhalo_mass)}"
            )

            zdeflector = redshifts[0]
            zsource = redshifts[1]

            if dm_type == "CDM":
                (
                    lens_model_list,
                    lens_kwargs_list,
                    lens_redshift_list,
                    cosmology,
                    macro_model_list,
                    macro_kwargs_list,
                    macro_redshift_list,
                    arcsecond_opening_angle,
                    host_mass,
                    whole_halo_mass,
                    num_subhalos,
                ) = cdm_constructor(
                    zsource=zsource,
                    zlens=zdeflector,
                    M_host=m_host,
                    host_theta_E_arcsec=host_theta_e_arcsec,
                    max_subhalo_mass=max_subhalo_mass,
                    Host_gamma=slope_host,
                    LOS_Norm=los,
                )

                results = DLU1Output(
                    lens_model_list=lens_model_list,
                    lens_kwargs_list=lens_kwargs_list,
                    lens_redshift_list=lens_redshift_list,
                    cosmology=cosmology,
                    macro_model_list=macro_model_list,
                    macro_kwargs_list=macro_kwargs_list,
                    macro_redshift_list=macro_redshift_list,
                    arcsecond_opening_angle=arcsecond_opening_angle,
                    host_mass=host_mass,
                    whole_halo_mass=whole_halo_mass,
                    num_subhalos=num_subhalos,
                    slope_Host=slope_host,
                    type_kwargs={},
                )
                end1 = time.time()
                print(f"Step 1 took {end1-start1} secs")
                return results, redshifts

            elif dm_type == "WDM":
                (
                    lens_model_list,
                    lens_kwargs_list,
                    lens_redshift_list,
                    cosmology,
                    macro_model_list,
                    macro_kwargs_list,
                    macro_redshift_list,
                    arcsecond_opening_angle,
                    log_mc,
                    host_mass,
                    whole_halo_mass,
                    num_subhalos,
                ) = wdm_constructor(
                    zsource=zsource,
                    zlens=zdeflector,
                    M_host=m_host,
                    host_theta_E_arcsec=host_theta_e_arcsec,
                    max_subhalo_mass=max_subhalo_mass,
                    Host_gamma=slope_host,
                    LOS_Norm=los,
                )

                results = DLU1Output(
                    lens_model_list=lens_model_list,
                    lens_kwargs_list=lens_kwargs_list,
                    lens_redshift_list=lens_redshift_list,
                    cosmology=cosmology,
                    macro_model_list=macro_model_list,
                    macro_kwargs_list=macro_kwargs_list,
                    macro_redshift_list=macro_redshift_list,
                    arcsecond_opening_angle=arcsecond_opening_angle,
                    host_mass=host_mass,
                    whole_halo_mass=whole_halo_mass,
                    num_subhalos=num_subhalos,
                    slope_Host=slope_host,
                    type_kwargs={"log_mc": log_mc},
                )
                end1 = time.time()
                print(f"Step 1 took {end1-start1} secs")
                return results, redshifts

            elif dm_type == "SIDM":
                (
                    lens_model_list,
                    lens_kwargs_list,
                    lens_redshift_list,
                    cosmology,
                    macro_model_list,
                    macro_kwargs_list,
                    macro_redshift_list,
                    mass_ranges_subhalos,
                    mass_ranges_field_halos,
                    probabilities_subhalos,
                    probabilities_field_halos,
                    arcsecond_opening_angle,
                    host_mass,
                    whole_halo_mass,
                    num_subhalos,
                ) = sidm_constructor(
                    zsource=zsource,
                    zlens=zdeflector,
                    M_host=m_host,
                    host_theta_E_arcsec=host_theta_e_arcsec,
                    max_subhalo_mass=max_subhalo_mass,
                    Host_gamma=slope_host,
                    LOS_Norm=los,
                )

                results = DLU1Output(
                    lens_model_list=lens_model_list,
                    lens_kwargs_list=lens_kwargs_list,
                    lens_redshift_list=lens_redshift_list,
                    cosmology=cosmology,
                    macro_model_list=macro_model_list,
                    macro_kwargs_list=macro_kwargs_list,
                    macro_redshift_list=macro_redshift_list,
                    arcsecond_opening_angle=arcsecond_opening_angle,
                    host_mass=host_mass,
                    whole_halo_mass=whole_halo_mass,
                    num_subhalos=num_subhalos,
                    slope_Host=slope_host,
                    type_kwargs={
                        "mass_ranges_subhalos": mass_ranges_subhalos,
                        "mass_ranges_field_halos": mass_ranges_field_halos,
                        "probabilities_subhalos": probabilities_subhalos,
                        "probabilities_field_halos": probabilities_field_halos,
                    },
                )
                end1 = time.time()
                print(f"Step 1 took {end1-start1} secs")
                return results, redshifts

            elif dm_type == "Axion":
                (
                    lens_model_list,
                    lens_kwargs_list,
                    lens_redshift_list,
                    cosmology,
                    macro_model_list,
                    macro_kwargs_list,
                    macro_redshift_list,
                    arcsecond_opening_angle,
                    M_axion,
                    flucs_shape,
                    flucs_args,
                    host_mass,
                    whole_halo_mass,
                    num_subhalos,
                ) = axion_constructor(
                    zsource=zsource,
                    zlens=zdeflector,
                    M_host=m_host,
                    host_theta_E_arcsec=host_theta_e_arcsec,
                    max_subhalo_mass=max_subhalo_mass,
                    Host_gamma=slope_host,
                    LOS_Norm=los,
                )

                results = DLU1Output(
                    lens_model_list=lens_model_list,
                    lens_kwargs_list=lens_kwargs_list,
                    lens_redshift_list=lens_redshift_list,
                    cosmology=cosmology,
                    macro_model_list=macro_model_list,
                    macro_kwargs_list=macro_kwargs_list,
                    macro_redshift_list=macro_redshift_list,
                    arcsecond_opening_angle=arcsecond_opening_angle,
                    host_mass=host_mass,
                    whole_halo_mass=whole_halo_mass,
                    num_subhalos=num_subhalos,
                    slope_Host=slope_host,
                    type_kwargs={
                        "M_axion": M_axion,
                        "flucs_shape": flucs_shape,
                        "flucs_args": flucs_args,
                    },
                )
                end1 = time.time()
                print(f"Step 1 took {end1-start1} secs")
                return results, redshifts

        except Exception as e:
            print(e)
            new_zsrc = redshifts[1] + np.random.uniform(0.25, 0.5)
            redshifts = np.array([redshifts[0], np.round(new_zsrc, 2)])
            print(f"Falied! Trying again with zsrc = {redshifts[1]}")
            attempt += 1
            good = False

    if not good:
        raise RuntimeError("All attempts failed to generate a valid model.")
