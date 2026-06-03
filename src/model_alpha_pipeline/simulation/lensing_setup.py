import numpy as np
from astropy import units as u
from astropy.constants import G, c, M_sun


def mass_to_radius_arcsec(Mass,redshift_src,redshift_def,dlu_1_results):
    M_kg = Mass * M_sun
    rad_to_arcsec = 206265

    DL = dlu_1_results.cosmology.luminosity_distance(redshift_def).to(u.m)
    DS = dlu_1_results.cosmology.luminosity_distance(redshift_src).to(u.m)
    DLS = DS - DL

    # Einstein radius
    theta = np.sqrt(4 * G * M_kg/c**2 * DLS/(DL*DS))

    # Return radius in arcsecods
    radius_arcsec = theta * rad_to_arcsec

    return radius_arcsec.value
    
def kwargs_light_mag(posxx,posyy,pixel_scale,dlu_2_results,light_profile='INTERPOL'):
    '''Build per-band light kwargs for lens and source.

    Parameters
    ----------
    light_profile : str
        'INTERPOL' : pixel-image-based interpolated light profile.
        'SERSIC'   : analytic SERSIC_ELLIPSE profile. Magnitude is per-band
                     from the HSC catalog (carried on dlu_2_results), while
                     R_sersic, n_sersic, e1, e2 follow the lens.py
                     convention.
    '''

    needed_hsc_bands = dlu_2_results.needed_hsc_bands
    kwargs_lens_light_mag = []
    kwargs_source_mag = []

    # dlu_2_results.{deflector_mag,source_mag} are ordered to match the
    # order of bands in needed_hsc_bands (see selection.extraction). We
    # iterate by enumerated band so the magnitude index matches that order
    # in both INTERPOL and SERSIC modes.
    for band_idx, band in enumerate([b for b in ['g', 'r', 'i', 'z'] if b in needed_hsc_bands]):

        if light_profile == 'INTERPOL':
            lens_kwargs = [{
                'magnitude': dlu_2_results.deflector_mag[band_idx],
                'image': dlu_2_results.deflector_images[band_idx],
                'center_x': 0.0, 'center_y': 0.0,
                'phi_G': 0.0, 'scale': pixel_scale,
            }]
            src_kwargs = [{
                'magnitude': dlu_2_results.source_mag[band_idx],
                'image': dlu_2_results.source_images[band_idx],
                'center_x': posxx, 'center_y': posyy,
                'phi_G': 0.0, 'scale': pixel_scale,
            }]

        elif light_profile == 'SERSIC':
            # Sersic shape parameters are per-galaxy (same across bands),
            # carried on dlu_2_results. Magnitudes are per-band from HSC.
            dfr_sp = dlu_2_results.deflector_sersic_params
            src_sp = dlu_2_results.source_sersic_params
            lens_kwargs = [{
                'magnitude': dlu_2_results.deflector_mag[band_idx],
                'R_sersic': dfr_sp['R_sersic'], 'n_sersic': dfr_sp['n_sersic'],
                'e1': dfr_sp['e1'], 'e2': dfr_sp['e2'],
                'center_x': 0.0, 'center_y': 0.0,
            }]
            src_kwargs = [{
                'magnitude': dlu_2_results.source_mag[band_idx],
                'R_sersic': src_sp['R_sersic'], 'n_sersic': src_sp['n_sersic'],
                'e1': src_sp['e1'], 'e2': src_sp['e2'],
                'center_x': posxx, 'center_y': posyy,
            }]

        else:
            raise ValueError(f"Unknown light_profile '{light_profile}'. Expected 'INTERPOL' or 'SERSIC'.")

        kwargs_lens_light_mag.append(lens_kwargs)
        kwargs_source_mag.append(src_kwargs)

    return kwargs_lens_light_mag, kwargs_source_mag
    

def build_lensing_setup(sampled_vals, dlu_1_results,dlu_2_results,light_profile='INTERPOL'):
    zdeflector = sampled_vals.redshifts[0]
    zsource = sampled_vals.redshifts[1]

    # Map the high-level light_profile flag to the lenstronomy model name.
    if light_profile == 'INTERPOL':
        light_model_name = 'INTERPOL'
    elif light_profile == 'SERSIC':
        light_model_name = 'SERSIC_ELLIPSE'
    else:
        raise ValueError(f"Unknown light_profile '{light_profile}'. Expected 'INTERPOL' or 'SERSIC'.")

    kwargs_model = {'lens_model_list': dlu_1_results.lens_model_list,  # list of lens models to be used
                'lens_redshift_list': dlu_1_results.lens_redshift_list,
                'lens_light_model_list': [light_model_name],  # list of unlensed light models to be used
                'source_light_model_list': [light_model_name],  # list of extended source models to be used
                'z_source':zsource,
                'cosmo': dlu_1_results.cosmology}

    theta_E_nss_arcsec = mass_to_radius_arcsec(dlu_1_results.whole_halo_mass,zsource,zdeflector,dlu_1_results)

    Host_kwargs_nss = {'theta_E':theta_E_nss_arcsec,'gamma': dlu_1_results.macro_kwargs_list[0]['gamma'],'e1':0.0,'e2':0.0,'center_x':0.0, 'center_y':0.0} 

    macro_kwargs_list_nss = [Host_kwargs_nss,dlu_1_results.macro_kwargs_list[1]]

    kwargs_model_nss = {'lens_model_list': dlu_1_results.macro_model_list,  # list of lens models to be used
                'lens_redshift_list': dlu_1_results.macro_redshift_list,
                'lens_light_model_list': [light_model_name],  # list of unlensed light models to be used
                'source_light_model_list': [light_model_name],  # list of extended source models to be used
                'z_source':zsource,
                'cosmo': dlu_1_results.cosmology}

    source_pos_xx,source_pos_yy = np.random.uniform(-dlu_1_results.macro_kwargs_list[0]['theta_E']*0.3, dlu_1_results.macro_kwargs_list[0]['theta_E']*0.3,None), np.random.uniform(-dlu_1_results.macro_kwargs_list[0]['theta_E']*0.3, dlu_1_results.macro_kwargs_list[0]['theta_E']*0.3,None)
    
    input_image_pix_scale = 0.168
    kwargs_lens_light_mag, kwargs_source_mag = kwargs_light_mag(source_pos_xx,source_pos_yy,input_image_pix_scale,dlu_2_results,light_profile=light_profile)

    return {
        "kwargs_model": kwargs_model,
        "kwargs_model_nss": kwargs_model_nss,
        "macro_kwargs_list_nss":macro_kwargs_list_nss,
        "kwargs_source_mag": kwargs_source_mag,
        "kwargs_lens_light_mag": kwargs_lens_light_mag,
        "source_x": source_pos_xx,
        "source_y": source_pos_yy,
    }