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
    

def kwargs_light_mag(posxx,posyy,pixel_scale,dlu_2_results):
    '''Returns kwargs_single_band values all bands in dlu_2_results.needed_hsc_bans while taking into account intrument of simulated image.'''
    
    needed_hsc_bands = dlu_2_results.needed_hsc_bands
    kwargs_lens_light_mag = []
    kwargs_source_mag = []

    if 'g' in needed_hsc_bands:
        # g-band
        # lens light
        kwargs_lens_light_mag_g = [{'magnitude': dlu_2_results.deflector_mag[0], 'image':dlu_2_results.deflector_images[0],'center_x': 0.0, 'center_y': 0.0, 'phi_G': 0.0, 'scale': pixel_scale}]
        # source light
        kwargs_source_mag_g = [{'magnitude': dlu_2_results.source_mag[0], 'image': dlu_2_results.source_images[0], 'center_x': posxx, 'center_y': posyy, 'phi_G': 0.0, 'scale': pixel_scale}]
        kwargs_lens_light_mag.append(kwargs_lens_light_mag_g)
        kwargs_source_mag.append(kwargs_source_mag_g)

    if 'r' in needed_hsc_bands:
        # r-band
        # lens light
        kwargs_lens_light_mag_r = [{'magnitude': dlu_2_results.deflector_mag[1], 'image':dlu_2_results.deflector_images[1],'center_x': 0.0, 'center_y': 0.0, 'phi_G': 0.0, 'scale': pixel_scale}]
        # source light
        kwargs_source_mag_r = [{'magnitude': dlu_2_results.source_mag[1], 'image': dlu_2_results.source_images[1], 'center_x': posxx, 'center_y': posyy, 'phi_G': 0.0, 'scale': pixel_scale}]
        kwargs_lens_light_mag.append(kwargs_lens_light_mag_r)
        kwargs_source_mag.append(kwargs_source_mag_r)

    if 'i' in needed_hsc_bands:
        # i-band
        # lens light
        kwargs_lens_light_mag_i = [{'magnitude': dlu_2_results.deflector_mag[2], 'image':dlu_2_results.deflector_images[2],'center_x': 0.0, 'center_y': 0.0, 'phi_G': 0.0, 'scale': pixel_scale}]
        # source light
        kwargs_source_mag_i =[{'magnitude': dlu_2_results.source_mag[2], 'image': dlu_2_results.source_images[2], 'center_x': posxx, 'center_y':posyy, 'phi_G': 0.0, 'scale': pixel_scale}]
        kwargs_lens_light_mag.append(kwargs_lens_light_mag_i)
        kwargs_source_mag.append(kwargs_source_mag_i)

    if 'z' in needed_hsc_bands:
        # z-band
        # lens light
        kwargs_lens_light_mag_z = [{'magnitude': dlu_2_results.deflector_mag[3], 'image':dlu_2_results.deflector_images[3],'center_x': 0.0, 'center_y': 0.0, 'phi_G': 0.0, 'scale': pixel_scale}]
        # source light
        kwargs_source_mag_z =[{'magnitude': dlu_2_results.source_mag[3], 'image': dlu_2_results.source_images[3], 'center_x': posxx, 'center_y':posyy, 'phi_G': 0.0, 'scale': pixel_scale}]
        kwargs_lens_light_mag.append(kwargs_lens_light_mag_z)
        kwargs_source_mag.append(kwargs_source_mag_z)

    return kwargs_lens_light_mag, kwargs_source_mag

    

def build_lensing_setup(sampled_vals, dlu_1_results,dlu_2_results):
    zdeflector = sampled_vals.redshifts[0]
    zsource = sampled_vals.redshifts[1]

    kwargs_model = {'lens_model_list': dlu_1_results.lens_model_list,  # list of lens models to be used
                'lens_redshift_list': dlu_1_results.lens_redshift_list,
                'lens_light_model_list': ['INTERPOL'],  # list of unlensed light models to be used
                'source_light_model_list': ['INTERPOL'],  # list of extended source models to be used, here we used the interpolated real galaxy
                'z_source':zsource,
                'cosmo': dlu_1_results.cosmology}

    theta_E_nss_arcsec = mass_to_radius_arcsec(dlu_1_results.whole_halo_mass,zsource,zdeflector,dlu_1_results)

    Host_kwargs_nss = {'theta_E':theta_E_nss_arcsec,'gamma': dlu_1_results.macro_kwargs_list[0]['gamma'],'e1':0.0,'e2':0.0,'center_x':0.0, 'center_y':0.0} 

    macro_kwargs_list_nss = [Host_kwargs_nss,dlu_1_results.macro_kwargs_list[1]]

    kwargs_model_nss = {'lens_model_list': dlu_1_results.macro_model_list,  # list of lens models to be used
                'lens_redshift_list': dlu_1_results.macro_redshift_list,
                'lens_light_model_list': ['INTERPOL'],  # list of unlensed light models to be used
                'source_light_model_list': ['INTERPOL'],  # list of extended source models to be used, here we used the interpolated real galaxy
                'z_source':zsource,
                'cosmo': dlu_1_results.cosmology}

    source_pos_xx,source_pos_yy = np.random.uniform(-dlu_1_results.macro_kwargs_list[0]['theta_E']*0.3, dlu_1_results.macro_kwargs_list[0]['theta_E']*0.3,None), np.random.uniform(-dlu_1_results.macro_kwargs_list[0]['theta_E']*0.3, dlu_1_results.macro_kwargs_list[0]['theta_E']*0.3,None)
    
    input_image_pix_scale = 0.168
    kwargs_lens_light_mag, kwargs_source_mag = kwargs_light_mag(source_pos_xx,source_pos_yy,input_image_pix_scale,dlu_2_results)

    return {
        "kwargs_model": kwargs_model,
        "kwargs_model_nss": kwargs_model_nss,
        "macro_kwargs_list_nss":macro_kwargs_list_nss,
        "kwargs_source_mag": kwargs_source_mag,
        "kwargs_lens_light_mag": kwargs_lens_light_mag,
        "source_x": source_pos_xx,
        "source_y": source_pos_yy,
    }
