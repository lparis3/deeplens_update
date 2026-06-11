import numpy as np

from new_pipeline_mp.observations.preprocessing import src_process


def center_extraction(image):
    brightest_pixel_values = np.array([np.max(image[0,:,:]), np.max(image[1,:,:]),np.max(image[2,:,:])])
    brightest_band = brightest_pixel_values.argmax()
    center_rows_columns = np.unravel_index(image[brightest_band,:,:].argmax(),image[brightest_band,:,:].shape)
    center_pixel_coor = np.array([center_rows_columns[1], center_rows_columns[0]])
    return center_pixel_coor


def z_to_bin(z,bin_edges):
    if len(bin_edges[bin_edges<=z]) == 0:
        bin_lower =  bin_edges[0]
        bin_upper = bin_edges[1]
    elif len(bin_edges[bin_edges>=z]) == 0:
        bin_lower = bin_edges[-2]
        bin_upper = bin_edges[-1]
    else:
        bin_lower = bin_edges[bin_edges<=z][-1]
        bin_upper = bin_edges[bin_edges>=z][0]
    return np.array([bin_lower,bin_upper])


def extraction(obs_data,z_pair,redshift_bin_edges,needed_hsc_bands,light_profile='INTERPOL'):
    '''Sample GalaxiesML Dataset for sources and deflectors.

    Parameters
    ----------
    light_profile : str
        'INTERPOL' (default): extract and process pixel images from the HSC
        catalog to use as interpolated light profiles.
        'SERSIC': skip pixel-image processing and only pull per-band
        magnitudes from the catalog. The Sersic shape parameters are then
        set downstream (see kwargs_light_mag).
    '''
    file = obs_data
    redshifts = np.array([file['specz_redshift']])[0]

    z_lens_ideal = z_pair[0]
    z_src_ideal = z_pair[1]

    z_lens_bin = z_to_bin(z_lens_ideal,redshift_bin_edges)
    z_src_bin = z_to_bin(z_src_ideal,redshift_bin_edges)

    #Deflector id
    idd_array = np.where((redshifts >= z_lens_bin[0]) & (redshifts <= z_lens_bin[1]))[0]
    idd = np.random.choice(idd_array)

    #Source id
    ids_array = np.where((redshifts >= z_src_bin[0])&(redshifts <= z_src_bin[1]))[0]
    ids = np.random.choice(ids_array)

    #Per-band magnitudes (needed for both INTERPOL and SERSIC modes)
    g_mag_d = file['g_cmodel_mag'][idd]
    r_mag_d = file['r_cmodel_mag'][idd]
    i_mag_d = file['i_cmodel_mag'][idd]
    z_mag_d = file['z_cmodel_mag'][idd]

    g_mag_s = file['g_cmodel_mag'][ids]
    r_mag_s = file['r_cmodel_mag'][ids]
    i_mag_s = file['i_cmodel_mag'][ids]
    z_mag_s = file['z_cmodel_mag'][ids]

    #Raw morphology images (kept for reference even in SERSIC mode)
    deflector_morph = file['image'][idd]
    source_morph = file['image'][ids]

    band_mag_map_d = {'g': g_mag_d, 'r': r_mag_d, 'i': i_mag_d, 'z': z_mag_d}
    band_mag_map_s = {'g': g_mag_s, 'r': r_mag_s, 'i': i_mag_s, 'z': z_mag_s}
    band_idx_map  = {'g': 0, 'r': 1, 'i': 2, 'z': 3}

    source_mag = []
    deflector_mag = []

    # Per-galaxy Sersic shape parameters. Populated in SERSIC mode only;
    # None in INTERPOL mode. Stored as scalars (one value per galaxy)
    # because Sersic shape is the same across bands. Hardcoded for now
    # to the lens.py convention; swap in a draw / catalog read later.
    source_sersic_params = None
    deflector_sersic_params = None

    if light_profile == 'INTERPOL':
        #Shape parameters needed only for pixel-image processing
        center_d = center_extraction(deflector_morph)
        sigma_y_d = file['g_half_light_radius'][idd]*2.
        sigma_x_d = file['g_half_light_radius'][idd]*2. * (1 - file['g_ellipticity'][idd])
        angle_d = np.pi/2. - np.deg2rad(file['g_pos_angle'][idd])

        center_s = center_extraction(source_morph)
        sigma_y_s = file['g_half_light_radius'][ids]*2.
        sigma_x_s = file['g_half_light_radius'][ids]*2. * (1 - file['g_ellipticity'][ids])
        angle_s = np.pi/2. - np.deg2rad(file['g_pos_angle'][ids])

        source_images = []
        deflector_images = []
        for band in ['g', 'r', 'i', 'z']:
            if band in needed_hsc_bands:
                bidx = band_idx_map[band]
                source_images.append(src_process(source_morph[bidx,:,:],center_s,sigma_x_s,sigma_y_s,angle_s))
                source_mag.append(band_mag_map_s[band])
                deflector_images.append(src_process(deflector_morph[bidx,:,:],center_d,sigma_x_d,sigma_y_d,angle_d))
                deflector_mag.append(band_mag_map_d[band])

        raw_info_src = {'raw_img': source_morph, 'pros_img': source_images, 'raw_center':center_s}
        raw_info_dfr = {'raw_img': deflector_morph, 'pros_img': deflector_images, 'raw_center':center_d}

    elif light_profile == 'SERSIC':
        #Only magnitudes are needed; shape params are set in kwargs_light_mag
        source_images = None
        deflector_images = None
        for band in ['g', 'r', 'i', 'z']:
            if band in needed_hsc_bands:
                source_mag.append(band_mag_map_s[band])
                deflector_mag.append(band_mag_map_d[band])

        # Per-galaxy Sersic shape params. Fixed for now (lens.py defaults),
        # but stored per galaxy so they can later be drawn from a
        # distribution or pulled from the catalog without touching
        # downstream code.
        source_sersic_params = {
            'R_sersic': 0.25, 'n_sersic': 1,
            'e1': -0.1, 'e2': 0.1,
        }
        deflector_sersic_params = {
            'R_sersic': 0.25, 'n_sersic': 1,
            'e1': -0.1, 'e2': 0.1,
        }

        # No pixel processing in SERSIC mode, so nothing meaningful to
        # carry in the raw-info dicts. The original catalog cutouts can
        # always be re-fetched from obs_data via the galaxy ids if
        # needed for diagnostics.
        raw_info_src = None
        raw_info_dfr = None

    else:
        raise ValueError(f"Unknown light_profile '{light_profile}'. Expected 'INTERPOL' or 'SERSIC'.")

    return (source_images, source_mag, deflector_images, deflector_mag,
            raw_info_src, raw_info_dfr,
            source_sersic_params, deflector_sersic_params)