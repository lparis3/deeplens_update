import numpy as np

from model_alpha_pipeline_mp.observations.preprocessing import src_process


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


def extraction(obs_data,z_pair,redshift_bin_edges):
    '''Sample GalaxiesML Dataset for sources and deflectors.'''
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

    #Deflector Info
    deflector_morph = file['image'][idd]
    center_d = center_extraction(deflector_morph)
    sigma_y_d = file['g_half_light_radius'][idd]*2.
    sigma_x_d = file['g_half_light_radius'][idd]*2. * (1 - file['g_ellipticity'][idd])
    angle_d = np.pi/2. - np.deg2rad(file['g_pos_angle'][idd])
    g_mag_d = file['g_cmodel_mag'][idd]
    r_mag_d = file['r_cmodel_mag'][idd]
    i_mag_d = file['i_cmodel_mag'][idd]
    zdeflector = z_lens_ideal

    #Source Info:
    source_morph = file['image'][ids]
    center_s = center_extraction(source_morph)
    sigma_y_s = file['g_half_light_radius'][ids]*2.
    sigma_x_s = file['g_half_light_radius'][ids]*2. * (1 - file['g_ellipticity'][ids])
    angle_s = np.pi/2. - np.deg2rad(file['g_pos_angle'][ids])
    g_mag_s = file['g_cmodel_mag'][ids]
    r_mag_s = file['r_cmodel_mag'][ids]
    i_mag_s = file['i_cmodel_mag'][ids]
    zsource = z_src_ideal

    #Processing of source and deflector images:
    src_g = src_process(source_morph[0,:,:],center_s,sigma_x_s,sigma_y_s,angle_s)
    src_r = src_process(source_morph[1,:,:],center_s,sigma_x_s,sigma_y_s,angle_s)
    src_i = src_process(source_morph[2,:,:],center_s,sigma_x_s,sigma_y_s,angle_s)

    dfr_g = src_process(deflector_morph[0,:,:],center_d,sigma_x_d,sigma_y_d,angle_d)
    dfr_r = src_process(deflector_morph[1,:,:],center_d,sigma_x_d,sigma_y_d,angle_d)
    dfr_i = src_process(deflector_morph[2,:,:],center_d,sigma_x_d,sigma_y_d,angle_d)

    source_images = np.array([src_g,src_r,src_i]) #processed source morphology
    source_mag = np.array([g_mag_s,r_mag_s,i_mag_s])
    raw_info_src = {'raw_img': source_morph, 'pros_img': source_images, 'raw_center':center_s}

    deflector_images = np.array([dfr_g,dfr_r,dfr_i])
    deflector_mag = np.array([g_mag_d,r_mag_d,i_mag_d])
    raw_info_dfr = {'raw_img': deflector_morph, 'pros_img': deflector_images, 'raw_center':center_d}

    return source_images,source_mag,deflector_images,deflector_mag, raw_info_src, raw_info_dfr
