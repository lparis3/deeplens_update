import numpy as np

def simulate(kwargs_numerics,band_kwargs,lens_light_kwargs,source_light_kwargs,lens_nonlight_kwargs,kwargs_model_):
    numpix = 127

    sim_g = SimAPI(numpix=numpix, kwargs_single_band=band_kwargs[0], kwargs_model=kwargs_model_)
    sim_r = SimAPI(numpix=numpix, kwargs_single_band=band_kwargs[1], kwargs_model=kwargs_model_)
    sim_i = SimAPI(numpix=numpix, kwargs_single_band=band_kwargs[2], kwargs_model=kwargs_model_)

    imSim_g = sim_g.image_model_class(kwargs_numerics)
    imSim_r = sim_r.image_model_class(kwargs_numerics)
    imSim_i = sim_i.image_model_class(kwargs_numerics)

    kwargs_lens_light_g, kwargs_source_g,_ = sim_g.magnitude2amplitude(lens_light_kwargs[0], source_light_kwargs[0])
    kwargs_lens_light_r, kwargs_source_r,_ = sim_r.magnitude2amplitude(lens_light_kwargs[1], source_light_kwargs[1])
    kwargs_lens_light_i, kwargs_source_i,_ = sim_i.magnitude2amplitude(lens_light_kwargs[2], source_light_kwargs[2])

    image_g_surface_brightness = imSim_g.image(lens_nonlight_kwargs, kwargs_source_g, kwargs_lens_light_g,point_source_add=False,source_add=True,lens_light_add=False) 
    image_r_surface_brightness = imSim_r.image(lens_nonlight_kwargs, kwargs_source_r, kwargs_lens_light_r,point_source_add=False,source_add=True,lens_light_add=False) 
    image_i_surface_brightness = imSim_i.image(lens_nonlight_kwargs, kwargs_source_i, kwargs_lens_light_i,point_source_add=False,source_add=True,lens_light_add=False) 
    #units of e-counts/sec/arcsec^2

    image_g_flux = image_g_surface_brightness * band_kwargs[0]['pixel_scale']**2
    image_r_flux = image_r_surface_brightness * band_kwargs[1]['pixel_scale']**2
    image_i_flux = image_i_surface_brightness * band_kwargs[2]['pixel_scale']**2
    #units of e-counts/sec

    image_g = (image_g_flux + sim_g.noise_for_model(model=image_g_flux,background_noise=False)) * band_kwargs[0]['exposure_time'] * band_kwargs[0]['num_exposures'] 
    image_r = (image_r_flux + sim_r.noise_for_model(model=image_r_flux,background_noise=False))  * band_kwargs[1]['exposure_time'] * band_kwargs[1]['num_exposures'] 
    image_i = (image_i_flux + sim_i.noise_for_model(model=image_i_flux,background_noise=False)) * band_kwargs[2]['exposure_time'] * band_kwargs[2]['num_exposures'] 
    #Each output pixel in units of e counts 

    
    total_exposure_times = np.array([band_kwargs[0]['exposure_time'] * band_kwargs[0]['num_exposures'],band_kwargs[1]['exposure_time'] * band_kwargs[1]['num_exposures'],band_kwargs[2]['exposure_time'] * band_kwargs[2]['num_exposures'] ])

    return image_g,image_r,image_i,total_exposure_times

def generate_images(setup_results, dlu_2_results):
    kwargs_numerics = {'point_source_supersampling_factor': 1}
    
    img_g,img_r,img_i,tot_exp_times = simulate(kwargs_numerics=kwargs_numerics,band_kwargs=dlu_2_results.bands,lens_light_kwargs=kwargs_lens_light_mag,source_light_kwargs=kwargs_source_mag,lens_nonlight_kwargs=dlu_1_results.lens_kwargs_list,kwargs_model_=kwargs_model)
    
    img_nss_g,img_nss_r,img_nss_i,_ = simulate(kwargs_numerics=kwargs_numerics,band_kwargs=dlu_2_results.bands,lens_light_kwargs=kwargs_lens_light_mag,source_light_kwargs=kwargs_source_mag,lens_nonlight_kwargs=macro_kwargs_list_nss,kwargs_model_=kwargs_model_nss)

    SNR_g_band = find_SNR(img_g)
    SNR_r_band = find_SNR(img_r)
    SNR_i_band = find_SNR(img_i)

    SNR = np.array([SNR_g_band,SNR_r_band,SNR_i_band]) #SNR of lensed residual in each band 

    sns_diff_g = img_g / img_nss_g
    sns_diff_r = img_r / img_nss_r
    sns_diff_i = img_i / img_nss_i

    img = (img_g,img_r,img_i)
    img_nss = (img_nss_g,img_nss_r,img_nss_i)
    sns_diff = (sns_diff_g,sns_diff_r,sns_diff_i)

    return {
        "img": img,
        "img_nss": img_nss,
        "sns_diff": sns_diff,
        "SNR": SNR,
        "tot_exp_times": tot_exp_times,
    }
