import numpy as np
from lenstronomy.SimulationAPI.sim_api import SimAPI


def find_SNR(image):
    '''Assuming only Poisson noise, finds snr image's brightest pixel'''
    SNR = np.sqrt(np.max(image))
    return SNR

def combine_ab_magnitudes(m1, m2):
    """
    Combine two AB magnitudes by summing their fluxes.
    Parameters
    ----------
    m1, m2 : float
        AB magnitudes of the two sources.
    Returns
    -------
    m_total : float
        AB magnitude corresponding to the summed flux.
    """
    # Convert AB mag to flux density (erg/s/cm^2/Hz)
    # m_AB = -2.5 * log10(f_nu) - 48.6
    f1 = 10 ** (-0.4 * (m1 + 48.6))
    f2 = 10 ** (-0.4 * (m2 + 48.6))
    
    # Sum the fluxes
    f_total = f1 + f2
    
    # Convert back to AB magnitude
    m_total = -2.5 * np.log10(f_total) - 48.6
    
    return m_total


def simulate(Instrument,kwargs_numerics,band_kwargs,lens_light_kwargs,source_light_kwargs,lens_nonlight_kwargs,kwargs_model_):
    numpix = 127
    if Instrument == 'LSST':
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

        return [image_g,image_r,image_i,total_exposure_times]

    elif Instrument == 'DES':
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

        return [image_g,image_r,image_i,total_exposure_times]

    elif Instrument=='Euclid':
        sim_VIS = SimAPI(numpix=numpix, kwargs_single_band=band_kwargs[0], kwargs_model=kwargs_model_)
        imSim_VIS = sim_VIS.image_model_class(kwargs_numerics)

        kwargs_lens_light_r, kwargs_source_r,_ = sim_VIS.magnitude2amplitude(lens_light_kwargs[0], source_light_kwargs[0])
        kwargs_lens_light_i, kwargs_source_i,_ = sim_VIS.magnitude2amplitude(lens_light_kwargs[1], source_light_kwargs[1])

        #Before lensing our r and i band images we will create an approximate VIS band image as well (VIS band ~ equal in r and i)
        #approximate_lens_magnitude_VIS = combine_ab_magnitudes(kwargs_lens_light_r[0]['magnitude'],kwargs_lens_light_i[0]['magnitude'])
        #approximate_source_magnitude_VIS = combine_ab_magnitudes(kwargs_source_r[0]['magnitude'],kwargs_source_i[0]['magnitude'])
        #approximate_lens_image_VIS = kwargs_lens_light_r[0]['image'] + kwargs_lens_light_i[0]['image']
        #approximate_source_image_VIS = kwargs_source_r[0]['image'] + kwargs_source_i[0]['image']
        #lens_light_kwargs_VIS = [{'magnitude': approximate_lens_magnitude_VIS, 'image': approximate_lens_image_VIS, 'center_x': kwargs_lens_light_r[0]['center_x'], 'center_y':kwargs_lens_light_r[0]['center_y'], 'phi_G': 0.0, 'scale': kwargs_lens_light_r[0]['scale']}]
        #source_light_kwargs_VIS = [{'magnitude': approximate_source_magnitude_VIS, 'image': approximate_source_image_VIS, 'center_x': kwargs_source_r[0]['center_x'], 'center_y':kwargs_source_r[0]['center_y'], 'phi_G': 0.0, 'scale': kwargs_source_r[0]['scale']}]
        #kwargs_lens_light_VIS, kwargs_source_VIS,_ = sim_VIS.magnitude2amplitude(lens_light_kwargs_VIS, source_light_kwargs_VIS)


        # r and i surface brightnesses are intermediate quantities used
        # to synthesize VIS (VIS ~ r + i for Euclid). They are not
        # independent output bands and are not returned.
        image_r_surface_brightness = imSim_VIS.image(lens_nonlight_kwargs, kwargs_source_r, kwargs_lens_light_r,point_source_add=False,source_add=True,lens_light_add=False)
        image_i_surface_brightness = imSim_VIS.image(lens_nonlight_kwargs, kwargs_source_i, kwargs_lens_light_i,point_source_add=False,source_add=True,lens_light_add=False)
        image_VIS_surface_brightness = image_r_surface_brightness + image_i_surface_brightness
        #units of e-counts/sec/arcsec^2

        image_VIS_flux = image_VIS_surface_brightness * band_kwargs[0]['pixel_scale']**2
        #units of e-counts/sec

        image_VIS = (image_VIS_flux + sim_VIS.noise_for_model(model=image_VIS_flux,background_noise=False)) * band_kwargs[0]['exposure_time'] * band_kwargs[0]['num_exposures']
        #Each output pixel in units of e counts

        total_exposure_times = np.array([band_kwargs[0]['exposure_time'] * band_kwargs[0]['num_exposures']])

        return [image_VIS, total_exposure_times]


    elif Instrument == 'Roman_VIS':
        sim_FO62 = SimAPI(numpix=numpix, kwargs_single_band=band_kwargs[0], kwargs_model=kwargs_model_)
        sim_FO87 = SimAPI(numpix=numpix, kwargs_single_band=band_kwargs[1], kwargs_model=kwargs_model_)

        imSim_FO62 = sim_FO62.image_model_class(kwargs_numerics)
        imSim_FO87 = sim_FO87.image_model_class(kwargs_numerics)

        kwargs_lens_light_FO62, kwargs_source_FO62,_ = sim_FO62.magnitude2amplitude(lens_light_kwargs[0], source_light_kwargs[0])
        kwargs_lens_light_FO87, kwargs_source_FO87,_ = sim_FO87.magnitude2amplitude(lens_light_kwargs[1], source_light_kwargs[1])

        image_FO62_surface_brightness = imSim_FO62.image(lens_nonlight_kwargs, kwargs_source_FO62, kwargs_lens_light_FO62,point_source_add=False,source_add=True,lens_light_add=False) 
        image_FO87_surface_brightness = imSim_FO87.image(lens_nonlight_kwargs, kwargs_source_FO87, kwargs_lens_light_FO87,point_source_add=False,source_add=True,lens_light_add=False) 
        #units of e-counts/sec/arcsec^2

        image_FO62_flux = image_FO62_surface_brightness * band_kwargs[0]['pixel_scale']**2
        image_FO87_flux = image_FO87_surface_brightness * band_kwargs[1]['pixel_scale']**2
        #units of e-counts/sec

        # num_exposures already carries the per-instrument scaling from
        # observation_builder.EXPOSURE_SCALING, so it is applied consistently to
        # both the noise model (via sim_FO*) and the image here -- matching the
        # other instruments. Do NOT reintroduce a post-multiply factor.
        image_FO62 = (image_FO62_flux + sim_FO62.noise_for_model(model=image_FO62_flux,background_noise=False)) * band_kwargs[0]['exposure_time'] * band_kwargs[0]['num_exposures']
        image_FO87 = (image_FO87_flux + sim_FO87.noise_for_model(model=image_FO87_flux,background_noise=False))  * band_kwargs[1]['exposure_time'] * band_kwargs[1]['num_exposures']
        #Each output pixel in units of e counts 

        total_exposure_times = np.array([band_kwargs[0]['exposure_time'] * band_kwargs[0]['num_exposures'],band_kwargs[1]['exposure_time'] * band_kwargs[1]['num_exposures']])

        return [image_FO62,image_FO87,total_exposure_times]



def generate_images(Instrument,setup_results, dlu_1_results,dlu_2_results):
    kwargs_numerics = {'point_source_supersampling_factor': 1}
    
    sim_results = simulate(Instrument=Instrument,kwargs_numerics=kwargs_numerics,band_kwargs=dlu_2_results.bands,lens_light_kwargs=setup_results['kwargs_lens_light_mag'],source_light_kwargs=setup_results['kwargs_source_mag'],lens_nonlight_kwargs=dlu_1_results.lens_kwargs_list,kwargs_model_=setup_results['kwargs_model'])
    
    sim_results_nss = simulate(Instrument=Instrument,kwargs_numerics=kwargs_numerics,band_kwargs=dlu_2_results.bands,lens_light_kwargs=setup_results['kwargs_lens_light_mag'],source_light_kwargs=setup_results['kwargs_source_mag'],lens_nonlight_kwargs=setup_results['macro_kwargs_list_nss'],kwargs_model_=setup_results['kwargs_model_nss'])

    # Fail loudly on non-finite images instead of letting NaN/inf propagate silently
    # into SNR and sns_diff below. A common cause is a band config with
    # num_exposures == 0 (zero total exposure time), which makes the noise model
    # divide by zero and return all-NaN; see observation_builder.instrument_config.
    def _assert_finite(results, label):
        # layout is [image_band_0, ..., image_band_{N-1}, total_exposure_times];
        # the final entry is exposure times, so only the image arrays are checked.
        for b, arr in enumerate(results[:-1]):
            if not np.isfinite(arr).all():
                finite_frac = np.isfinite(arr).mean()
                raise ValueError(
                    f"{Instrument} {label} band {b} contains non-finite pixels "
                    f"(finite fraction {finite_frac:.3f}). Check num_exposures and "
                    f"exposure_time in the band config -- a zero total exposure time "
                    f"yields all-NaN noise."
                )

    _assert_finite(sim_results, "image")
    _assert_finite(sim_results_nss, "img_nss")

    SNR = []
    for i in range(0,len(sim_results)-1):
        SNR.append(find_SNR(sim_results[i]))
    SNR = np.array(SNR) #SNR of lensed residual in each band 

    sns_diff = []
    for i in range(0,len(sim_results)-1):
        sns_diff.append(sim_results[i]/sim_results_nss[i])
    sns_diff = np.array(sns_diff) #comparison of lensed image with and without subsctructure

    # sim_results layout: [image_band_0, ..., image_band_{N-1}, total_exposure_times]
    # So all entries except the last are per-band image arrays.
    img = list(sim_results[:-1])
    img_nss = list(sim_results_nss[:-1])
    total_exposure_times = sim_results[-1]

    return {
        "img": img,
        "img_nss": img_nss,
        "sns_diff": sns_diff,
        "SNR": SNR,
        "tot_exp_times": total_exposure_times,
    }