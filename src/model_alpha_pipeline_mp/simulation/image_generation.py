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

        # convergence is band-independent -> compute once from any band's imSim,
        # on the same grid as the image so it lines up pixel-for-pixel
        x_grid, y_grid = imSim_g.Data.pixel_coordinates
        kappa = imSim_g.LensModel.kappa(x_grid, y_grid, lens_nonlight_kwargs)

        # unlensed source on the same grid, clean (no noise), in e- counts.
        # de_lensed=True renders the source with no ray-shooting; unconvolved=False
        # convolves the band PSF so the unlensed image matches the lensed image's PSF
        # treatment (set unconvolved=True to skip it if the INTERPOL input is already PSF'd).
        image_g_unlensed = imSim_g.source_surface_brightness(kwargs_source_g, de_lensed=True, unconvolved=False) * band_kwargs[0]['pixel_scale']**2 * band_kwargs[0]['exposure_time'] * band_kwargs[0]['num_exposures']
        image_r_unlensed = imSim_r.source_surface_brightness(kwargs_source_r, de_lensed=True, unconvolved=False) * band_kwargs[1]['pixel_scale']**2 * band_kwargs[1]['exposure_time'] * band_kwargs[1]['num_exposures']
        image_i_unlensed = imSim_i.source_surface_brightness(kwargs_source_i, de_lensed=True, unconvolved=False) * band_kwargs[2]['pixel_scale']**2 * band_kwargs[2]['exposure_time'] * band_kwargs[2]['num_exposures']

        return [image_g,image_r,image_i,total_exposure_times,kappa,[image_g_unlensed,image_r_unlensed,image_i_unlensed]]

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

        # convergence is band-independent -> compute once from any band's imSim,
        # on the same grid as the image so it lines up pixel-for-pixel
        x_grid, y_grid = imSim_g.Data.pixel_coordinates
        kappa = imSim_g.LensModel.kappa(x_grid, y_grid, lens_nonlight_kwargs)

        # unlensed source on the same grid, clean (no noise), in e- counts.
        # de_lensed=True renders the source with no ray-shooting; unconvolved=False
        # convolves the band PSF so the unlensed image matches the lensed image's PSF
        # treatment (set unconvolved=True to skip it if the INTERPOL input is already PSF'd).
        image_g_unlensed = imSim_g.source_surface_brightness(kwargs_source_g, de_lensed=True, unconvolved=False) * band_kwargs[0]['pixel_scale']**2 * band_kwargs[0]['exposure_time'] * band_kwargs[0]['num_exposures']
        image_r_unlensed = imSim_r.source_surface_brightness(kwargs_source_r, de_lensed=True, unconvolved=False) * band_kwargs[1]['pixel_scale']**2 * band_kwargs[1]['exposure_time'] * band_kwargs[1]['num_exposures']
        image_i_unlensed = imSim_i.source_surface_brightness(kwargs_source_i, de_lensed=True, unconvolved=False) * band_kwargs[2]['pixel_scale']**2 * band_kwargs[2]['exposure_time'] * band_kwargs[2]['num_exposures']

        return [image_g,image_r,image_i,total_exposure_times,kappa,[image_g_unlensed,image_r_unlensed,image_i_unlensed]]

    elif Instrument=='Euclid':
        sim_VIS = SimAPI(numpix=numpix, kwargs_single_band=band_kwargs[0], kwargs_model=kwargs_model_)
        imSim_VIS = sim_VIS.image_model_class(kwargs_numerics)

        kwargs_lens_light_r, kwargs_source_r,_ = sim_VIS.magnitude2amplitude(lens_light_kwargs[0], source_light_kwargs[0])
        kwargs_lens_light_i, kwargs_source_i,_ = sim_VIS.magnitude2amplitude(lens_light_kwargs[1], source_light_kwargs[1])


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

        # convergence is band-independent -> compute once from the VIS imSim,
        # on the same grid as the image so it lines up pixel-for-pixel
        x_grid, y_grid = imSim_VIS.Data.pixel_coordinates
        kappa = imSim_VIS.LensModel.kappa(x_grid, y_grid, lens_nonlight_kwargs)

        # unlensed source on the same grid, clean (no noise), in e- counts. Mirror the
        # lensed VIS construction: VIS ~ r + i. de_lensed=True renders the source with no
        # ray-shooting; unconvolved=False convolves the band PSF so the unlensed image
        # matches the lensed PSF treatment (set unconvolved=True to skip it if already PSF'd).
        sb_r_unlensed = imSim_VIS.source_surface_brightness(kwargs_source_r, de_lensed=True, unconvolved=False)
        sb_i_unlensed = imSim_VIS.source_surface_brightness(kwargs_source_i, de_lensed=True, unconvolved=False)
        image_VIS_unlensed = (sb_r_unlensed + sb_i_unlensed) * band_kwargs[0]['pixel_scale']**2 * band_kwargs[0]['exposure_time'] * band_kwargs[0]['num_exposures']

        return [image_VIS, total_exposure_times, kappa, [image_VIS_unlensed]]


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

        # convergence is band-independent -> compute once from any band's imSim,
        # on the same grid as the image so it lines up pixel-for-pixel
        x_grid, y_grid = imSim_FO62.Data.pixel_coordinates
        kappa = imSim_FO62.LensModel.kappa(x_grid, y_grid, lens_nonlight_kwargs)

        # unlensed source on the same grid, clean (no noise), in e- counts.
        # de_lensed=True renders the source with no ray-shooting; unconvolved=False
        # convolves the band PSF so the unlensed image matches the lensed image's PSF
        # treatment (set unconvolved=True to skip it if the INTERPOL input is already PSF'd).
        image_FO62_unlensed = imSim_FO62.source_surface_brightness(kwargs_source_FO62, de_lensed=True, unconvolved=False) * band_kwargs[0]['pixel_scale']**2 * band_kwargs[0]['exposure_time'] * band_kwargs[0]['num_exposures']
        image_FO87_unlensed = imSim_FO87.source_surface_brightness(kwargs_source_FO87, de_lensed=True, unconvolved=False) * band_kwargs[1]['pixel_scale']**2 * band_kwargs[1]['exposure_time'] * band_kwargs[1]['num_exposures']

        return [image_FO62,image_FO87,total_exposure_times,kappa,[image_FO62_unlensed,image_FO87_unlensed]]



def generate_images(Instrument,setup_results, dlu_1_results,dlu_2_results):
    kwargs_numerics = {'point_source_supersampling_factor': 1}
    
    sim_results = simulate(Instrument=Instrument,kwargs_numerics=kwargs_numerics,band_kwargs=dlu_2_results.bands,lens_light_kwargs=setup_results['kwargs_lens_light_mag'],source_light_kwargs=setup_results['kwargs_source_mag'],lens_nonlight_kwargs=dlu_1_results.lens_kwargs_list,kwargs_model_=setup_results['kwargs_model'])
    
    sim_results_nss = simulate(Instrument=Instrument,kwargs_numerics=kwargs_numerics,band_kwargs=dlu_2_results.bands,lens_light_kwargs=setup_results['kwargs_lens_light_mag'],source_light_kwargs=setup_results['kwargs_source_mag'],lens_nonlight_kwargs=setup_results['macro_kwargs_list_nss'],kwargs_model_=setup_results['kwargs_model_nss'])

    # Fail loudly on non-finite images instead of letting NaN/inf propagate silently
    # into SNR and sns_diff below. A common cause is a band config with
    # num_exposures == 0 (zero total exposure time), which makes the noise model
    # divide by zero and return all-NaN; see observation_builder.instrument_config.
    def _assert_finite(results, label):
        # layout is [image_band_0, ..., image_band_{N-1}, total_exposure_times, kappa, img_unlensed];
        # the last three entries are exposure times, the convergence map, and the list of
        # unlensed images, so only the lensed image arrays are checked. kappa is intentionally
        # excluded: for a lens centered at (0,0) on an odd grid, a singular profile
        # (SIS/SIE/EPL) diverges at the central pixel, so a non-finite center pixel is expected.
        # img_unlensed is also excluded here (it's a list, and it's a clean no-noise product).
        for b, arr in enumerate(results[:-3]):
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
    for i in range(0,len(sim_results)-3):
        SNR.append(find_SNR(sim_results[i]))
    SNR = np.array(SNR) #SNR of lensed residual in each band 

    sns_diff = []
    for i in range(0,len(sim_results)-3):
        sns_diff.append(sim_results[i]/sim_results_nss[i])
    sns_diff = np.array(sns_diff) #comparison of lensed image with and without subsctructure

    # sim_results layout: [image_band_0, ..., image_band_{N-1}, total_exposure_times, kappa, img_unlensed]
    # So all entries except the last three are per-band lensed image arrays.
    img = list(sim_results[:-3])
    img_nss = list(sim_results_nss[:-3])
    total_exposure_times = sim_results[-3]

    # convergence maps (2d, aligned pixel-for-pixel with the images).
    # kappa is the full model, kappa_nss the macro-only model; their difference
    # isolates the substructure convergence -- the kappa-space analog of sns_diff.
    kappa = sim_results[-2]
    kappa_nss = sim_results_nss[-2]
    kappa_sub = kappa - kappa_nss

    # unlensed source per band (clean, no noise), same grid/units as `img`.
    # It depends only on the source light, not the lens model, so the nss run's
    # copy is identical; we return the one from the full run.
    img_unlensed = sim_results[-1]
    return {
        "img": img,
        "img_nss": img_nss,
        "sns_diff": sns_diff,
        "SNR": SNR,
        "tot_exp_times": total_exposure_times,
        "kappa": kappa,
        "kappa_nss": kappa_nss,
        "kappa_sub": kappa_sub,
        "img_unlensed": img_unlensed,
    }