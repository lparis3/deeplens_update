import numpy as np
from scipy.interpolate import RegularGridInterpolator
from lenstronomy.SimulationAPI.sim_api import SimAPI


# ---------------------------------------------------------------------------
# Ray-shooting reuse across bands.
#
# For an extended source, imSim.image(... source_add=True, lens_light_add=False,
# point_source_add=False) internally does, for a single source plane:
#     beta = LensModel.ray_shooting(grid, kwargs_lens)        # band-INDEPENDENT, expensive
#     sb   = SourceModel.surface_brightness(beta, kwargs_source) * _flux_scaling
#     img  = ImageNumerics.re_size_convolve(sb, unconvolved=False)   # band-dependent
# The deflection field depends only on the lens model + grid, which are identical
# across all bands of one instrument (same numpix + pixel scale + supersampling).
# So we ray-shoot ONCE and reuse beta for every band. The two helpers below
# reproduce imSim.image() bit-for-bit while skipping the redundant
# ray-shooting on all but the first band.
# ---------------------------------------------------------------------------
def _shared_deflection(imSim_ref, kwargs_lens):
    """Ray-shoot the image-plane grid through the lens once; returns the
    reference eval-grid coords and the deflected (source-plane) coords."""
    ra, dec = imSim_ref.ImageNumerics.coordinates_evaluate
    beta_x, beta_y = imSim_ref.LensModel.ray_shooting(ra, dec, kwargs_lens)
    return ra, dec, beta_x, beta_y


def _lensed_source_sb(imSim, kwargs_source, ra_ref, dec_ref, beta_x, beta_y):
    """Lensed source surface brightness reusing a precomputed deflection field.
    Bit-for-bit equivalent to
        imSim.image(kwargs_lens, kwargs_source, None,
                    point_source_add=False, source_add=True, lens_light_add=False).
    The assert guards the one precondition: this band must share the reference
    pixel grid (true for all bands of a single instrument)."""
    ra, dec = imSim.ImageNumerics.coordinates_evaluate
    assert np.array_equal(ra, ra_ref) and np.array_equal(dec, dec_ref), \
        "band eval grid differs from reference; cannot reuse ray-shooting"
    sb = imSim.SourceModel.surface_brightness(beta_x, beta_y, kwargs_source) * imSim._flux_scaling
    return imSim.ImageNumerics.re_size_convolve(sb, unconvolved=False)


# ---------------------------------------------------------------------------
# Convergence-map resolution.
#
# kappa over ~10^4 multi-plane deflectors at the full 127^2 image resolution is
# the single most expensive operation in the sim (tens of seconds). Its cost
# scales ~ KAPPA_NUMPIX^2, so evaluating on a coarse grid and upsampling back to
# the image grid is ~4-5x faster at KAPPA_NUMPIX=48 while staying within a few %
# RMS of the full-res map. The residual error concentrates at subhalo cusps
# (small scales), so raise toward 64/96 if you need sharper substructure in the
# stored kappa, or lower toward 32 for more speed.
# ---------------------------------------------------------------------------
KAPPA_NUMPIX = 48
# If True, bilinearly upsample the coarse map back onto the 127^2 image grid so
# the stored convergence map matches the image shape/footprint (drop-in). If
# False, return the coarse KAPPA_NUMPIX^2 map as-is (smaller, different shape).
KAPPA_UPSAMPLE = True


def _convergence_map(imSim, kwargs_lens, kappa_numpix=KAPPA_NUMPIX, upsample=KAPPA_UPSAMPLE):
    """Convergence (kappa) on a coarse grid spanning the image field of view,
    optionally upsampled back onto the full image grid.

    Coarse axes are taken from the image grid itself (imSim.Data.pixel_coordinates),
    so the map inherits the exact coordinate frame and stays aligned with the image.
    Assumes an axis-aligned grid (no rotation) -- the SimAPI default here."""
    x_full, y_full = imSim.Data.pixel_coordinates          # 2d (ny, nx), arcsec
    x_ax, y_ax = x_full[0, :], y_full[:, 0]                 # 1d coordinate axes
    xc = np.linspace(x_ax[0], x_ax[-1], kappa_numpix)
    yc = np.linspace(y_ax[0], y_ax[-1], kappa_numpix)
    Xc, Yc = np.meshgrid(xc, yc)
    kappa_coarse = imSim.LensModel.kappa(
        Xc.ravel(), Yc.ravel(), kwargs_lens).reshape(kappa_numpix, kappa_numpix)
    if not upsample:
        return kappa_coarse
    interp = RegularGridInterpolator((yc, xc), kappa_coarse, method='linear',
                                     bounds_error=False, fill_value=None)
    pts = np.stack([y_full.ravel(), x_full.ravel()], axis=-1)
    return interp(pts).reshape(y_full.shape)


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

        # Ray-shoot once (band-independent) and reuse the deflection field across bands.
        ra_ref, dec_ref, beta_x, beta_y = _shared_deflection(imSim_g, lens_nonlight_kwargs)
        image_g_surface_brightness = _lensed_source_sb(imSim_g, kwargs_source_g, ra_ref, dec_ref, beta_x, beta_y)
        image_r_surface_brightness = _lensed_source_sb(imSim_r, kwargs_source_r, ra_ref, dec_ref, beta_x, beta_y)
        image_i_surface_brightness = _lensed_source_sb(imSim_i, kwargs_source_i, ra_ref, dec_ref, beta_x, beta_y)
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

        # convergence map (band-independent). Evaluated on a coarse KAPPA_NUMPIX^2
        # grid and upsampled to the image grid -- full-res kappa over ~10^4
        # deflectors is the most expensive op in the sim; see _convergence_map.
        kappa = _convergence_map(imSim_g, lens_nonlight_kwargs)

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

        # Ray-shoot once (band-independent) and reuse the deflection field across bands.
        ra_ref, dec_ref, beta_x, beta_y = _shared_deflection(imSim_g, lens_nonlight_kwargs)
        image_g_surface_brightness = _lensed_source_sb(imSim_g, kwargs_source_g, ra_ref, dec_ref, beta_x, beta_y)
        image_r_surface_brightness = _lensed_source_sb(imSim_r, kwargs_source_r, ra_ref, dec_ref, beta_x, beta_y)
        image_i_surface_brightness = _lensed_source_sb(imSim_i, kwargs_source_i, ra_ref, dec_ref, beta_x, beta_y)
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

        # convergence map (band-independent). Evaluated on a coarse KAPPA_NUMPIX^2
        # grid and upsampled to the image grid -- full-res kappa over ~10^4
        # deflectors is the most expensive op in the sim; see _convergence_map.
        kappa = _convergence_map(imSim_g, lens_nonlight_kwargs)

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
        # Ray-shoot once (band-independent) and reuse the deflection field across bands.
        ra_ref, dec_ref, beta_x, beta_y = _shared_deflection(imSim_VIS, lens_nonlight_kwargs)
        image_r_surface_brightness = _lensed_source_sb(imSim_VIS, kwargs_source_r, ra_ref, dec_ref, beta_x, beta_y)
        image_i_surface_brightness = _lensed_source_sb(imSim_VIS, kwargs_source_i, ra_ref, dec_ref, beta_x, beta_y)
        image_VIS_surface_brightness = image_r_surface_brightness + image_i_surface_brightness
        #units of e-counts/sec/arcsec^2

        image_VIS_flux = image_VIS_surface_brightness * band_kwargs[0]['pixel_scale']**2
        #units of e-counts/sec

        image_VIS = (image_VIS_flux + sim_VIS.noise_for_model(model=image_VIS_flux,background_noise=False)) * band_kwargs[0]['exposure_time'] * band_kwargs[0]['num_exposures']
        #Each output pixel in units of e counts

        total_exposure_times = np.array([band_kwargs[0]['exposure_time'] * band_kwargs[0]['num_exposures']])

        # convergence map (band-independent). Evaluated on a coarse KAPPA_NUMPIX^2
        # grid and upsampled to the image grid -- full-res kappa over ~10^4
        # deflectors is the most expensive op in the sim; see _convergence_map.
        kappa = _convergence_map(imSim_VIS, lens_nonlight_kwargs)

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

        # Ray-shoot once (band-independent) and reuse the deflection field across bands.
        ra_ref, dec_ref, beta_x, beta_y = _shared_deflection(imSim_FO62, lens_nonlight_kwargs)
        image_FO62_surface_brightness = _lensed_source_sb(imSim_FO62, kwargs_source_FO62, ra_ref, dec_ref, beta_x, beta_y)
        image_FO87_surface_brightness = _lensed_source_sb(imSim_FO87, kwargs_source_FO87, ra_ref, dec_ref, beta_x, beta_y)
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

        # convergence map (band-independent). Evaluated on a coarse KAPPA_NUMPIX^2
        # grid and upsampled to the image grid -- full-res kappa over ~10^4
        # deflectors is the most expensive op in the sim; see _convergence_map.
        kappa = _convergence_map(imSim_FO62, lens_nonlight_kwargs)

        # unlensed source on the same grid, clean (no noise), in e- counts.
        # de_lensed=True renders the source with no ray-shooting; unconvolved=False
        # convolves the band PSF so the unlensed image matches the lensed image's PSF
        # treatment (set unconvolved=True to skip it if the INTERPOL input is already PSF'd).
        image_FO62_unlensed = imSim_FO62.source_surface_brightness(kwargs_source_FO62, de_lensed=True, unconvolved=False) * band_kwargs[0]['pixel_scale']**2 * band_kwargs[0]['exposure_time'] * band_kwargs[0]['num_exposures']
        image_FO87_unlensed = imSim_FO87.source_surface_brightness(kwargs_source_FO87, de_lensed=True, unconvolved=False) * band_kwargs[1]['pixel_scale']**2 * band_kwargs[1]['exposure_time'] * band_kwargs[1]['num_exposures']

        return [image_FO62,image_FO87,total_exposure_times,kappa,[image_FO62_unlensed,image_FO87_unlensed]]



def generate_images(Instrument, setup_results, dlu_1_results, dlu_2_results, nss=True):
    """Render the lensed images plus the convergence map and unlensed source.

    nss : bool
        If True (default), also render the macro-only ("no substructure") model
        and return the substructure comparisons. If False, that second render is
        skipped entirely and every nss-related output (img_nss, sns_diff,
        kappa_nss, kappa_sub) is returned as a zero-filled array of the same
        shape/dtype as the nss=True case, so the written output file structure is
        identical either way.
    """
    kwargs_numerics = {'point_source_supersampling_factor': 1}

    sim_results = simulate(Instrument=Instrument,kwargs_numerics=kwargs_numerics,band_kwargs=dlu_2_results.bands,lens_light_kwargs=setup_results['kwargs_lens_light_mag'],source_light_kwargs=setup_results['kwargs_source_mag'],lens_nonlight_kwargs=dlu_1_results.lens_kwargs_list,kwargs_model_=setup_results['kwargs_model'])

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

    # sim_results layout: [image_band_0, ..., image_band_{N-1}, total_exposure_times, kappa, img_unlensed]
    # So all entries except the last three are per-band lensed image arrays.
    n_bands = len(sim_results) - 3
    img = list(sim_results[:-3])
    total_exposure_times = sim_results[-3]
    kappa = sim_results[-2]            # full-model convergence map
    img_unlensed = sim_results[-1]     # unlensed source per band (lens-model-independent)

    SNR = []
    for i in range(0, n_bands):
        SNR.append(find_SNR(sim_results[i]))
    SNR = np.array(SNR)  # SNR of lensed image in each band

    if nss:
        # macro-only ("no substructure") render and the substructure comparisons.
        sim_results_nss = simulate(Instrument=Instrument,kwargs_numerics=kwargs_numerics,band_kwargs=dlu_2_results.bands,lens_light_kwargs=setup_results['kwargs_lens_light_mag'],source_light_kwargs=setup_results['kwargs_source_mag'],lens_nonlight_kwargs=setup_results['macro_kwargs_list_nss'],kwargs_model_=setup_results['kwargs_model_nss'])
        _assert_finite(sim_results_nss, "img_nss")

        img_nss = list(sim_results_nss[:-3])

        sns_diff = []
        for i in range(0, n_bands):
            sns_diff.append(sim_results[i] / sim_results_nss[i])
        sns_diff = np.array(sns_diff)  # lensed image with vs without substructure

        kappa_nss = sim_results_nss[-2]   # macro-only convergence map
        kappa_sub = kappa - kappa_nss     # substructure convergence
    else:
        # nss disabled. Return zero-filled arrays with the SAME shape/dtype as the
        # nss=True case (rather than None) so the written HDF5 file has an identical
        # structure regardless of the toggle. Only img_nss and kappa_nss reach the
        # file (as the *_nss image datasets and convergence_field_nss); sns_diff and
        # kappa_sub aren't persisted but are zeroed too for a consistent return dict.
        # A real macro render is never exactly all-zero (it carries noise), so the
        # all-zero arrays double as a sentinel that the nss model was not computed.
        img_nss = [np.zeros_like(b) for b in img]
        sns_diff = np.zeros_like(np.array(img))
        kappa_nss = np.zeros_like(kappa)
        kappa_sub = np.zeros_like(kappa)

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