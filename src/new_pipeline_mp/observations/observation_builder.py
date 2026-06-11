import time
import numpy as np
from new_pipeline_mp.structures.dataclasses import dlu_2_output
from new_pipeline_mp.observations.selection import extraction

# ---------------------------------------------------------------------------
# Per-instrument exposure-count scaling -- the SNR tuning knob.
#
# Each value multiplies num_exposures in that instrument's band kwargs. Because
# num_exposures is used BOTH to build the SimAPI noise model and to scale the
# final image, signal and noise stay consistent and the real per-pixel SNR
# scales as sqrt(scale). Tune these to hit a target average SNR; 1.0 leaves the
# survey default untouched.
#
# IMPORTANT: scale exposures HERE (on num_exposures, before SimAPI is built),
# never as a post-multiply on the finished image. A post-multiply scales signal
# and noise by the same factor and leaves the real SNR unchanged (it only
# inflates counts, which fools the sqrt(max) SNR metric).
# ---------------------------------------------------------------------------
EXPOSURE_SCALING = {
    'LSST': 1.0,
    'DES': 1.0,
    'Euclid': 30.0,
    'Roman_VIS': 30.0
}

def make_bins(zvals, min_per_bin=50, n_start=100):
    '''Takes an array of redshift values and returns values segmented into bins'''
    # start with many equal-width bins
    edges = np.linspace(zvals.min(), zvals.max(), n_start + 1)
    counts, _ = np.histogram(zvals, edges)

    # convert to lists for merging
    edges = list(edges)
    counts = list(counts)

    i = 0
    while i < len(counts):
        if counts[i] < min_per_bin:
            if i == 0:
                counts[i+1] += counts[i]
                del counts[i]
                del edges[i+1]
            else:
                counts[i-1] += counts[i]
                del counts[i]
                del edges[i]
                i -= 1
        else:
            i += 1

    edges = np.array(edges)

    # assign indices to bins
    bin_indices = []
    for j in range(len(edges) - 1):
        idx = np.where((zvals >= edges[j]) & (zvals < edges[j+1]))[0]
        bin_indices.append(idx)

    return edges


def instrument_config(Instrument):

    if Instrument == 'LSST':
        from lenstronomy.SimulationAPI.ObservationConfig.LSST import LSST
        band1 = 'g'
        band2 = 'r'
        band3 = 'i'
        needed_hsc_bands = ['g','r','i']
        LSST_g = LSST(band=band1, psf_type='GAUSSIAN', coadd_years=10)
        LSST_r = LSST(band=band2, psf_type='GAUSSIAN', coadd_years=10)
        LSST_i = LSST(band=band3, psf_type='GAUSSIAN', coadd_years=10)
        lsst = [LSST_g, LSST_r, LSST_i]
        return lsst, [band1,band2,band3], needed_hsc_bands

    elif Instrument == 'DES':
        from lenstronomy.SimulationAPI.ObservationConfig.DES import DES
        band1 = 'g'
        band2 = 'r'
        band3 = 'i'
        needed_hsc_bands = ['g','r','i']
        DES_g = DES(band = band1,psf_type='GAUSSIAN',coadd_years=6)
        DES_r = DES(band = band2,psf_type='GAUSSIAN',coadd_years=6)
        DES_i = DES(band = band3,psf_type='GAUSSIAN',coadd_years=6)
        des = [DES_g,DES_r,DES_i]
        return des, [band1,band2,band3], needed_hsc_bands
    
    elif Instrument == 'Euclid':
        import lenstronomy.SimulationAPI.ObservationConfig.Euclid as euclid_mod
        from lenstronomy.SimulationAPI.ObservationConfig.Euclid import Euclid
        band1 = 'VIS'
        needed_hsc_bands = ['r','i']
        # lenstronomy's Euclid config binds self.obs to the module-global VIS_obs
        # dict WITHOUT copying it. With coadd_years=6 the num_exposures branch is
        # skipped, so the baseline of 4 is used as-is. But any Euclid() built with
        # coadd_years<6 anywhere in this process mutates that shared global in place
        # (its recurrence decays num_exposures toward 0, which makes the noise model
        # divide by a zero total exposure time -> all-NaN images). Resetting the
        # baseline here guards this build against contamination from such a call.
        euclid_mod.VIS_obs['num_exposures'] = 4
        Euclid_VIS = Euclid(band = band1,psf_type='GAUSSIAN',coadd_years=6)
        euclid = [Euclid_VIS]
        return euclid, [band1],needed_hsc_bands

    elif Instrument == 'Roman_VIS':
        from lenstronomy.SimulationAPI.ObservationConfig.Roman import Roman
        band1 = 'F062'
        band2 = 'F087'
        needed_hsc_bands = ['r','z']
        Roman_F062 = Roman(band = band1,psf_type='PIXEL',survey_mode='time_domain_wide')
        Roman_F087 = Roman(band = band2,psf_type='PIXEL',survey_mode='time_domain_wide')
        roman = [Roman_F062,Roman_F087]
        return roman, [band1,band2],needed_hsc_bands




def dlu_2(Instrument,observational_data,z_pair,redshift_bin_edges,light_profile='INTERPOL'):
    '''Chooses real observations of galaxies to be used as light profile for source and lens.

    Parameters
    ----------
    light_profile : str
        'INTERPOL' (default): use HSC pixel cutouts as interpolated light
        profiles for both source and lens.
        'SERSIC': use analytic Sersic profiles (SERSIC_ELLIPSE), with
        magnitudes drawn from the HSC catalog and other shape parameters
        following the lens.py convention.
    '''

    #1. Configure instrument specific parameters
    start1 = time.time()

    instrument_params,band_labels,needed_hsc_bands = instrument_config(Instrument=Instrument)
    scale = EXPOSURE_SCALING.get(Instrument, 1.0)
    bands = []
    for instrument_param in instrument_params:
        band = instrument_param.kwargs_single_band()
        # scale num_exposures (kept a positive integer) before it is used to
        # build the noise model and the final image; see EXPOSURE_SCALING above.
        band['num_exposures'] = max(1, int(round(band['num_exposures'] * scale)))
        bands.append(band)

    end1 = time.time()
    #print(f'Step 2 took {end1-start1} secs')

    #2.Data Extraction
    start2 = time.time()

    source_images,source_mag,deflector_images,deflector_mag, raw_src, raw_dfr, \
        source_sersic_params, deflector_sersic_params = extraction(observational_data,z_pair,redshift_bin_edges,needed_hsc_bands,light_profile=light_profile)

    end2 = time.time()
    #print(f'Step 3 took {end2-start2} secs')

    results = dlu_2_output(bands=bands,
                           band_labels=band_labels,
                           needed_hsc_bands=needed_hsc_bands,
                           source_images = source_images,
                           source_mag = source_mag,
                           deflector_images=deflector_images,
                           deflector_mag=deflector_mag,
                           raw_src=raw_src,
                           raw_dfr=raw_dfr,
                           source_sersic_params=source_sersic_params,
                           deflector_sersic_params=deflector_sersic_params)

    return results