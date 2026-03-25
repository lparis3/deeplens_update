import time

from model_alpha_pipeline.structures.dataclasses import dlu_2_output
from model_alpha_pipeline.observations.selection import extraction


def instrument_config(Instrument):

    if Instrument == 'LSST':
        from lenstronomy.SimulationAPI.ObservationConfig.LSST import LSST
        band1 = 'g'
        band2 = 'r'
        band3 = 'i'
        LSST_g = LSST(band=band1, psf_type='GAUSSIAN', coadd_years=10)
        LSST_r = LSST(band=band2, psf_type='GAUSSIAN', coadd_years=10)
        LSST_i = LSST(band=band3, psf_type='GAUSSIAN', coadd_years=10)
        lsst = [LSST_g, LSST_r, LSST_i]
        return lsst, [band1,band2,band3]

    elif Instrument == 'DES':
        from lenstronomy.SimulationAPI.ObservationConfig.DES import DES
        band1 = 'g'
        band2 = 'r'
        band3 = 'i'
        DES_g = DES(band = band1,psf_type='GAUSSIAN',coadd_years=3)
        DES_r = DES(band = band2,psf_type='GAUSSIAN',coadd_years=3)
        DES_i = DES(band = band3,psf_type='GAUSSIAN',coadd_years=3)
        des = [DES_g,DES_r,DES_i]
        return des, [band1,band2,band3]


def dlu_2(Instrument,observational_data,z_pair,redshift_bin_edges):
    '''Chooses real observations of galaxies to be used as light profile for source and lens.'''

    #1. Configure instrument specific parameters
    start1 = time.time()

    instrument_param,band_labels = instrument_config(Instrument=Instrument)
    band_g, band_r, band_i = instrument_param
    kwargs_g_band = band_g.kwargs_single_band()
    kwargs_r_band = band_r.kwargs_single_band()
    kwargs_i_band = band_i.kwargs_single_band()
    bands = [kwargs_g_band,kwargs_r_band,kwargs_i_band]

    end1 = time.time()
    print(f'Step 2 took {end1-start1} secs')

    #2.Data Extraction
    start2 = time.time()

    source_images,source_mag,deflector_images,deflector_mag, raw_src, raw_dfr = extraction(observational_data,z_pair,redshift_bin_edges)

    end2 = time.time()
    print(f'Step 3 took {end2-start2} seconds')

    results = dlu_2_output(bands=bands,
                           band_labels=band_labels,
                           source_images = source_images,
                           source_mag = source_mag,
                           deflector_images=deflector_images,
                           deflector_mag=deflector_mag,
                           raw_src=raw_src,
                           raw_dfr=raw_dfr)

    return results
