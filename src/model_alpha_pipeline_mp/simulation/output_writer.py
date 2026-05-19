import numpy as np
import h5py


def collect_simulation_output(
    i,
    DM_type,
    Instrument,
    timestamp,
    sampled_vals,
    dlu_1_results,
    dlu_2_results,
    setup_results,
    image_results,
):
    """
    Build a plain-Python dict describing everything that should be written for
    one simulation. Runs in the WORKER process — no HDF5 access.

    Returns a dict with two top-level keys:
        'datasets': { name: (data, comment_or_None) }
            For scalar/string metadata datasets, `data` is a list of strings
            matching the original output_writer's [value, comment] convention.
            For image arrays, `data` is the numpy array itself and we add
            'attrs' (see below) to carry image metadata.
        'image_datasets': { name: {'data': ndarray, 'attrs': {k: [val, comment]}} }
            Separated so the parent knows to attach attrs after creating.
    """
    zdeflector = sampled_vals.redshifts[0]
    zsource = sampled_vals.redshifts[1]

    DL = dlu_1_results.cosmology.angular_diameter_distance(zdeflector) / 1000
    DS = dlu_1_results.cosmology.angular_diameter_distance(zsource) / 1000
    DLS = dlu_1_results.cosmology.angular_diameter_distance_z1z2(zdeflector, zsource) / 1000

    datasets = {
        'd_l': [str(DL), 'Angular diameter distance to deflector galaxy in Gpc'],
        'd_s': [str(DS), 'Angular diameter distance to source galaxy in Gpc'],
        'd_ls': [str(DLS), 'Angular diameter distance between deflector and source galaxy in Gpc'],
        'exposure_time': [str(image_results['tot_exp_times'][0]),
                          str(image_results['tot_exp_times'][1]),
                          str(image_results['tot_exp_times'][2]),
                          'Exposure time in seconds for all bands'],
        'instrument': [Instrument, 'Instrument'],
        'log_mhigh': [str(np.log10(sampled_vals.max_subhalo_mass)), 'Log10 of largest possible subhalo mass'],
        'log_mlow': [str(6), 'Log10 of lowest possible subhalo mass'],
        'host_mass': [str(np.log10(dlu_1_results.host_mass)), 'Log10 of mass of host lens in units of M_sun'],
        'lens_mass': [str(np.log10(dlu_1_results.whole_halo_mass)), 'Log10 of mass of entire lens in units of M_sun'],
        'num_subhalos': [str(dlu_1_results.num_subhalos), 'Number of subhalos'],
        'r_tidal': [str(0.25), 'see Pyhalo documentation'],
        'sigma_sub': [str(0.025), 'see Pyhalo documentation'],
        'snr': [str(image_results['SNR']), 'SNR of lensed in all bands'],
        'theta_e': [str(dlu_1_results.macro_kwargs_list[0]['theta_E']), 'Einstein radius in arcseconds'],
        'uid': [str(i), f'simulation number in {DM_type} batch created on {timestamp}'],
        'z_lens': [str(zdeflector), 'Redshift of deflector'],
        'z_source': [str(zsource), 'Redshift of source'],
        'bands': [dlu_2_results.band_labels[0], dlu_2_results.band_labels[1], dlu_2_results.band_labels[2],
                  'Instrument bands in which image was simulated'],
        'DM_type': [DM_type, 'Type of dark matter assumed'],
        'source_pos': [str([setup_results['source_x'], setup_results['source_y']]),
                       'Plane coordinates of source wrt center of deflector (in arcseconds)'],
        'host_slope': [str(dlu_1_results.slope_Host), 'EPL slope of host halo'],
        'ellipticity': [str([dlu_1_results.macro_kwargs_list[0]['e1'],
                             dlu_1_results.macro_kwargs_list[0]['e2']]),
                        'Ellipticity values for host halo mass profile'],
    }

    # DM-type-specific metadata
    if DM_type == 'WDM':
        datasets['log_mc'] = [str(dlu_1_results.type_kwargs['log_mc']),
                              'Mass at which underneath the WDM mass function is surpressed when compared to CDM']
    elif DM_type == 'Axion':
        datasets['m_axion'] = [str(dlu_1_results.type_kwargs['M_axion']),
                               'log10 of mass of axion particle']
        datasets['flucs_shape'] = [dlu_1_results.type_kwargs['flucs_shape'],
                                   'Shape at which to render fluctuations of the host halo (See Pyhalo)']
        datasets['flucs_args'] = [str(dlu_1_results.type_kwargs['flucs_args']['angle']),
                                  str(dlu_1_results.type_kwargs['flucs_args']['rmin']),
                                  str(dlu_1_results.type_kwargs['flucs_args']['rmax']),
                                  'Arugments corresponding to geometry of host halo fluctuations (See Pyhalo)']
    elif DM_type == 'SIDM':
        datasets['subhalo_mass_ranges'] = [str(dlu_1_results.type_kwargs['mass_ranges_subhalos']),
                                           'Mass ranges at which subhalos are sampled from']
        datasets['field_halo_mass_ranges'] = [str(dlu_1_results.type_kwargs['mass_ranges_field_halos']),
                                              'Mass ranges at which field halos are sampled from']
        datasets['prob_subhalo'] = [str(dlu_1_results.type_kwargs['probabilities_subhalos']),
                                    'Probability of subhalo being sampled from corresponding mass range']
        datasets['prob_field_halo'] = [str(dlu_1_results.type_kwargs['probabilities_field_halos']),
                                       'Probability of field halo being sampled from corresponding mass range']

    # Image datasets: array data + per-band attrs
    image_datasets = {}
    for band_idx in range(3):
        band = dlu_2_results.band_labels[band_idx]
        name = f'exposure_{i}_{band}'
        image_datasets[name] = {
            'data': np.asarray(image_results['img'][band_idx]),
            'attrs': {
                'filter': [band, 'Filter'],
                'fov': [str(dlu_1_results.arcsecond_opening_angle), 'Field of view [arcsec]'],
                'pixel_scale': [str(dlu_2_results.bands[band_idx]['pixel_scale']), 'Pixel scale [arcsec/pixel]'],
                'lens_magnitude': [str(dlu_2_results.deflector_mag[band_idx]), 'Lens magnitude'],
                'source_magnitude': [str(dlu_2_results.source_mag[band_idx]), 'Unlensed source galaxy magnitude'],
                'units': ['counts', 'Units of pixel values'],
            },
        }
        nss_name = f'exposure_{i}_{band}_nss'
        image_datasets[nss_name] = {
            'data': np.asarray(image_results['img_nss'][band_idx]),
            'attrs': {},
        }

    return {'datasets': datasets, 'image_datasets': image_datasets}


def write_simulation_output(hf, i, collected):
    """
    Write a collected-output dict into the HDF5 file under
    `images/strong_lens_{i}/...`. Runs in the PARENT process.
    """
    dt = h5py.string_dtype(encoding='utf-8')
    group_path = f'images/strong_lens_{i}'

    # String-array metadata datasets
    for name, value_list in collected['datasets'].items():
        hf.create_dataset(f'{group_path}/{name}',
                          data=np.array(value_list, dtype=dt),
                          dtype=dt)

    # Image datasets + attrs
    for name, payload in collected['image_datasets'].items():
        dset = hf.create_dataset(f'{group_path}/{name}', data=payload['data'])
        for attr_key, attr_val in payload['attrs'].items():
            dset.attrs[attr_key] = np.array(attr_val, dtype=dt)