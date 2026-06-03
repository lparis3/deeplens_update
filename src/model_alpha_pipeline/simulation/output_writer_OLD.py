import numpy as np
import h5py

def write_simulation_output(
    i,
    DM_type,
    Instrument,
    hf,
    timestamp,
    sampled_vals,
    dlu_1_results,
    dlu_2_results,
    setup_results,
    image_results,
):
    

    zdeflector = sampled_vals.redshifts[0]
    zsource = sampled_vals.redshifts[1]

    dt = h5py.string_dtype(encoding='utf-8')        
    DL = dlu_1_results.cosmology.angular_diameter_distance(zdeflector)/1000
    hf.create_dataset(f'images/strong_lens_{i}/d_l', data=np.array([str(DL),'Angular diameter distance to deflector galaxy in Gpc'],dtype=dt),dtype=dt)
    DS =  dlu_1_results.cosmology.angular_diameter_distance(zsource)/1000
    hf.create_dataset(f'images/strong_lens_{i}/d_s', data=np.array([str(DS),'Angular diameter distance to source galaxy in Gpc'],dtype=dt),dtype=dt)
    DLS =  dlu_1_results.cosmology.angular_diameter_distance_z1z2(zdeflector,zsource)/1000
    hf.create_dataset(f'images/strong_lens_{i}/d_ls', data=np.array([str(DLS),'Angular diameter distance between deflector and source galaxy in Gpc'],dtype=dt),dtype=dt)
    hf.create_dataset(f'images/strong_lens_{i}/exposure_time', data=np.array([str(image_results['tot_exp_times'][0]),str(image_results['tot_exp_times'][1]),str(image_results['tot_exp_times'][2]), 'Exposure time in seconds for all bands'],dtype=dt),dtype=dt)
    hf.create_dataset(f'images/strong_lens_{i}/instrument', data=np.array([Instrument, 'Instrument'],dtype=dt),dtype=dt)
    hf.create_dataset(f'images/strong_lens_{i}/log_mhigh', data=np.array([str(np.log10(sampled_vals.max_subhalo_mass)),'Log10 of largest possible subhalo mass'],dtype=dt),dtype=dt)
    hf.create_dataset(f'images/strong_lens_{i}/log_mlow', data=np.array([str(6),'Log10 of lowest possible subhalo mass'],dtype=dt),dtype=dt)
    hf.create_dataset(f'images/strong_lens_{i}/host_mass', data=np.array([str(np.log10(dlu_1_results.host_mass)),'Log10 of mass of host lens in units of M_sun'],dtype=dt),dtype=dt)
    hf.create_dataset(f'images/strong_lens_{i}/lens_mass', data=np.array([str(np.log10(dlu_1_results.whole_halo_mass)),'Log10 of mass of entire lens in units of M_sun'],dtype=dt),dtype=dt)
    hf.create_dataset(f'images/strong_lens_{i}/num_subhalos', data=np.array([str(dlu_1_results.num_subhalos), 'Number of subhalos'],dtype=dt),dtype=dt)
    hf.create_dataset(f'images/strong_lens_{i}/r_tidal', data=np.array([str(0.25), 'see Pyhalo documentation'],dtype=dt),dtype=dt)
    hf.create_dataset(f'images/strong_lens_{i}/sigma_sub', data=np.array([str(0.025), 'see Pyhalo documentation'],dtype=dt),dtype=dt)
    hf.create_dataset(f'images/strong_lens_{i}/snr', data=np.array([str(image_results['SNR']), 'SNR of lensed in all bands'],dtype=dt),dtype=dt)
    hf.create_dataset(f'images/strong_lens_{i}/theta_e', data=np.array([str(dlu_1_results.macro_kwargs_list[0]['theta_E']), 'Einstein radius in arcseconds'],dtype=dt),dtype=dt)
    hf.create_dataset(f'images/strong_lens_{i}/uid',data=np.array([str(i),f'simulation number in {DM_type} batch created on {timestamp}'],dtype=dt),dtype=dt)
    hf.create_dataset(f'images/strong_lens_{i}/z_lens', data=np.array([str(zdeflector),'Redshift of deflector'],dtype=dt),dtype=dt)
    hf.create_dataset(f'images/strong_lens_{i}/z_source', data=np.array([str(zsource),'Redshift of source'],dtype=dt),dtype=dt)
    hf.create_dataset(f'images/strong_lens_{i}/bands', data=np.array([dlu_2_results.band_labels[0],dlu_2_results.band_labels[1],dlu_2_results.band_labels[2], 'Instrument bands in which image was simulated'],dtype=dt),dtype=dt)
    hf.create_dataset(f'images/strong_lens_{i}/DM_type',data = np.array([DM_type, 'Type of dark matter assumed'],dtype=dt),dtype=dt)
    hf.create_dataset(f'images/strong_lens_{i}/source_pos',data = np.array([str([setup_results['source_x'],setup_results['source_y']]), 'Plane coordinates of source wrt center of deflector (in arcseconds)'],dtype=dt),dtype=dt)
    hf.create_dataset(f'images/strong_lens_{i}/host_slope',data = np.array([str(dlu_1_results.slope_Host), 'EPL slope of host halo'],dtype=dt),dtype=dt)
    hf.create_dataset(f'images/strong_lens_{i}/ellipticity',data = np.array([str([dlu_1_results.macro_kwargs_list[0]['e1'],dlu_1_results.macro_kwargs_list[0]['e2']]), 'Ellipticity values for host halo mass profile'],dtype=dt),dtype=dt)


    if DM_type == 'WDM':
         hf.create_dataset(f'images/strong_lens_{i}/log_mc',data = np.array([str(dlu_1_results.type_kwargs['log_mc']), 'Mass at which underneath the WDM mass function is surpressed when compared to CDM'],dtype=dt),dtype=dt)
    elif DM_type == 'Axion':
         hf.create_dataset(f'images/strong_lens_{i}/m_axion',data = np.array([str(dlu_1_results.type_kwargs['M_axion']), 'log10 of mass of axion particle'],dtype=dt),dtype=dt)
         hf.create_dataset(f'images/strong_lens_{i}/flucs_shape',data = np.array([dlu_1_results.type_kwargs['flucs_shape'], 'Shape at which to render fluctuations of the host halo (See Pyhalo)'],dtype=dt),dtype=dt)
         hf.create_dataset(f'images/strong_lens_{i}/flucs_args',data = np.array([str(dlu_1_results.type_kwargs['flucs_args']['angle']),str(dlu_1_results.type_kwargs['flucs_args']['rmin']),str(dlu_1_results.type_kwargs['flucs_args']['rmax']), 'Arugments corresponding to geometry of host halo fluctuations (See Pyhalo)'],dtype=dt),dtype=dt)
    elif DM_type == 'SIDM':
         hf.create_dataset(f'images/strong_lens_{i}/subhalo_mass_ranges',data = np.array([str(dlu_1_results.type_kwargs['mass_ranges_subhalos']), 'Mass ranges at which subhalos are sampled from'],dtype=dt),dtype=dt)
         hf.create_dataset(f'images/strong_lens_{i}/field_halo_mass_ranges',data = np.array([str(dlu_1_results.type_kwargs['mass_ranges_field_halos']), 'Mass ranges at which field halos are sampled from'],dtype=dt),dtype=dt)
         hf.create_dataset(f'images/strong_lens_{i}/prob_subhalo',data = np.array([str(dlu_1_results.type_kwargs['probabilities_subhalos']), 'Probability of subhalo being sampled from corresponding mass range'],dtype=dt),dtype=dt)
         hf.create_dataset(f'images/strong_lens_{i}/prob_field_halo',data = np.array([str(dlu_1_results.type_kwargs['probabilities_field_halos']), 'Probability of field halo being sampled from corresponding mass range'],dtype=dt),dtype=dt)


    exposure_0=hf.create_dataset(f'images/strong_lens_{i}/exposure_{i}_{dlu_2_results.band_labels[0]}',data = image_results['img'][0])
    exposure_0.attrs['filter'] = np.array([dlu_2_results.band_labels[0],'Filter'],dtype=dt)
    exposure_0.attrs['fov'] = np.array([str(dlu_1_results.arcsecond_opening_angle),'Field of view [arcsec]'],dtype=dt)
    exposure_0.attrs['pixel_scale'] = np.array([str(dlu_2_results.bands[0]['pixel_scale']),'Pixel scale [arcsec/pixel]'],dtype=dt)
    exposure_0.attrs['lens_magnitude'] = np.array([str(dlu_2_results.deflector_mag[0]),'Lens magnitude'],dtype=dt)
    exposure_0.attrs['source_magnitude'] = np.array([str(dlu_2_results.source_mag[0]),'Unlensed source galaxy magnitude'],dtype=dt)
    exposure_0.attrs['units'] = np.array(['counts','Units of pixel values'],dtype=dt)
    hf.create_dataset(f'images/strong_lens_{i}/exposure_{i}_{dlu_2_results.band_labels[0]}_nss',data = image_results['img_nss'][0])

    exposure_1=hf.create_dataset(f'images/strong_lens_{i}/exposure_{i}_{dlu_2_results.band_labels[1]}',data = image_results['img'][1])
    exposure_1.attrs['filter'] = np.array([dlu_2_results.band_labels[1],'Filter'],dtype=dt)
    exposure_1.attrs['fov'] = np.array([str(dlu_1_results.arcsecond_opening_angle),'Field of view [arcsec]'],dtype=dt)
    exposure_1.attrs['pixel_scale'] = np.array([str(dlu_2_results.bands[1]['pixel_scale']),'Pixel scale [arcsec/pixel]'],dtype=dt)
    exposure_1.attrs['lens_magnitude'] = np.array([str(dlu_2_results.deflector_mag[1]),'Lens magnitude'],dtype=dt)
    exposure_1.attrs['source_magnitude'] = np.array([str(dlu_2_results.source_mag[1]),'Unlensed source galaxy magnitude'],dtype=dt)
    exposure_1.attrs['units'] = np.array(['counts','Units of pixel values'],dtype=dt)
    hf.create_dataset(f'images/strong_lens_{i}/exposure_{i}_{dlu_2_results.band_labels[1]}_nss',data = image_results['img_nss'][1])
    
    exposure_2=hf.create_dataset(f'images/strong_lens_{i}/exposure_{i}_{dlu_2_results.band_labels[2]}',data = image_results['img'][2])
    exposure_2.attrs['filter'] = np.array([dlu_2_results.band_labels[2],'Filter'],dtype=dt)
    exposure_2.attrs['fov'] = np.array([str(dlu_1_results.arcsecond_opening_angle),'Field of view [arcsec]'],dtype=dt)
    exposure_2.attrs['pixel_scale'] = np.array([str(dlu_2_results.bands[2]['pixel_scale']),'Pixel scale [arcsec/pixel]'],dtype=dt)
    exposure_2.attrs['lens_magnitude'] = np.array([str(dlu_2_results.deflector_mag[2]),'Lens magnitude'],dtype=dt)
    exposure_2.attrs['source_magnitude'] = np.array([str(dlu_2_results.source_mag[2]),'Unlensed source galaxy magnitude'],dtype=dt)
    exposure_2.attrs['units'] = np.array(['counts','Units of pixel values'],dtype=dt)
    hf.create_dataset(f'images/strong_lens_{i}/exposure_{i}_{dlu_2_results.band_labels[2]}_nss',data = image_results['img_nss'][2])
    
