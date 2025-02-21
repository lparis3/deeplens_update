import numpy as np
import matplotlib.pyplot as plt

def dl_sim(DM_Type,Instrument):
    '''Adaptive interface to simulate mock lensing images.'''
    
    #1. Configure instrument specific parameters
    def instrument_config(Instrument):
        if Instrument == 'Roman':
            from lenstronomy.SimulationAPI.ObservationConfig.Roman import Roman
            Roman_g = Roman(band='F062', psf_type='PIXEL', survey_mode='wide_area')
            Roman_r = Roman(band='F106', psf_type='PIXEL', survey_mode='wide_area')
            Roman_i = Roman(band='F184', psf_type='PIXEL', survey_mode='wide_area')
            roman = [Roman_g, Roman_r, Roman_i]
            print('Roman config set!')
            return roman
        
        elif Instrument == 'LSST':
            from lenstronomy.SimulationAPI.ObservationConfig.LSST import LSST
            LSST_g = LSST(band='g', psf_type='GAUSSIAN', coadd_years=10)
            LSST_r = LSST(band='r', psf_type='GAUSSIAN', coadd_years=10)
            LSST_i = LSST(band='i', psf_type='GAUSSIAN', coadd_years=10)
            lsst = [LSST_g, LSST_r, LSST_i]
            print('LSST config set!')
            return lsst
        
        elif Instrument == 'DES':
            from lenstronomy.SimulationAPI.ObservationConfig.DES import DES
            DES_g = DES(band = 'g',psf_type='GAUSSIAN',coadd_years=3)
            DES_r = DES(band = 'r',psf_type='GAUSSIAN',coadd_years=3)
            DES_i = DES(band = 'i',psf_type='GAUSSIAN',coadd_years=3)
            des = [DES_g,DES_r,DES_i]
            print('DES config set!')
            return des
    
    instrument_param = instrument_config(Instrument=Instrument)
    band_g, band_r, band_i = instrument_param
    kwargs_g_band = band_g.kwargs_single_band()
    kwargs_r_band = band_r.kwargs_single_band()
    kwargs_i_band = band_i.kwargs_single_band()
    bands = [kwargs_g_band,kwargs_r_band,kwargs_i_band]

    
    #2.Observational data extraction for input images 
    from deeplens_sim.obs_data_extraction import extraction 
    
    source_images,source_mag,deflector_images,deflector_mag, redshifts = extraction()  


    #3.  Construct host halo, sub halos, and field halos 
    from deeplens_sim.halo_creation import Halo_constructor

    if DM_Type == 'CDM':
      lens_model_list,lens_kwargs_list,lens_redshift_list,cosmology,Macro_model_list, Macro_kwargs_list, Macro_redshift_list,arcsecond_opening_angle,Host_mass=Halo_constructor(DM_Type,redshifts)
    elif DM_Type == 'WDM':
      lens_model_list,lens_kwargs_list,lens_redshift_list,cosmology,Macro_model_list, Macro_kwargs_list, Macro_redshift_list, log_mc, arcsecond_opening_angle,Host_mass = Halo_constructor(DM_Type,redshifts)
    elif DM_Type == 'SIDM':
      lens_model_list,lens_kwargs_list,lens_redshift_list,cosmology,Macro_model_list, Macro_kwargs_list, Macro_redshift_list, mass_ranges_subhalos, mass_ranges_field_halos, probabilities_subhalos, probabilities_field_halos, arcsecond_opening_angle, Host_mass=Halo_constructor(DM_Type,redshifts)
    elif DM_Type == 'Axion':
      lens_model_list,lens_kwargs_list,lens_redshift_list,cosmology,Macro_model_list, Macro_kwargs_list, Macro_redshift_list,M_axion,arcsecond_opening_angle, Host_mass=Halo_constructor(DM_Type,redshifts)

    print('Halos constructed!')

    #4.Consolidate lens and source models and finalize kwargs
    kwargs_model = {'lens_model_list': lens_model_list,  # list of lens models to be used
                'lens_redshift_list': lens_redshift_list,
                'lens_light_model_list': ['INTERPOL'],  # list of unlensed light models to be used
                'source_light_model_list': ['INTERPOL'],  # list of extended source models to be used, here we used the interpolated real galaxy
                'z_source':zsource,
                'cosmo': cosmology}
    
    kwargs_model_nss = {'lens_model_list': Macro_model_list,  # list of lens models to be used
                'lens_redshift_list': Macro_redshift_list,
                'lens_light_model_list': ['INTERPOL'],  # list of unlensed light models to be used
                'source_light_model_list': ['INTERPOL'],  # list of extended source models to be used, here we used the interpolated real galaxy
                'z_source':zsource,
                'cosmo': cosmology}
    
    source_pos_xx,source_pos_yy = np.random.uniform(-Macro_kwargs_list[0]['theta_E']*0.3, Macro_kwargs_list[0]['theta_E']*0.3), np.random.uniform(-Macro_kwargs_list[0]['theta_E']*0.3, Macro_kwargs_list[0]['theta_E']*0.3)

    band_g, band_r, band_i = instrument_param
    kwargs_g_band = band_g.kwargs_single_band()
    kwargs_r_band = band_r.kwargs_single_band()
    kwargs_i_band = band_i.kwargs_single_band()
    bands = [kwargs_g_band,kwargs_r_band,kwargs_i_band]

    def kwargs_light_mag(posxx,posyy,pixel_scale):
        '''Returns kwargs_single_band values for g,r,and i while taking into account intrument of simulated image.'''
        # g-band
        # lens light
        kwargs_lens_light_mag_g = [{'magnitude': deflector_mag[0], 'image':deflector_images[0],'center_x': 0.0, 'center_y': 0.0, 'phi_G': 0.0, 'scale': pixel_scale}]
        # source light
        kwargs_source_mag_g = [{'magnitude': source_mag[0], 'image': source_images[0], 'center_x': posxx, 'center_y': posyy, 'phi_G': 0.0, 'scale': pixel_scale}]


        # r-band
        # lens light
        kwargs_lens_light_mag_r = [{'magnitude': deflector_mag[1], 'image':deflector_images[1],'center_x': 0.0, 'center_y': 0.0, 'phi_G': 0.0, 'scale': pixel_scale}]
        # source light
        kwargs_source_mag_r = [{'magnitude': source_mag[1], 'image': source_images[1], 'center_x': posxx, 'center_y': posyy, 'phi_G': 0.0, 'scale': pixel_scale}]


        # i-band
        # lens light
        kwargs_lens_light_mag_i = [{'magnitude': deflector_mag[2], 'image':deflector_images[2],'center_x': 0.0, 'center_y': 0.0, 'phi_G': 0.0, 'scale': pixel_scale}]
        # source light
        kwargs_source_mag_i =[{'magnitude': source_mag[2], 'image': source_images[2], 'center_x': posxx, 'center_y':posyy, 'phi_G': 0.0, 'scale': pixel_scale}]

        kwargs_lens_light_mag = [kwargs_lens_light_mag_g,kwargs_lens_light_mag_r,kwargs_lens_light_mag_i]
        kwargs_source_mag = [kwargs_source_mag_g,kwargs_source_mag_r,kwargs_source_mag_i]

        return kwargs_lens_light_mag, kwargs_source_mag

    kwargs_lens_light_mag, kwargs_source_mag = kwargs_light_mag(source_pos_xx,source_pos_yy,kwargs_b_band['pixel_scale'])

    print('Ready to simulate!')

    #5. Simulate Image with and without substructure
    from lenstronomy.SimulationAPI.sim_api import SimAPI
    import lenstronomy.Plots.plot_util as plot_util

    kwargs_numerics = {'point_source_supersampling_factor': 1}
    
    def simulate(kwargs_numerics,band_kwargs,lens_light_kwargs,source_light_kwargs,lens_nonlight_kwargs,kwargs_model_):
        numpix = 64

        sim_g = SimAPI(numpix=numpix, kwargs_single_band=band_kwargs[0], kwargs_model=kwargs_model)
        sim_r = SimAPI(numpix=numpix, kwargs_single_band=band_kwargs[1], kwargs_model=kwargs_model)
        sim_i = SimAPI(numpix=numpix, kwargs_single_band=band_kwargs[2], kwargs_model=kwargs_model)

        imSim_g = sim_g.image_model_class(kwargs_numerics)
        imSim_r = sim_r.image_model_class(kwargs_numerics)
        imSim_i = sim_i.image_model_class(kwargs_numerics)

        kwargs_lens_light_g, kwargs_source_g,_ = sim_g.magnitude2amplitude(lens_light_kwargs[0], source_light_kwargs[0])
        kwargs_lens_light_r, kwargs_source_r,_ = sim_r.magnitude2amplitude(lens_light_kwargs[1], source_light_kwargs[1])
        kwargs_lens_light_i, kwargs_source_i,_ = sim_i.magnitude2amplitude(lens_light_kwargs[2], source_light_kwargs[2])

        image_g = imSim_g.image(lens_nonlight_kwargs, kwargs_source_g, kwargs_lens_light_g,point_source_add=False,lens_light_add=True)
        image_r = imSim_r.image(lens_nonlight_kwargs, kwargs_source_r, kwargs_lens_light_r,point_source_add=False,lens_light_add=True)
        image_i = imSim_i.image(lens_nonlight_kwargs, kwargs_source_i, kwargs_lens_light_i,point_source_add=False,lens_light_add=True)


        img = np.zeros((image_g.shape[0], image_g.shape[1], 3), dtype=float)
       
        img[:,:,0] = image_g
        img[:,:,1] = image_r
        img[:,:,2] = image_i
        data_class = sim_g.data_class
        return img, data_class

    img, coords = simulate(kwargs_numerics=kwargs_numerics,band_kwargs=bands,lens_light_kwargs=kwargs_lens_light_mag,source_light_kwargs=kwargs_source_mag,lens_nonlight_kwargs=lens_kwargs_list,kwargs_model_=kwargs_model)
    img_nss,coords = simulate(kwargs_numerics,band_kwargs=bands,lens_light_kwargs = kwargs_lens_light_mag, source_light_kwargs = kwargs_source_mag,lens_nonlight_kwargs = lens_kwargs_list, kwargs_model=kwargs_model_nss)

    sns_diff = img/img_nss

    #Prepare Outputs
    if DM_Type == 'CDM':
        instr_dict = {'name': Instrument, 'pixel_scale': kwargs_g_band['pixel_scale'],'psf':kwargs_g_band['psf_type']}
        source_dict = {'zsource':zsource,'mag_src': source_mag, 'phi_G':kwargs_source_mag[0][0]['phi_G'],'center':np.array([source_pos_xx,source_pos_yy])}
        ext_shear_dict = {'gamma_1':Macro_kwargs_list[1]['gamma1'], 'gamma_2':Macro_kwargs_list[1]['gamma2']}
        sub_defl_dict = {'arcsec_opening':arcsecond_opening_angle}
        host_defl_dict = {'zdeflector':zdeflector,'host_mass':Host_mass,'host_slope':Host_slope,'theta_E':Macro_kwargs_list[0]['theta_E'],'ellipticity':np.array([Macro_kwargs_list[0]['e1'],Macro_kwargs_list[0]['e2']])}
        defl_dict = {'subhalos':sub_defl_dict,'host':host_defl_dict,'ext_shear':ext_shear_dict}
        image_dict = {'image':img,'image_no_sub':img_nss,'contrast':sns_diff}
        output = {'image': image_dict, 'deflector':defl_dict,'source':source_dict,'instrument':instr_dict}

    if DM_Type == 'Axion':
        instr_dict = {'name': Instrument, 'pixel_scale': kwargs_g_band['pixel_scale'],'psf':kwargs_g_band['psf_type']}
        source_dict = {'zsource':zsource,'mag_src': source_mag, 'phi_G':kwargs_source_mag[0][0]['phi_G'],'center':np.array([source_pos_xx,source_pos_yy])}
        ext_shear_dict = {'gamma_1':Macro_kwargs_list[1]['gamma1'], 'gamma_2':Macro_kwargs_list[1]['gamma2']}
        sub_defl_dict = {'arcsec_opening':arcsecond_opening_angle, 'axion_mass':M_axion}
        host_defl_dict = {'zdeflector':zdeflector,'host_mass':Host_mass,'host_slope':Host_slope,'theta_E':Macro_kwargs_list[0]['theta_E'],'ellipticity':np.array([Macro_kwargs_list[0]['e1'],Macro_kwargs_list[0]['e2']])}
        defl_dict = {'subhalos':sub_defl_dict,'host':host_defl_dict,'ext_shear':ext_shear_dict}
        image_dict = {'image':img,'image_no_sub':img_nss,'contrast':sns_diff}
        output = {'image': image_dict, 'deflector':defl_dict,'source':source_dict,'instrument':instr_dict}
    
    if DM_Type == 'WDM':
        instr_dict = {'name': Instrument, 'pixel_scale': kwargs_g_band['pixel_scale'],'psf':kwargs_g_band['psf_type']}
        source_dict = {'zsource':zsource,'mag_src': source_mag, 'phi_G':kwargs_source_mag[0][0]['phi_G'],'center':np.array([source_pos_xx,source_pos_yy])}
        ext_shear_dict = {'gamma_1':Macro_kwargs_list[1]['gamma1'], 'gamma_2':Macro_kwargs_list[1]['gamma2']}
        sub_defl_dict = {'arcsec_opening':arcsecond_opening_angle,'supressed_mass':log_mc}
        host_defl_dict = {'zdeflector':zdeflector,'host_mass':Host_mass,'host_slope':Host_slope,'theta_E':Macro_kwargs_list[0]['theta_E'],'ellipticity':np.array([Macro_kwargs_list[0]['e1'],Macro_kwargs_list[0]['e2']])}
        defl_dict = {'subhalos':sub_defl_dict,'host':host_defl_dict,'ext_shear':ext_shear_dict}
        image_dict = {'image':img,'image_no_sub':img_nss,'contrast':sns_diff}
        output = {'image': image_dict, 'deflector':defl_dict,'source':source_dict,'instrument':instr_dict}

    if DM_Type == 'SIDM':
        instr_dict = {'name': Instrument, 'pixel_scale': kwargs_g_band['pixel_scale'],'psf':kwargs_g_band['psf_type']}
        source_dict = {'zsource':zsource,'mag_src': source_mag, 'phi_G':kwargs_source_mag[0][0]['phi_G'],'center':np.array([source_pos_xx,source_pos_yy])}
        ext_shear_dict = {'gamma_1':Macro_kwargs_list[1]['gamma1'], 'gamma_2':Macro_kwargs_list[1]['gamma2']}
        sub_defl_dict = {'arcsec_opening':arcsecond_opening_angle,'subhalo_mass_ranges':mass_ranges_subhalos,'field_halos_mass_ranges':mass_ranges_field_halos,'prob_subhalo':probabilities_subhalos,'prob_field_halo':probabilities_field_halos}
        host_defl_dict = {'zdeflector':zdeflector,'host_mass':Host_mass,'host_slope':Host_slope,'theta_E':Macro_kwargs_list[0]['theta_E'],'ellipticity':np.array([Macro_kwargs_list[0]['e1'],Macro_kwargs_list[0]['e2']])}
        defl_dict = {'subhalos':sub_defl_dict,'host':host_defl_dict,'ext_shear':ext_shear_dict}
        image_dict = {'image':img,'image_no_sub':img_nss,'contrast':sns_diff}
        output = {'image': image_dict, 'deflector':defl_dict,'source':source_dict,'instrument':instr_dict}

    print('Done!')
    return output


    
