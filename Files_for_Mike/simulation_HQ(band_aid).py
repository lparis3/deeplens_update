
#%%
#SIMULATION CODE
import os
import traceback
import datetime
import matplotlib.pyplot as plt
import time
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.ndimage import shift


def dlu_sim_1(Instrument):
    '''Adaptive interface to simulate lensing images.'''

    #1. Configure instrument specific parameters
    start1 = time.time()
    def instrument_config(Instrument):
        if Instrument == 'Roman':
            from lenstronomy.SimulationAPI.ObservationConfig.Roman import Roman
            band1 = 'F062'
            band2 = 'F106'
            band3 = 'F184'
            Roman_g = Roman(band=band1, psf_type='PIXEL', survey_mode='wide_area')
            Roman_r = Roman(band=band2, psf_type='PIXEL', survey_mode='wide_area')
            Roman_i = Roman(band=band3, psf_type='PIXEL', survey_mode='wide_area')
            roman = [Roman_g, Roman_r, Roman_i]
            return roman, [band1,band2,band3]
        
        elif Instrument == 'LSST':
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
    
    instrument_param,band_labels = instrument_config(Instrument=Instrument)
    band_g, band_r, band_i = instrument_param
    kwargs_g_band = band_g.kwargs_single_band()
    kwargs_r_band = band_r.kwargs_single_band()
    kwargs_i_band = band_i.kwargs_single_band()
    bands = [kwargs_g_band,kwargs_r_band,kwargs_i_band]

    end1 = time.time()
    print(f'Step 1 took {end1-start1} secs')
    #2.Data Extraction
    start2 = time.time()
    from scipy.ndimage import gaussian_filter
    import numpy as np

     
            
    def center_extraction(image):
        brightest_pixel_values = np.array([np.max(image[0,:,:]), np.max(image[1,:,:]),np.max(image[2,:,:])])
        brightest_band = brightest_pixel_values.argmax()
        center_rows_columns = np.unravel_index(image[brightest_band,:,:].argmax(),image[brightest_band,:,:].shape)
        center_pixel_coor = np.array([center_rows_columns[1], center_rows_columns[0]])
        return center_pixel_coor


    def extraction():
        '''Sample GalaxiesML Dataset for sources and deflectors.'''
        data_file = 'deeplens_update.hdf5'
        with h5py.File(data_file, 'r') as file:
            redshifts = np.array([file['specz_redshift']])[0]
            #Deflector id
            idd_array = np.where((redshifts <= (np.float64(2))))[0] 
            idd = np.random.choice(idd_array)
            zdeflector = (redshifts[idd])
            
            #Source id
            ids_array = np.array(np.where(redshifts >= zdeflector + 0.25))[0]
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
            zdeflector = np.round(file['specz_redshift'][idd],2)
            
            
            #Source Info:
            source_morph = file['image'][ids]
            center_s = center_extraction(source_morph)
            sigma_y_s = file['g_half_light_radius'][ids]*2. 
            sigma_x_s = file['g_half_light_radius'][ids]*2. * (1 - file['g_ellipticity'][ids])  
            angle_s = np.pi/2. - np.deg2rad(file['g_pos_angle'][ids]) 
            g_mag_s = file['g_cmodel_mag'][ids]
            r_mag_s = file['r_cmodel_mag'][ids]
            i_mag_s = file['i_cmodel_mag'][ids]
            zsource = np.round(file['specz_redshift'][ids],2)
            

            #Necessary functions for processing:
            def create_elliptical_gaussian(shape, center, sigma_x, sigma_y, angle=0):
                """Generate a 2D elliptical Gaussian mask."""
                y, x = np.indices(shape)
                x0, y0 = center
                x = x - x0
                y = y - y0

                # Apply rotation for the angle
                x_rot = x * np.cos(angle) - y * np.sin(angle)
                y_rot = x * np.sin(angle) + y * np.cos(angle)
                
                # Elliptical Gaussian function
                mask = np.exp(-((x_rot**2 / (2 * sigma_x**2)) + (y_rot**2 / (2 * sigma_y**2))))
                return mask

            def src_process(band_image,center,sigma_x,sigma_y,angle):   
                # Center source or deflector in image
                x_shift = np.shape(band_image)[0]/2 - center[0]
                y_shift = np.shape(band_image)[1]/2 - center[1]
                band_image = shift(band_image,[y_shift,x_shift],mode='constant')
                center = [np.shape(band_image)[0]/2,np.shape(band_image)[0]/2]
                
                # Background estimation (simple mean of edge pixels for example)
                background_level = np.mean(np.concatenate([band_image[0, :], band_image[-1, :], band_image[:, 0], band_image[:, -1]]))
                band_image -= background_level  # Subtract background
                
                #Remove all unphysical negative pixels
                band_image[band_image < 0] = 0

                # Apply Gaussian convolution to smooth the image
                band_image = gaussian_filter(band_image, sigma=1)  # What sigma is best for us? 

                # Create elliptical Gaussian mask
                mask = create_elliptical_gaussian(band_image.shape, center, sigma_x, sigma_y, angle)

                # Apply mask to isolate the central galaxy
                masked_image = band_image * mask
                return masked_image

        #Processing of source and deflector images:
            src_g = src_process(source_morph[0,:,:],center_s,sigma_x_s,sigma_y_s,angle_s)
            src_r = src_process(source_morph[1,:,:],center_s,sigma_x_s,sigma_y_s,angle_s)
            src_i = src_process(source_morph[2,:,:],center_s,sigma_x_s,sigma_y_s,angle_s)

            dfr_g = src_process(deflector_morph[0,:,:],center_d,sigma_x_d,sigma_y_d,angle_d)
            dfr_r = src_process(deflector_morph[1,:,:],center_d,sigma_x_d,sigma_y_d,angle_d)
            dfr_i = src_process(deflector_morph[2,:,:],center_d,sigma_x_d,sigma_y_d,angle_d)
            

        def mag_booster(source_image,deflector_image,source_mag, deflector_mag):
            '''(maybe) Boosts the magnitude of lens until brightest pixel is brighter than corresponding pixel of deflector'''
            new_mag = source_mag
            if np.max(source_image)>deflector_image[np.unravel_index(np.argmax(source_image), np.shape(source_image))]:
                if np.random.uniform(0.0,1.0,None) < 0.5:
                    new_mag = deflector_mag - 0.5
                    return new_mag
                else:
                    return new_mag
            else:
                new_mag = source_mag - np.random.uniform(0.0,0.5,None)
                return new_mag
           
        
        source_images = np.array([src_g,src_i,src_r]) #processed source morphology
        source_mag = np.array([g_mag_s,r_mag_s,i_mag_s])
        raw_info_src = {'raw_img': source_morph, 'pros_img': source_images, 'raw_center':center_s}
        
        deflector_images = np.array([dfr_g,dfr_r,dfr_i])
        deflector_mag = np.array([g_mag_d,r_mag_d,i_mag_d])
        raw_info_dfr = {'raw_img': deflector_morph, 'pros_img': deflector_images, 'raw_center':center_d}

        source_mag_boost = mag_booster(source_images,deflector_images,source_mag,deflector_mag)
        redshifts = [zdeflector,zsource]
        return source_images,source_mag_boost,deflector_images,deflector_mag, redshifts, raw_info_src, raw_info_dfr

    source_images,source_mag,deflector_images,deflector_mag, redshifts, raw_src, raw_dfr = extraction()  

    end2 = time.time()
    print(f'Step 2 took {end2-start2} seconds')
    return bands,source_images,source_mag,deflector_images,deflector_mag, redshifts, raw_src, raw_dfr, start1, band_labels


class SkipSimulation_theta_E1(Exception):
        pass
class SkipSimulation_subhalo_frac1(Exception):
        pass
class SkipSimulation_snr1(Exception):
        pass

def dlu_sim_2(DM_Type,Instrument,et_quotient,bands,source_images,source_mag,deflector_images,deflector_mag, redshifts, raw_src, raw_dfr,start_time,band_labels,Interlopers = True, lens_light = False):  
    #3. Construct host halo, sub halos, and field halos 
    start3 = time.time()
    import pyHalo
    import pyHalo.preset_models
    from astropy import units as u
    from astropy.constants import G, c, M_sun


    class SkipSimulation_theta_E2(Exception):
        pass

    class SkipSimulation_subhalo_frac2(Exception):
        pass


    if Interlopers == True:
        LOS = 1.0
    else:
        LOS = 0.0

    def Host_mass(mean = 13, sigma =1.0):
        '''Samples log of mass of ENTIRE halo from normal distribution'''
        M_host = np.random.normal(loc = mean,scale = sigma,size =1)
        return M_host[0]

    def Host_slope(gammaL = 1.9, gammaH = 2.2):
        '''Samples host halo's log slope between lower and upper bounds from Gilman et al. 2022'''
        return np.random.uniform(gammaL,gammaH,None)
    


    def CDM_constructor(zsource, zlens, M_host, log_mlow,log_mhigh,Host_gamma, LOS_Norm):
        '''This function constructs a lens (host halo, sub halo, LOS halos, and lens galaxy) under the assumption of CDM'''
        arcsec_opening_angle = 10

        #First, use Pyhalo to create subhalo realization
        Subhalo_constructor= pyHalo.preset_models.preset_model_from_name('CDM')
        Subhalo_realization = Subhalo_constructor(z_lens=zlens,z_source=zsource,log_m_host=M_host,log_mlow=log_mlow,log_mhigh=log_mhigh,cone_opening_angle_arcsec=arcsec_opening_angle, LOS_normalization=LOS_Norm) 

        #Extract both lensing quantities from subhalo realization (reshifts, lensing models, lensing kwargs) and the cosmology instance from source to observer
        Sub_model_list, Sub_redshift_array, Sub_kwargs, _ = Subhalo_realization.lensing_quantities()
        cosmology = Subhalo_realization.astropy_instance

        #Find masses of halo system 
        subhalo_Halo_masses = [halo.mass * M_sun.value for halo in Subhalo_realization.halos]
        Host_lens_mass = (10**M_host * M_sun.value) 
        M_Whole_lens = Host_lens_mass + np.sum(subhalo_Halo_masses)

        Subhalo_masses_sum = np.sum([halo.mass * M_sun.value for halo in Subhalo_realization.halos])
        subhalo_mass_percentage = Subhalo_masses_sum/(M_Whole_lens)
        if subhalo_mass_percentage > 0.1:
            raise SkipSimulation_subhalo_frac2


        #Calc Einstein radius
        def mass_to_radius(Mass,redshift_src,redshift_def):
            M_Halo = Mass
            rad_to_arcsec = 206265

            DL = cosmology.angular_diameter_distance(redshift_def).to(u.m)
            DS = cosmology.angular_diameter_distance(redshift_src).to(u.m)
            DLS = cosmology.angular_diameter_distance_z1z2(redshift_def,redshift_src).to(u.m)

            # Einstein radius
            theta = np.sqrt(4 * G * M_Halo/c**2 * DLS/(DL*DS))

            # Return radius in arcsecods
            radius_arcsec = theta * rad_to_arcsec

            return radius_arcsec.value 

        theta_E = mass_to_radius(Host_lens_mass,zsource,zdeflector)

        if (theta_E) > 6.0:
            print("Ignoring simulation: Theta_E > 4.0 arcsec")
            raise SkipSimulation_theta_E2

        #Host halo
        Host_halo = 'EPL'
        Host_kwargs = {'theta_E':theta_E,'gamma':Host_gamma,'e1':0.0,'e2':0.0,'center_x':0.0, 'center_y':0.0} 

        #External shear
        External_shear = 'SHEAR'
        Shear_kwargs = {'gamma1':0.05,'gamma2':0.00,'ra_0': 0.0, 'dec_0': 0.0} 

        #Macrolens
        Macro_model_list = [Host_halo,External_shear]
        Macro_kwargs_list = [Host_kwargs,Shear_kwargs]
        Macro_redshift_list = [zlens,zlens]

        #Combine macro-model with substructure 
        lens_model_list = Macro_model_list + Sub_model_list
        lens_kwargs_list = Macro_kwargs_list + Sub_kwargs
        lens_redshift_list = Macro_redshift_list + list(Sub_redshift_array)

        
        return lens_model_list,lens_kwargs_list,lens_redshift_list,cosmology,Macro_model_list, Macro_kwargs_list, Macro_redshift_list, arcsec_opening_angle,M_host,M_Whole_lens,len(Subhalo_realization.halos)

    def Axion_constructor(zsource, zlens, M_host, log_mlow,log_mhigh,Host_gamma, LOS_Norm):
        '''This function constructs a lens (host halo, sub halo, LOS halos, and lens galaxy) under the assumption of axionic dark matter'''
        
        M_axion = np.random.uniform(-22.0,-19.0,None)
        flucs_shape='ring'
        arcsec_opening_angle = 10
        debrogile_wavelength_order = 0.6 * (10**(-22)/10**M_axion)
        rmax = 1.1 * debrogile_wavelength_order 
        rmin = 0.9 * debrogile_wavelength_order 
        flucs_args={'angle': np.random.uniform(0.0,2*np.pi,None), 'rmin': rmin, 'rmax': rmax}


        #First, use Pyhalo to create subhalo realization
        Subhalo_constructor= pyHalo.preset_models.preset_model_from_name('ULDM')
        Subhalo_realization = Subhalo_constructor(z_lens=zlens,z_source=zsource,log10_m_uldm = M_axion ,flucs_shape=flucs_shape,flucs_args=flucs_args,log_m_host=M_host,log_mlow=log_mlow,log_mhigh=log_mhigh,cone_opening_angle_arcsec=arcsec_opening_angle, LOS_normalization=LOS_Norm) 
      
        
        #Extract both lensing quantities from subhalo realization (reshifts, lensing models, lensing kwargs) and the cosmology instance from source to observer
        Sub_model_list, Sub_redshift_array, Sub_kwargs, _ = Subhalo_realization.lensing_quantities()
        cosmology = Subhalo_realization.astropy_instance
        
        #Find masses of halo system 
        subhalo_Halo_masses = [halo.mass * M_sun.value for halo in Subhalo_realization.halos]
        Host_lens_mass = (10**M_host * M_sun.value) 
        M_Whole_lens = Host_lens_mass + np.sum(subhalo_Halo_masses)

        Subhalo_masses_sum = np.sum([halo.mass * M_sun.value for halo in Subhalo_realization.halos])
        subhalo_mass_percentage = Subhalo_masses_sum/(M_Whole_lens)
        if subhalo_mass_percentage > 0.10:
            raise SkipSimulation_subhalo_frac2

        #Calc Einstein radius
        def mass_to_radius(Mass,redshift_src,redshift_def):
            M_Halo = Mass 
            rad_to_arcsec = 206265

            DL = cosmology.angular_diameter_distance(redshift_def).to(u.m)
            DS = cosmology.angular_diameter_distance(redshift_src).to(u.m)
            DLS = cosmology.angular_diameter_distance_z1z2(redshift_def,redshift_src).to(u.m)

            # Einstein radius
            theta = np.sqrt(4 * G * M_Halo/c**2 * DLS/(DL*DS))

            # Return radius in arcsecods
            radius_arcsec = theta * rad_to_arcsec

            return radius_arcsec.value

        theta_E = mass_to_radius(Host_lens_mass,zsource,zdeflector)

        if (theta_E) > 6.0:
            print("Ignoring simulation: Theta_E > 6.0 arcsec")
            raise SkipSimulation_theta_E2

        #Host halo
        Host_halo = 'EPL'
        Host_kwargs = {'theta_E':theta_E,'gamma': Host_gamma,'e1':0.0,'e2':0.0,'center_x':0.0, 'center_y':0.0} 

        #External shear
        External_shear = 'SHEAR'
        Shear_kwargs = {'gamma1':0.05,'gamma2':0.00,'ra_0': 0, 'dec_0': 0} 

        #Macrolens
        Macro_model_list = [Host_halo,External_shear]
        Macro_kwargs_list = [Host_kwargs,Shear_kwargs]
        Macro_redshift_list = [zlens,zlens]

        #Combine macro-model with substructure 
        lens_model_list = Macro_model_list + Sub_model_list
        lens_kwargs_list = Macro_kwargs_list + Sub_kwargs
        lens_redshift_list = Macro_redshift_list + list(Sub_redshift_array)

    

        return lens_model_list,lens_kwargs_list,lens_redshift_list,cosmology,Macro_model_list, Macro_kwargs_list, Macro_redshift_list, arcsec_opening_angle, M_axion, flucs_shape,flucs_args,M_host,M_Whole_lens, len(Subhalo_realization.halos)
        
    def WDM_constructor(zsource, zlens, M_host, log_mlow,log_mhigh, Host_gamma,LOS_Norm):
        '''This function constructs a lens (host halo, sub halo, LOS halos, and lens galaxy) under the assumption of WDM'''
        
        log_mc = np.random.uniform(5.0,10.0,size=None)
        arcsec_opening_angle = 10

        #First, use Pyhalo to create subhalo realization
        Subhalo_constructor= pyHalo.preset_models.preset_model_from_name('WDM')
        Subhalo_realization = Subhalo_constructor(z_lens=zlens,z_source=zsource,log_mc=log_mc,log_m_host=M_host,log_mlow=log_mlow,log_mhigh=log_mhigh,cone_opening_angle_arcsec=arcsec_opening_angle,LOS_normalization=LOS_Norm) 

        #Extract both lensing quantities from subhalo realization (reshifts, lensing models, lensing kwargs) and the cosmology instance from source to observer
        Sub_model_list, Sub_redshift_array, Sub_kwargs, _ = Subhalo_realization.lensing_quantities()
        cosmology = Subhalo_realization.astropy_instance

        #Find masses of halo system 
        subhalo_Halo_masses = [halo.mass * M_sun.value for halo in Subhalo_realization.halos]
        Host_lens_mass = (10**M_host * M_sun.value) 
        M_Whole_lens = Host_lens_mass + np.sum(subhalo_Halo_masses)

        Subhalo_masses_sum = np.sum([halo.mass * M_sun.value for halo in Subhalo_realization.halos])
        subhalo_mass_percentage = Subhalo_masses_sum/(M_Whole_lens)
        if subhalo_mass_percentage > 0.10:
            raise SkipSimulation_subhalo_frac2

        #Calc Einstein radius
        def mass_to_radius(Mass,redshift_src,redshift_def):
            M_Halo = Mass 
            rad_to_arcsec = 206265

            DL = cosmology.angular_diameter_distance(redshift_def).to(u.m)
            DS = cosmology.angular_diameter_distance(redshift_src).to(u.m)
            DLS = cosmology.angular_diameter_distance_z1z2(redshift_def,redshift_src).to(u.m)

            # Einstein radius
            theta = np.sqrt(4 * G * M_Halo/c**2 * DLS/(DL*DS))

            # Return radius in arcsecods
            radius_arcsec = theta * rad_to_arcsec

            return radius_arcsec.value

        theta_E = mass_to_radius(Host_lens_mass,zsource,zdeflector)
        
        if (theta_E) > 6.0:
            print("Ignoring simulation: Theta_E > 6.0 arcsec")
            raise SkipSimulation_theta_E2
        
        #Host Halo
        Host_halo = 'EPL'
        Host_kwargs = {'theta_E':theta_E,'gamma': Host_gamma,'e1':0.0,'e2':0.0,'center_x':0.0, 'center_y':0.0} 

        #External shear
        External_shear = 'SHEAR'
        Shear_kwargs = {'gamma1':0.05,'gamma2':0.00,'ra_0': 0, 'dec_0': 0} 

        #Macrolens
        Macro_model_list = [Host_halo,External_shear]
        Macro_kwargs_list = [Host_kwargs,Shear_kwargs]
        Macro_redshift_list = [zlens,zlens]

        #Combine macro-model with substructure 
        lens_model_list = Macro_model_list + Sub_model_list
        lens_kwargs_list = Macro_kwargs_list + Sub_kwargs
        lens_redshift_list = Macro_redshift_list + list(Sub_redshift_array)

        
        return lens_model_list,lens_kwargs_list,lens_redshift_list,cosmology, Macro_model_list, Macro_kwargs_list, Macro_redshift_list, arcsec_opening_angle,log_mc,M_host,M_Whole_lens, len(Subhalo_realization.halos)

    def SIDM_constructor(zsource, zlens, M_host, log_mlow,log_mhigh,Host_gamma,LOS_Norm):
        '''This function constructs a lens (host halo, sub halo, LOS halos, and lens galaxy) under the assumption of SIDM'''
        
        mass_ranges_subhalos=[[6.0, 7.0], [7.0, 8.0], [8.0, 9.0], [9.0, 10.0]]
        mass_ranges_field_halos=[[6.0, 7.5], [7.5, 8.5], [8.5, 10.0]]
        
        probabilities_subhalos=[np.random.uniform(0.0,1.0,size = None)]
        for i in range(1,len(mass_ranges_subhalos)):
            collapse_prob = probabilities_subhalos[i-1] - np.random.uniform(0.0,probabilities_subhalos[i-1],None)
            probabilities_subhalos.append(collapse_prob)


        field_collapse_1 = probabilities_subhalos[1] - np.random.uniform(0.0,probabilities_subhalos[1],None)
        probabilities_field_halos=[field_collapse_1]
        for i in range(1,len(mass_ranges_field_halos)):
            upper_limit=np.min([probabilities_field_halos[i-1],probabilities_subhalos[i],probabilities_subhalos[i+1]],None)
            field_collapse = upper_limit - np.random.uniform(0.0,upper_limit,None)
            probabilities_field_halos.append(field_collapse)

        cone_opening_angle_arcsec=10

        #First, use Pyhalo to create subhalo realization
        Subhalo_constructor= pyHalo.preset_models.preset_model_from_name('SIDM_core_collapse')
        Subhalo_realization = Subhalo_constructor(z_lens=zlens,z_source=zsource,mass_ranges_subhalos=mass_ranges_subhalos,mass_ranges_field_halos=mass_ranges_field_halos,probabilities_subhalos=probabilities_subhalos,probabilities_field_halos=probabilities_field_halos,log_m_host=M_host,log_mlow=log_mlow,log_mhigh=log_mhigh,cone_opening_angle_arcsec=cone_opening_angle_arcsec,LOS_normalization=LOS_Norm) 

        #Extract both lensing quantities from subhalo realization (reshifts, lensing models, lensing kwargs) and the cosmology instance from source to observer
        Sub_model_list, Sub_redshift_array, Sub_kwargs, _ = Subhalo_realization.lensing_quantities()
        cosmology = Subhalo_realization.astropy_instance

        #Find masses of halo system 
        subhalo_Halo_masses = [halo.mass * M_sun.value for halo in Subhalo_realization.halos]
        Host_lens_mass = (10**M_host * M_sun.value) 
        M_Whole_lens = Host_lens_mass + np.sum(subhalo_Halo_masses)

        Subhalo_masses_sum = np.sum([halo.mass * M_sun.value for halo in Subhalo_realization.halos])
        subhalo_mass_percentage = Subhalo_masses_sum/(M_Whole_lens)
        if subhalo_mass_percentage > 0.10:
            raise SkipSimulation_subhalo_frac2

        #Calc Einstein radius
        def mass_to_radius(Mass,redshift_src,redshift_def):
            M_Halo = Mass 
            rad_to_arcsec = 206265

            DL = cosmology.angular_diameter_distance(redshift_def).to(u.m)
            DS = cosmology.angular_diameter_distance(redshift_src).to(u.m)
            DLS = cosmology.angular_diameter_distance_z1z2(redshift_def,redshift_src).to(u.m)

            # Einstein radius
            theta = np.sqrt(4 * G * M_Halo/c**2 * DLS/(DL*DS))

            # Return radius in arcsecods
            radius_arcsec = theta * rad_to_arcsec

            return radius_arcsec.value

        theta_E = mass_to_radius(Host_lens_mass,zsource,zdeflector)

        if (theta_E) > 6.0:
            print("Ignoring simulation: Theta_E > 6.0 arcsec")
            raise SkipSimulation_theta_E2


        #Host Halo
        Host_halo = 'EPL'
        Host_kwargs = {'theta_E':theta_E,'gamma': Host_gamma,'e1':0.0,'e2':0.0,'center_x':0.0, 'center_y':0.0} 

        #External shear
        External_shear = 'SHEAR'
        Shear_kwargs = {'gamma1':0.05,'gamma2':0.00,'ra_0': 0.0, 'dec_0': 0.0} 

        #Macrolens
        Macro_model_list = [Host_halo,External_shear]
        Macro_kwargs_list = [Host_kwargs,Shear_kwargs]
        Macro_redshift_list = [zlens,zlens]

        #Combine macro-model with substructure 
        lens_model_list = Macro_model_list + Sub_model_list
        lens_kwargs_list = Macro_kwargs_list + Sub_kwargs
        lens_redshift_list = Macro_redshift_list + list(Sub_redshift_array)


        return lens_model_list,lens_kwargs_list,lens_redshift_list,cosmology, Macro_model_list, Macro_kwargs_list, Macro_redshift_list, mass_ranges_subhalos, mass_ranges_field_halos, probabilities_subhalos, probabilities_field_halos, cone_opening_angle_arcsec,M_host,M_Whole_lens, len(Subhalo_realization.halos)

    slope_Host = Host_slope()
    M_host = 13.3

    log_mlow = 6
    log_mhigh = 10
    
    print(f'Redshifts being used are {redshifts}')
    attempt = 1
    max_attempts = 10
    good = False
    while good == False and attempt <= max_attempts:
        try:
            print(f'Attempt:{attempt}')
            zdeflector = redshifts[0]
            zsource = redshifts[1]
            print(f'z_source:{zsource}')
            if DM_Type == 'CDM':
                lens_model_list,lens_kwargs_list,lens_redshift_list,cosmology,Macro_model_list, Macro_kwargs_list, Macro_redshift_list,arcsecond_opening_angle,host_mass,lens_mass, num_subhalos=CDM_constructor(zsource=zsource,zlens=zdeflector,M_host=M_host,log_mlow=log_mlow,log_mhigh=log_mhigh,Host_gamma=slope_Host,LOS_Norm=LOS)
            elif DM_Type == 'WDM':
                lens_model_list,lens_kwargs_list,lens_redshift_list,cosmology,Macro_model_list, Macro_kwargs_list, Macro_redshift_list, log_mc, arcsecond_opening_angle,host_mass,lens_mass, num_subhalos =WDM_constructor(zsource=zsource,zlens=zdeflector,M_host=M_host,log_mlow=log_mlow,log_mhigh=log_mhigh,Host_gamma=slope_Host,LOS_Norm=LOS)
            elif DM_Type == 'SIDM':
                lens_model_list,lens_kwargs_list,lens_redshift_list,cosmology,Macro_model_list, Macro_kwargs_list, Macro_redshift_list, mass_ranges_subhalos, mass_ranges_field_halos, probabilities_subhalos, probabilities_field_halos, arcsecond_opening_angle,host_mass,lens_mass, num_subhalos=SIDM_constructor(zsource=zsource,zlens=zdeflector,M_host=M_host,log_mlow=log_mlow,log_mhigh=log_mhigh,Host_gamma=slope_Host,LOS_Norm=LOS)
            elif DM_Type == 'Axion':
                lens_model_list,lens_kwargs_list,lens_redshift_list,cosmology,Macro_model_list, Macro_kwargs_list, Macro_redshift_list,arcsecond_opening_angle,M_axion,flucs_shape,flucs_args,host_mass,lens_mass, num_subhalos=Axion_constructor(zsource=zsource,zlens=zdeflector,M_host=M_host,log_mlow=log_mlow,log_mhigh=log_mhigh,Host_gamma=slope_Host,LOS_Norm=LOS)
            print('Success!')
            good = True
        except SkipSimulation_theta_E2 as e:
            raise SkipSimulation_theta_E1
        except SkipSimulation_subhalo_frac2 as e:
            raise SkipSimulation_subhalo_frac1       
        except Exception as e:
            log(f'Failed on attempt {attempt}: {e}')
            print(e)
            redshifts = np.array([redshifts[0],redshifts[1] + np.random.uniform(0.25,0.5)])
            print(f'Falied! Trying again with zsrc = {redshifts[1]}')
            attempt += 1
            good = False

    if not good:
        raise RuntimeError("All attempts failed to generate a valid model.")


    end3 = time.time()
    print(f'Step 3 took {end3-start3} secs')
    #4. Consolidate lens and source models and finalize kwargs
    start4 = time.time()
    kwargs_model = {'lens_model_list': lens_model_list,  # list of lens models to be used
                'lens_redshift_list': lens_redshift_list,
                'lens_light_model_list': ['INTERPOL'],  # list of unlensed light models to be used
                'source_light_model_list': ['INTERPOL'],  # list of extended source models to be used, here we used the interpolated real galaxy
                'z_source':zsource,
                'cosmo': cosmology}
    

    def mass_to_radius(Mass,redshift_src,redshift_def):
            M_Halo = 10**Mass * M_sun
            rad_to_arcsec = 206265

            DL = cosmology.luminosity_distance(redshift_def).to(u.m)
            DS = cosmology.luminosity_distance(redshift_src).to(u.m)
            DLS = DS - DL

            # Einstein radius
            theta = np.sqrt(4 * G * M_Halo/c**2 * DLS/(DL*DS))

            # Return radius in arcsecods
            radius_arcsec = theta * rad_to_arcsec

            return radius_arcsec.value

    theta_E_nss = mass_to_radius(lens_mass,zsource,zdeflector)

    Host_kwargs_nss = {'theta_E':theta_E_nss,'gamma': Macro_kwargs_list[0]['gamma'],'e1':0.0,'e2':0.0,'center_x':0.0, 'center_y':0.0} 

    Macro_kwargs_list_nss = [Host_kwargs_nss,Macro_kwargs_list[1]]

    kwargs_model_nss = {'lens_model_list': Macro_model_list,  # list of lens models to be used
                'lens_redshift_list': Macro_redshift_list,
                'lens_light_model_list': ['INTERPOL'],  # list of unlensed light models to be used
                'source_light_model_list': ['INTERPOL'],  # list of extended source models to be used, here we used the interpolated real galaxy
                'z_source':zsource,
                'cosmo': cosmology}

    
    source_pos_xx,source_pos_yy = np.random.uniform(-Macro_kwargs_list[0]['theta_E']*0.3, Macro_kwargs_list[0]['theta_E']*0.3,None), np.random.uniform(-Macro_kwargs_list[0]['theta_E']*0.3, Macro_kwargs_list[0]['theta_E']*0.3,None)
    
    # Source coordinates are in arcseconds from center of image

    #Note: image and pixel scale you insert are for the origin telescope
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
    hsc_pix_scale = 0.168
    kwargs_lens_light_mag, kwargs_source_mag = kwargs_light_mag(source_pos_xx,source_pos_yy,hsc_pix_scale)

    end4=time.time()
    print(f'Step 4 took {end4-start4} secs')
    #5. Simulate Image with no substructure
    start5 = time.time()
    from lenstronomy.SimulationAPI.sim_api import SimAPI
    import lenstronomy.Plots.plot_util as plot_util

    kwargs_numerics = {'point_source_supersampling_factor': 1}
    
    def simulate(kwargs_numerics,band_kwargs,lens_light_kwargs,source_light_kwargs,lens_nonlight_kwargs,kwargs_model_):
        numpix = 128

        sim_g = SimAPI(numpix=numpix, kwargs_single_band=band_kwargs[0], kwargs_model=kwargs_model_)
        sim_r = SimAPI(numpix=numpix, kwargs_single_band=band_kwargs[1], kwargs_model=kwargs_model_)
        sim_i = SimAPI(numpix=numpix, kwargs_single_band=band_kwargs[2], kwargs_model=kwargs_model_)

        imSim_g = sim_g.image_model_class(kwargs_numerics)
        imSim_r = sim_r.image_model_class(kwargs_numerics)
        imSim_i = sim_i.image_model_class(kwargs_numerics)

        kwargs_lens_light_g, kwargs_source_g,_ = sim_g.magnitude2amplitude(lens_light_kwargs[0], source_light_kwargs[0])
        kwargs_lens_light_r, kwargs_source_r,_ = sim_r.magnitude2amplitude(lens_light_kwargs[1], source_light_kwargs[1])
        kwargs_lens_light_i, kwargs_source_i,_ = sim_i.magnitude2amplitude(lens_light_kwargs[2], source_light_kwargs[2])

        image_g_flux = imSim_g.image(lens_nonlight_kwargs, kwargs_source_g, kwargs_lens_light_g,point_source_add=False,source_add=True,lens_light_add=lens_light) 
        image_r_flux = imSim_r.image(lens_nonlight_kwargs, kwargs_source_r, kwargs_lens_light_r,point_source_add=False,source_add=True,lens_light_add=lens_light) 
        image_i_flux = imSim_i.image(lens_nonlight_kwargs, kwargs_source_i, kwargs_lens_light_i,point_source_add=False,source_add=True,lens_light_add=lens_light) 
       
        image_g = image_g_flux * band_kwargs[0]['exposure_time'] * band_kwargs[0]['num_exposures'] / et_quotient 
        image_r = image_r_flux * band_kwargs[1]['exposure_time'] * band_kwargs[1]['num_exposures'] /et_quotient
        image_i = image_i_flux * band_kwargs[2]['exposure_time'] * band_kwargs[2]['num_exposures'] /et_quotient
        #Each output pixel in units of e counts 

        image_g_noise = image_g + sim_g.noise_for_model(model=image_g)
        image_r_noise = image_r + sim_r.noise_for_model(model=image_r)
        image_i_noise = image_i + sim_i.noise_for_model(model=image_i)

        total_exposure_times = np.array([band_kwargs[0]['exposure_time'] * band_kwargs[0]['num_exposures'] / et_quotient,band_kwargs[1]['exposure_time'] * band_kwargs[1]['num_exposures'] /et_quotient,band_kwargs[2]['exposure_time'] * band_kwargs[2]['num_exposures'] /et_quotient])

        return image_g_noise,image_r_noise,image_i_noise,image_g,image_r,image_i,total_exposure_times
    
    img_g_noise,img_r_noise,img_i_noise,img_g,img_r,img_i,tot_exp_times = simulate(kwargs_numerics=kwargs_numerics,band_kwargs=bands,lens_light_kwargs=kwargs_lens_light_mag,source_light_kwargs=kwargs_source_mag,lens_nonlight_kwargs=lens_kwargs_list,kwargs_model_=kwargs_model)
    
    SNR_brightest = np.array([np.sqrt(np.max(img_g_noise)),np.sqrt(np.max(img_r_noise)),np.sqrt(np.max(img_i_noise))]) #SNR of brightest pixel in each band 

    if np.max(SNR_brightest) > 100:
        raise SkipSimulation_snr1


    img_nss_g_noise,img_nss_r_noise,img_nss_i_noise,_,_,_,_ = simulate(kwargs_numerics=kwargs_numerics,band_kwargs=bands,lens_light_kwargs=kwargs_lens_light_mag,source_light_kwargs=kwargs_source_mag,lens_nonlight_kwargs=Macro_kwargs_list_nss,kwargs_model_=kwargs_model_nss)

    sns_diff_g = img_g_noise / img_nss_g_noise
    sns_diff_r = img_r_noise / img_nss_r_noise
    sns_diff_i = img_i_noise / img_nss_i_noise

    img = (img_g_noise,img_r_noise,img_i_noise)
    img_nss = (img_nss_g_noise,img_nss_r_noise,img_nss_i_noise)
    img_NOise = (img_g,img_r,img_i)
    sns_diff = (sns_diff_g,sns_diff_r,sns_diff_i)

    end5 = time.time()
    print(f'Step 5 took {end5-start5} secs')

    
    #Prepare Outputs
    DL = cosmology.angular_diameter_distance(zdeflector)/1000
    hf.create_dataset(f'images/strong_lens_{i}/d_l', data=np.array([str(DL),'Angular diameter distance to deflector galaxy in Gpc'],dtype=dt),dtype=dt)
    DS = cosmology.angular_diameter_distance(zsource)/1000
    hf.create_dataset(f'images/strong_lens_{i}/d_s', data=np.array([str(DS),'Angular diameter distance to source galaxy in Gpc'],dtype=dt),dtype=dt)
    DLS = cosmology.angular_diameter_distance_z1z2(zdeflector,zsource)/1000
    hf.create_dataset(f'images/strong_lens_{i}/d_ls', data=np.array([str(DLS),'Angular diameter distance between deflector and source galaxy in Gpc'],dtype=dt),dtype=dt)
    hf.create_dataset(f'images/strong_lens_{i}/exposure_time', data=np.array([str(tot_exp_times[0]),str(tot_exp_times[1]),str(tot_exp_times[2]), 'Exposure time in seconds for all bands'],dtype=dt),dtype=dt)
    hf.create_dataset(f'images/strong_lens_{i}/instrument', data=np.array([Instrument, 'Instrument'],dtype=dt),dtype=dt)
    hf.create_dataset(f'images/strong_lens_{i}/log_mhigh', data=np.array([str(log_mhigh),'Log10 of largest possible subhalo mass'],dtype=dt),dtype=dt)
    hf.create_dataset(f'images/strong_lens_{i}/log_mlow', data=np.array([str(log_mlow),'Log10 of lowest possible subhalo mass'],dtype=dt),dtype=dt)
    hf.create_dataset(f'images/strong_lens_{i}/host_mass', data=np.array([str(host_mass),'Log10 of mass of host lens in units of M_sun'],dtype=dt),dtype=dt)
    hf.create_dataset(f'images/strong_lens_{i}/lens_mass', data=np.array([str(lens_mass),'Log10 of mass of entire lens in units of M_sun'],dtype=dt),dtype=dt)
    hf.create_dataset(f'images/strong_lens_{i}/num_subhalos', data=np.array([str(num_subhalos), 'Number of subhalos'],dtype=dt),dtype=dt)
    hf.create_dataset(f'images/strong_lens_{i}/r_tidal', data=np.array([str(0.25), 'see Pyhalo documentation'],dtype=dt),dtype=dt)
    hf.create_dataset(f'images/strong_lens_{i}/sigma_sub', data=np.array([str(0.025), 'see Pyhalo documentation'],dtype=dt),dtype=dt)
    hf.create_dataset(f'images/strong_lens_{i}/snr', data=np.array([str(SNR_brightest), 'SNR of brightest pixel in all bands'],dtype=dt),dtype=dt)
    hf.create_dataset(f'images/strong_lens_{i}/theta_e', data=np.array([str(Macro_kwargs_list[0]['theta_E']), 'Einstein radius in arcseconds'],dtype=dt),dtype=dt)
    hf.create_dataset(f'images/strong_lens_{i}/uid',data=np.array([str(i),f'simulation number in {DM_Type} batch created on {timestamp}'],dtype=dt),dtype=dt)
    hf.create_dataset(f'images/strong_lens_{i}/z_lens', data=np.array([str(zdeflector),'Redshift of deflector'],dtype=dt),dtype=dt)
    hf.create_dataset(f'images/strong_lens_{i}/z_source', data=np.array([str(zsource),'Redshift of source'],dtype=dt),dtype=dt)
    hf.create_dataset(f'images/strong_lens_{i}/bands', data=np.array([band_labels[0],band_labels[1],band_labels[2], 'Instrument bands in which image was simulated'],dtype=dt),dtype=dt)
    hf.create_dataset(f'images/strong_lens_{i}/DM_type',data = np.array([DM_Type, 'Type of dark matter assumed'],dtype=dt),dtype=dt)
    hf.create_dataset(f'images/strong_lens_{i}/source_pos',data = np.array([str([source_pos_xx,source_pos_yy]), 'Plane coordinates of source wrt center of deflector (in arcseconds)'],dtype=dt),dtype=dt)
    hf.create_dataset(f'images/strong_lens_{i}/host_slope',data = np.array([str(Host_slope), 'EPL slope of host halo'],dtype=dt),dtype=dt)
    hf.create_dataset(f'images/strong_lens_{i}/ellipticity',data = np.array([str([Macro_kwargs_list[0]['e1'],Macro_kwargs_list[0]['e2']]), 'Ellipticity values for host halo mass profile'],dtype=dt),dtype=dt)


    if DM_Type == 'WDM':
        hf.create_dataset(f'images/strong_lens_{i}/log_mc',data = np.array([log_mc, 'Mass at which underneath the WDM mass function is surpressed when compared to CDM'],dtype=dt),dtype=dt)
    elif DM_Type == 'Axion':
        hf.create_dataset(f'images/strong_lens_{i}/m_axion',data = np.array([str(M_axion), 'log10 of mass of axion particle'],dtype=dt),dtype=dt)
        hf.create_dataset(f'images/strong_lens_{i}/flucs_shape',data = np.array([flucs_shape, 'Shape at which to render fluctuations of the host halo (See Pyhalo)'],dtype=dt),dtype=dt)
        hf.create_dataset(f'images/strong_lens_{i}/flucs_args',data = np.array([str(flucs_args[0]),str(flucs_args[1]),str(flucs_args[2]), 'Arugments corresponding to geometry of host halo fluctuations (See Pyhalo)'],dtype=dt),dtype=dt)
    elif DM_Type == 'SIDM':
        hf.create_dataset(f'images/strong_lens_{i}/subhalo_mass_ranges',data = np.array([str(mass_ranges_subhalos), 'Mass ranges at which subhalos are sampled from'],dtype=dt),dtype=dt)
        hf.create_dataset(f'images/strong_lens_{i}/field_halo_mass_ranges',data = np.array([str(mass_ranges_field_halos), 'Mass ranges at which field halos are sampled from'],dtype=dt),dtype=dt)
        hf.create_dataset(f'images/strong_lens_{i}/prob_subhalo',data = np.array([str(probabilities_subhalos), 'Probability of subhalo being sampled from corresponding mass range'],dtype=dt),dtype=dt)
        hf.create_dataset(f'images/strong_lens_{i}/prob_field_halo',data = np.array([str(probabilities_field_halos), 'Probability of field halo being sampled from corresponding mass range'],dtype=dt),dtype=dt)



    exposure_0=hf.create_dataset(f'images/strong_lens_{i}/exposure_{i}_{band_labels[0]}',data = img[0])
    exposure_0.attrs['filter'] = np.array([band_labels[0],'Filter'],dtype=dt)
    exposure_0.attrs['fov'] = np.array([str(arcsecond_opening_angle),'Field of view [arcsec]'],dtype=dt)
    exposure_0.attrs['pixel_scale'] = np.array([str(bands[0]['pixel_scale']),'Pixel scale [arcsec/pixel]'],dtype=dt)
    exposure_0.attrs['lens_magnitude'] = np.array([str(deflector_mag[0]),'Lens magnitude'],dtype=dt)
    exposure_0.attrs['source_magnitude'] = np.array([str(source_mag[0]),'Unlensed source galaxy magnitude'],dtype=dt)
    exposure_0.attrs['units'] = np.array(['counts','Units of pixel values'],dtype=dt)
    hf.create_dataset(f'images/strong_lens_{i}/exposure_{i}_{band_labels[0]}_nss',data = img_nss[0])

    
    exposure_1=hf.create_dataset(f'images/strong_lens_{i}/exposure_{i}_{band_labels[1]}',data = img[1])
    exposure_1.attrs['filter'] = np.array([band_labels[1],'Filter'],dtype=dt)
    exposure_1.attrs['fov'] = np.array([str(arcsecond_opening_angle),'Field of view [arcsec]'],dtype=dt)
    exposure_1.attrs['pixel_scale'] = np.array([str(bands[1]['pixel_scale']),'Pixel scale [arcsec/pixel]'],dtype=dt)
    exposure_1.attrs['lens_magnitude'] = np.array([str(deflector_mag[1]),'Lens magnitude'],dtype=dt)
    exposure_1.attrs['source_magnitude'] = np.array([str(source_mag[1]),'Unlensed source galaxy magnitude'],dtype=dt)
    exposure_1.attrs['units'] = np.array(['counts','Units of pixel values'],dtype=dt)
    hf.create_dataset(f'images/strong_lens_{i}/exposure_{i}_{band_labels[1]}_nss',data = img_nss[1])
    
    exposure_2=hf.create_dataset(f'images/strong_lens_{i}/exposure_{i}_{band_labels[2]}',data = img[2])
    exposure_2.attrs['filter'] = np.array([band_labels[2],'Filter'],dtype=dt)
    exposure_2.attrs['fov'] = np.array([str(arcsecond_opening_angle),'Field of view [arcsec]'],dtype=dt)
    exposure_2.attrs['pixel_scale'] = np.array([str(bands[2]['pixel_scale']),'Pixel scale [arcsec/pixel]'],dtype=dt)
    exposure_2.attrs['lens_magnitude'] = np.array([str(deflector_mag[2]),'Lens magnitude'],dtype=dt)
    exposure_2.attrs['source_magnitude'] = np.array([str(source_mag[2]),'Unlensed source galaxy magnitude'],dtype=dt)
    exposure_2.attrs['units'] = np.array(['counts','Units of pixel values'],dtype=dt)
    hf.create_dataset(f'images/strong_lens_{i}/exposure_{i}_{band_labels[2]}_nss',data = img_nss[2])
    

    end = time.time()
    print(f'Total simulation took {end-start_time} secs')
   



#%%
##########################################
#LOGISTICAL STUFF
# Define constants
instruments = ['Roman']
DM_types = ['CDM', 'Axion']
total_sim_num = 1000
timestamp = datetime.datetime.now().strftime("[%Y-%m-%d]")
log_file = f"sim_log(No_lens_light_{timestamp}).txt"
if not os.path.exists(f'model_alpha_{timestamp}'):
    os.mkdir(f'model_alpha_{timestamp}')


# Ensure log file exists
if not os.path.exists(log_file):
    with open(log_file, "w") as f:
        f.write("Simulation Log\n")

def log(message):
    timestamp_specific = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    with open(log_file, "a") as f:
        f.write(f"{timestamp} {message}\n")
    print(f"{timestamp_specific} {message}")


# Read completed simulations
completed_sims = set()
with open(log_file, "r") as f:
    for line in f:
        if "COMPLETED" in line:
            completed_sims.add(line.split(" ")[-1].strip())


#Set up simulation loop
def run_simulation(args):
    DM_type, instrument, i = args
    sim_id = f"{DM_type}_{instrument}_{i}"
    if sim_id in completed_sims:
        return #BECAUSE OF THIS LINE YOU MUST RESET THE TEXT FILE WHENEVER YOU WANT TO REDO SIMS!!!!

    while True:
        try:
            bands, source_images, source_mag, deflector_images, deflector_mag, redshifts, raw_src, raw_dfr,start_t,band_labels = dlu_sim_1(instrument)
            log(f"STARTED simulation {sim_id} with z_scr={redshifts[1]}, z_dfr={redshifts[0]}")

            dlu_sim_2(DM_type, instrument, 10**(0.9), bands, source_images, source_mag,
                                deflector_images, deflector_mag, redshifts, raw_src, raw_dfr,start_t,band_labels)

            log(f"COMPLETED simulation {sim_id}")
            break

        except SkipSimulation_theta_E1:
            log("Trying a new simulation due to too large theta_E")
            continue
        except SkipSimulation_subhalo_frac1:
            log("Trying a new simulation due to unphysical subhalo mass fraction")
            continue
        except SkipSimulation_snr1:
            log("Trying a new simulation due to too high snr")
            continue
        except Exception as e:
            log(f"REDSHIFT z_scr={redshifts[1]}, z_dfr={redshifts[0]}, Δz={redshifts[1] - redshifts[0]}")
            log(f"ERROR in simulation {sim_id}: {e}")
            log(traceback.format_exc())
            log("Trying a new simulation due to error")
            continue


for type in DM_types:
    for instrument in instruments:
        output_file_name = f"model_alpha_{type}_{instrument}_{timestamp}.h5"
        output_path = f'./model_alpha_{timestamp}/{output_file_name}'
        dt = h5py.string_dtype(encoding='utf-8')
        if os.path.exists(output_path):
            with h5py.File(f'./model_alpha_{timestamp}/{output_file_name}','r+') as hf:
                for i in range(int(total_sim_num/(len(DM_types)*len(instruments)))):
                    if f'images/strong_lens_{i}' not in hf:
                        hf.create_group(f'images/strong_lens_{i}')
                        run_simulation([type,instrument,i])
        else:
            with h5py.File(f'./model_alpha_{timestamp}/{output_file_name}','w') as hf:
                hf.create_group('images')
                for i in range(int(total_sim_num/(len(DM_types)*len(instruments)))):
                    hf.create_group(f'images/strong_lens_{i}')
                    run_simulation([type,instrument,i])









