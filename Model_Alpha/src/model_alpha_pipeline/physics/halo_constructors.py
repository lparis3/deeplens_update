import numpy as np
import pyHalo

from model_alpha_pipeline.structures.dataclasses import LensSystem


def cdm_constructor(
    z_source,
    z_lens,
    m_host,
    host_theta_e_arcsec,
    max_subhalo_mass,
    host_gamma,
    los_norm,
):
    """
    Construct a lens realization under CDM assumptions.
    """
    arcsec_opening_angle = 10
    log_mlow = 6
    log_mhigh = np.log10(max_subhalo_mass)
    log_mhost = np.log10(m_host)

    subhalo_constructor = pyHalo.preset_models.preset_model_from_name("CDM")
    subhalo_realization = subhalo_constructor(
        z_lens=z_lens,
        z_source=z_source,
        log_m_host=log_mhost,
        log_mlow=log_mlow,
        log_mhigh=log_mhigh,
        cone_opening_angle_arcsec=arcsec_opening_angle,
        LOS_normalization=los_norm,
    )

    sub_model_list, sub_redshift_array, sub_kwargs, _ = subhalo_realization.lensing_quantities()
    cosmology = subhalo_realization.astropy_instance

    host_halo = "EPL"
    host_kwargs = {
        "theta_E": host_theta_e_arcsec,
        "gamma": host_gamma,
        "e1": 0.0,
        "e2": 0.0,
        "center_x": 0.0,
        "center_y": 0.0,
    }

    external_shear = "SHEAR"
    shear_kwargs = {"gamma1": 0.05, "gamma2": 0.05}

    macro_model_list = [host_halo, external_shear]
    macro_kwargs_list = [host_kwargs, shear_kwargs]
    macro_redshift_list = [z_lens, z_lens]

    lens_model_list = macro_model_list + sub_model_list
    lens_kwargs_list = macro_kwargs_list + sub_kwargs
    lens_redshift_list = macro_redshift_list + list(sub_redshift_array)

    return (
        lens_model_list,
        lens_kwargs_list,
        lens_redshift_list,
        cosmology,
        macro_model_list,
        macro_kwargs_list,
        macro_redshift_list,
        arcsec_opening_angle,
        log_mlow,
        m_host,
        m_host,
        len(subhalo_realization.halos),
    )


def axion_constructor(zsource, zlens, M_host, host_theta_E_arcsec,max_subhalo_mass,Host_gamma, LOS_Norm):
        '''This function constructs a lens (host halo, sub halo, LOS halos, and lens galaxy) under the assumption of axionic dark matter'''
        arcsec_opening_angle = 10
        log_mlow = 6 #in units of solar masses
        log_mhigh = np.log10(max_subhalo_mass) #in units of solar masses
        log_Mhost = np.log10(M_host)
        
        #Set axion parameters
        M_axion = np.random.uniform(-22.0,-19.0,None)
        flucs_shape='ring'
        arcsec_opening_angle = 10
        debrogile_wavelength_order = 0.6 * (10**(-22)/10**M_axion)
        rmax = 1.1 * debrogile_wavelength_order 
        rmin = 0.9 * debrogile_wavelength_order 
        flucs_args={'angle': np.random.uniform(0.0,2*np.pi,None), 'rmin': rmin, 'rmax': rmax}


        #First, use Pyhalo to create subhalo realization
        Subhalo_constructor= pyHalo.preset_models.preset_model_from_name('ULDM')
        Subhalo_realization = Subhalo_constructor(z_lens=zlens,z_source=zsource,log10_m_uldm = M_axion ,flucs_shape=flucs_shape,flucs_args=flucs_args,log_m_host=log_Mhost,log_mlow=log_mlow,log_mhigh=log_mhigh,cone_opening_angle_arcsec=arcsec_opening_angle, LOS_normalization=LOS_Norm) 
      
        
        #Extract both lensing quantities from subhalo realization (reshifts, lensing models, lensing kwargs) and the cosmology instance from source to observer
        Sub_model_list, Sub_redshift_array, Sub_kwargs, _ = Subhalo_realization.lensing_quantities()
        cosmology = Subhalo_realization.astropy_instance
        
    
        #Host halo
        Host_halo = 'EPL'
        Host_kwargs = {'theta_E':host_theta_E_arcsec,'gamma': Host_gamma,'e1':0.0,'e2':0.0,'center_x':0.0, 'center_y':0.0} 

        #External shear
        External_shear = 'SHEAR'
        Shear_kwargs = {'gamma1':0.05,'gamma2':0.00,'ra_0': 0, 'dec_0': 0} 

        #Macrolens
        Macro_model_list = [Host_halo,External_shear]
        macro_kwargs_list = [Host_kwargs,Shear_kwargs]
        Macro_redshift_list = [zlens,zlens]

        #Combine macro-model with substructure 
        lens_model_list = Macro_model_list + Sub_model_list
        lens_kwargs_list = macro_kwargs_list + Sub_kwargs
        lens_redshift_list = Macro_redshift_list + list(Sub_redshift_array)

        #Find masses of halo system 
        subhalo_Halo_masses = [halo.mass for halo in Subhalo_realization.halos]
        Host_lens_mass = M_host
        M_Whole_lens = Host_lens_mass + np.sum(subhalo_Halo_masses) #in units of solar masses


        return lens_model_list,lens_kwargs_list,lens_redshift_list,cosmology,Macro_model_list, macro_kwargs_list, Macro_redshift_list, arcsec_opening_angle, M_axion,flucs_shape,flucs_args,M_host,M_Whole_lens,len(Subhalo_realization.halos)
        
def wdm_constructor(zsource, zlens,M_host,host_theta_E_arcsec,max_subhalo_mass, Host_gamma,LOS_Norm):
        '''This function constructs a lens (host halo, sub halo, LOS halos, and lens galaxy) under the assumption of WDM'''
        arcsec_opening_angle = 10
        log_mlow = 6 #in units of solar masses
        log_mhigh = np.log10(max_subhalo_mass) #in units of solar masses
        log_Mhost = np.log10(M_host)
        
        #Set WDM parameters
        log_mc = np.random.uniform(5.0,10.0,size=None)
        arcsec_opening_angle = 10
      
        #First, use Pyhalo to create subhalo realization
        Subhalo_constructor= pyHalo.preset_models.preset_model_from_name('WDM')
        Subhalo_realization = Subhalo_constructor(z_lens=zlens,z_source=zsource,log_mc=log_mc,log_m_host=log_Mhost,log_mlow=log_mlow,log_mhigh=log_mhigh,cone_opening_angle_arcsec=arcsec_opening_angle,LOS_normalization=LOS_Norm) 
      
        #Extract both lensing quantities from subhalo realization (reshifts, lensing models, lensing kwargs) and the cosmology instance from source to observer
        Sub_model_list, Sub_redshift_array, Sub_kwargs, _ = Subhalo_realization.lensing_quantities()
        cosmology = Subhalo_realization.astropy_instance
      
      
        #Host Halo
        Host_halo = 'EPL'
        Host_kwargs = {'theta_E':host_theta_E_arcsec,'gamma': Host_gamma,'e1':0.0,'e2':0.0,'center_x':0.0, 'center_y':0.0} 
      
        #External shear
        External_shear = 'SHEAR'
        Shear_kwargs = {'gamma1':0.05,'gamma2':0.00,'ra_0': 0, 'dec_0': 0} 
      
        #Macrolens
        Macro_model_list = [Host_halo,External_shear]
        macro_kwargs_list = [Host_kwargs,Shear_kwargs]
        Macro_redshift_list = [zlens,zlens]
      
        #Combine macro-model with substructure 
        lens_model_list = Macro_model_list + Sub_model_list
        lens_kwargs_list = macro_kwargs_list + Sub_kwargs
        lens_redshift_list = Macro_redshift_list + list(Sub_redshift_array)
      
        #Find masses of halo system 
        subhalo_Halo_masses = [halo.mass for halo in Subhalo_realization.halos]
        Host_lens_mass = M_host
        M_Whole_lens = Host_lens_mass + np.sum(subhalo_Halo_masses) #in units of solar masses
      
        return lens_model_list,lens_kwargs_list,lens_redshift_list,cosmology, Macro_model_list, macro_kwargs_list, Macro_redshift_list, arcsec_opening_angle,log_mc,M_host,M_Whole_lens, len(Subhalo_realization.halos)

def sidm_constructor(zsource, zlens,M_host,host_theta_E_arcsec,max_subhalo_mass,Host_gamma,LOS_Norm):
        '''This function constructs a lens (host halo, sub halo, LOS halos, and lens galaxy) under the assumption of SIDM'''
        
        arcsec_opening_angle = 10
        log_mlow = 6 #in units of solar masses
        log_mhigh = np.log10(max_subhalo_mass) #in units of solar masses
        log_Mhost = np.log10(M_host)
        
        #Set SIDM params
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

        #First, use Pyhalo to create subhalo realization
        Subhalo_constructor= pyHalo.preset_models.preset_model_from_name('SIDM_core_collapse')
        Subhalo_realization = Subhalo_constructor(z_lens=zlens,z_source=zsource,mass_ranges_subhalos=mass_ranges_subhalos,mass_ranges_field_halos=mass_ranges_field_halos,probabilities_subhalos=probabilities_subhalos,probabilities_field_halos=probabilities_field_halos,log_m_host=log_Mhost,log_mlow=log_mlow,log_mhigh=log_mhigh,cone_opening_angle_arcsec=arcsec_opening_angle,LOS_normalization=LOS_Norm) 

        #Extract both lensing quantities from subhalo realization (reshifts, lensing models, lensing kwargs) and the cosmology instance from source to observer
        Sub_model_list, Sub_redshift_array, Sub_kwargs, _ = Subhalo_realization.lensing_quantities()
        cosmology = Subhalo_realization.astropy_instance

        #Host Halo
        Host_halo = 'EPL'
        Host_kwargs = {'theta_E':host_theta_E_arcsec,'gamma': Host_gamma,'e1':0.0,'e2':0.0,'center_x':0.0, 'center_y':0.0} 

        #External shear
        External_shear = 'SHEAR'
        Shear_kwargs = {'gamma1':0.05,'gamma2':0.00,'ra_0': 0.0, 'dec_0': 0.0} 

        #Macrolens
        Macro_model_list = [Host_halo,External_shear]
        macro_kwargs_list = [Host_kwargs,Shear_kwargs]
        Macro_redshift_list = [zlens,zlens]

        #Combine macro-model with substructure 
        lens_model_list = Macro_model_list + Sub_model_list
        lens_kwargs_list = macro_kwargs_list + Sub_kwargs
        lens_redshift_list = Macro_redshift_list + list(Sub_redshift_array)

        #Find masses of halo system 
        subhalo_Halo_masses = [halo.mass for halo in Subhalo_realization.halos]
        Host_lens_mass = M_host
        M_Whole_lens = Host_lens_mass + np.sum(subhalo_Halo_masses) #in units of solar masses

        return lens_model_list,lens_kwargs_list,lens_redshift_list,cosmology, Macro_model_list, macro_kwargs_list, Macro_redshift_list, mass_ranges_subhalos, mass_ranges_field_halos, probabilities_subhalos, probabilities_field_halos, arcsec_opening_angle,M_host,M_Whole_lens, len(Subhalo_realization.halos)
