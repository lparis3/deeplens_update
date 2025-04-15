#Construct host halo, sub halos, and field halos 
import pyHalo
import pyHalo.preset_models
import numpy as np
from astropy import units as u
from astropy.constants import G, c, M_sun

def Halo_constructor(DM_type, redshifts, Interlopers):
  if Interlopers == True:
    LOS = 1.0
  else:
    LOS = 0.0

  def Host_mass(mean = 13, sigma =1.0):
      '''Samples log of host mass from normal distribution'''
      M_host = np.random.normal(loc = mean,scale = sigma,size =1)
      return M_host[0]

  def Host_slope(gammaL = 1.9, gammaH = 2.2):
      '''Samples host halos log slope between lower and upper bounds from Gilman et al. 2022'''
      return np.random.uniform(gammaL,gammaH,None)

  def CDM_constructor(zsource, zlens, M_host, Host_gamma, LOS_Norm):
      '''This function constructs a lens (host halo, sub halo, LOS halos, and lens galaxy) under the assumption of CDM'''
      arcsec_opening_angle = 10

      #First, use Pyhalo to create subhalo realization
      Subhalo_constructor= pyHalo.preset_models.preset_model_from_name('CDM')
      Subhalo_realization = Subhalo_constructor(z_lens=zlens,z_source=zsource,log_m_host=M_host,cone_opening_angle_arcsec=arcsec_opening_angle, LOS_normalization=LOS_Norm) 
      print(len(Subhalo_realization.halos))

      #Extract both lensing quantities from subhalo realization (reshifts, lensing models, lensing kwargs) and the cosmology instance from source to observer
      Sub_model_list, Sub_redshift_array, Sub_kwargs, _ = Subhalo_realization.lensing_quantities()
      cosmology = Subhalo_realization.astropy_instance

      #Calc Einstein radius
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

      theta_E = mass_to_radius(M_host,zsource,zdeflector)


      #Host halo
      Host_halo = 'EPL'
      Host_kwargs = {'theta_E':theta_E,'gamma':Host_gamma,'e1':0.4,'e2':-0.1,'center_x':0.0, 'center_y':0.0} 

      #External shear
      External_shear = 'SHEAR'
      Shear_kwargs = {'gamma1':0.03,'gamma2':0.01,'ra_0': 0.0, 'dec_0': 0.0} 

      #Macrolens
      Macro_model_list = [Host_halo,External_shear]
      Macro_kwargs_list = [Host_kwargs,Shear_kwargs]
      Macro_redshift_list = [zlens,zlens]

      #Combine macro-model with substructure 
      lens_model_list = Macro_model_list + Sub_model_list
      lens_kwargs_list = Macro_kwargs_list + Sub_kwargs
      lens_redshift_list = Macro_redshift_list + list(Sub_redshift_array)

      return lens_model_list,lens_kwargs_list,lens_redshift_list,cosmology,Macro_model_list, Macro_kwargs_list, Macro_redshift_list, arcsec_opening_angle

  def Axion_constructor(zsource, zlens, M_host, Host_gamma, LOS_Norm):
      '''This function constructs a lens (host halo, sub halo, LOS halos, and lens galaxy) under the assumption of axionic dark matter'''
      
      M_axion = -22
      #np.random.uniform(-22.0,-19.0,None)
      flucs_shape='ring'
      arcsec_opening_angle = 10
      debrogile_wavelength_order = 0.6 * (10**(-22)/10**M_axion)
      rmax = 1.1 * debrogile_wavelength_order 
      rmin = 0.9 * debrogile_wavelength_order 
      flucs_args={'angle': np.random.uniform(0.0,2*np.pi,None), 'rmin': rmin, 'rmax': rmax}


      #First, use Pyhalo to create subhalo realization
      Subhalo_constructor= pyHalo.preset_models.preset_model_from_name('ULDM')
      Subhalo_realization = Subhalo_constructor(z_lens=zlens,z_source=zsource,log10_m_uldm = M_axion ,flucs_shape=flucs_shape,flucs_args=flucs_args,log_m_host=M_host,cone_opening_angle_arcsec=arcsec_opening_angle, LOS_normalization=LOS_Norm) 
      print(len(Subhalo_realization.halos))
      #Extract both lensing quantities from subhalo realization (reshifts, lensing models, lensing kwargs) and the cosmology instance from source to observer
      Sub_model_list, Sub_redshift_array, Sub_kwargs, _ = Subhalo_realization.lensing_quantities()
      cosmology = Subhalo_realization.astropy_instance

      #Calc Einstein radius
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

      theta_E = mass_to_radius(M_host,zsource,zdeflector)

      #Host halo
      Host_halo = 'EPL'
      Host_kwargs = {'theta_E':theta_E,'gamma': Host_gamma,'e1':0.4,'e2':-0.1,'center_x':0.0, 'center_y':0.0} 

      #External shear
      External_shear = 'SHEAR'
      Shear_kwargs = {'gamma1':0.03,'gamma2':0.01,'ra_0': 0, 'dec_0': 0} 

      #Macrolens
      Macro_model_list = [Host_halo,External_shear]
      Macro_kwargs_list = [Host_kwargs,Shear_kwargs]
      Macro_redshift_list = [zlens,zlens]

      #Combine macro-model with substructure 
      lens_model_list = Macro_model_list + Sub_model_list
      lens_kwargs_list = Macro_kwargs_list + Sub_kwargs
      lens_redshift_list = Macro_redshift_list + list(Sub_redshift_array)

      return lens_model_list,lens_kwargs_list,lens_redshift_list,cosmology,Macro_model_list, Macro_kwargs_list, Macro_redshift_list, arcsec_opening_angle, M_axion, flucs_shape,flucs_args
      
  def WDM_constructor(zsource, zlens, M_host, Host_gamma,LOS_Norm):
      '''This function constructs a lens (host halo, sub halo, LOS halos, and lens galaxy) under the assumption of WDM'''
      
      log_mc = np.random.uniform(5.0,10.0,size=None)
      arcsec_opening_angle = 10

      #First, use Pyhalo to create subhalo realization
      Subhalo_constructor= pyHalo.preset_models.preset_model_from_name('WDM')
      Subhalo_realization = Subhalo_constructor(z_lens=zlens,z_source=zsource,log_mc=log_mc,log_m_host=M_host,cone_opening_angle_arcsec=arcsec_opening_angle,LOS_normalization=LOS_Norm) 

      #Extract both lensing quantities from subhalo realization (reshifts, lensing models, lensing kwargs) and the cosmology instance from source to observer
      Sub_model_list, Sub_redshift_array, Sub_kwargs, _ = Subhalo_realization.lensing_quantities()
      cosmology = Subhalo_realization.astropy_instance

      #Calc Einstein radius
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

      theta_E = mass_to_radius(M_host,zsource,zdeflector)

      #Host Halo
      Host_halo = 'EPL'
      Host_kwargs = {'theta_E':theta_E,'gamma': Host_gamma,'e1':0.4,'e2':-0.1,'center_x':0.0, 'center_y':0.0} 

      #External shear
      External_shear = 'SHEAR'
      Shear_kwargs = {'gamma1':0.03,'gamma2':0.01,'ra_0': 0, 'dec_0': 0} 

      #Macrolens
      Macro_model_list = [Host_halo,External_shear]
      Macro_kwargs_list = [Host_kwargs,Shear_kwargs]
      Macro_redshift_list = [zlens,zlens]

      #Combine macro-model with substructure 
      lens_model_list = Macro_model_list + Sub_model_list
      lens_kwargs_list = Macro_kwargs_list + Sub_kwargs
      lens_redshift_list = Macro_redshift_list + list(Sub_redshift_array)

      return lens_model_list,lens_kwargs_list,lens_redshift_list,cosmology, Macro_model_list, Macro_kwargs_list, Macro_redshift_list, arcsec_opening_angle,log_mc

  def SIDM_constructor(zsource, zlens, M_host, Host_gamma,LOS_Norm):
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
      Subhalo_realization = Subhalo_constructor(z_lens=zlens,z_source=zsource,mass_ranges_subhalos=mass_ranges_subhalos,mass_ranges_field_halos=mass_ranges_field_halos,probabilities_subhalos=probabilities_subhalos,probabilities_field_halos=probabilities_field_halos,log_m_host=M_host,cone_opening_angle_arcsec=cone_opening_angle_arcsec,LOS_normalization=LOS_Norm) 

      #Extract both lensing quantities from subhalo realization (reshifts, lensing models, lensing kwargs) and the cosmology instance from source to observer
      Sub_model_list, Sub_redshift_array, Sub_kwargs, _ = Subhalo_realization.lensing_quantities()
      cosmology = Subhalo_realization.astropy_instance

      #Calc Einstein radius
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

      theta_E = mass_to_radius(M_host,zsource,zdeflector)

      #Host Halo
      Host_halo = 'EPL'
      Host_kwargs = {'theta_E':theta_E,'gamma': Host_gamma,'e1':0.4,'e2':-0.1,'center_x':0.0, 'center_y':0.0} 

      #External shear
      External_shear = 'SHEAR'
      Shear_kwargs = {'gamma1':0.03,'gamma2':0.01,'ra_0': 0.0, 'dec_0': 0.0} 

      #Macrolens
      Macro_model_list = [Host_halo,External_shear]
      Macro_kwargs_list = [Host_kwargs,Shear_kwargs]
      Macro_redshift_list = [zlens,zlens]

      #Combine macro-model with substructure 
      lens_model_list = Macro_model_list + Sub_model_list
      lens_kwargs_list = Macro_kwargs_list + Sub_kwargs
      lens_redshift_list = Macro_redshift_list + list(Sub_redshift_array)

      return lens_model_list,lens_kwargs_list,lens_redshift_list,cosmology, Macro_model_list, Macro_kwargs_list, Macro_redshift_list, mass_ranges_subhalos, mass_ranges_field_halos, probabilities_subhalos, probabilities_field_halos, cone_opening_angle_arcsec

  m_Host = Host_mass()
  slope_Host = Host_slope()
  zdeflector = redshifts[0]
  zsource = redshifts[1]

  
  if DM_type == 'CDM':
      lens_model_list,lens_kwargs_list,lens_redshift_list,cosmology,Macro_model_list, Macro_kwargs_list, Macro_redshift_list,arcsecond_opening_angle=CDM_constructor(zsource=zsource,zlens=zdeflector,M_host=m_Host,Host_gamma=slope_Host)
      return lens_model_list,lens_kwargs_list,lens_redshift_list,cosmology,Macro_model_list, Macro_kwargs_list, Macro_redshift_list,arcsecond_opening_angle,m_Host,slope_Host
  
  elif DM_type == 'WDM':
      lens_model_list,lens_kwargs_list,lens_redshift_list,cosmology,Macro_model_list, Macro_kwargs_list, Macro_redshift_list, log_mc, arcsecond_opening_angle =WDM_constructor(zsource=zsource,zlens=zdeflector,M_host=m_Host,Host_gamma=slope_Host)
      return lens_model_list,lens_kwargs_list,lens_redshift_list,cosmology,Macro_model_list, Macro_kwargs_list, Macro_redshift_list, log_mc, arcsecond_opening_angle,m_Host,slope_Host
  
  elif DM_type == 'SIDM':
      lens_model_list,lens_kwargs_list,lens_redshift_list,cosmology,Macro_model_list, Macro_kwargs_list, Macro_redshift_list, mass_ranges_subhalos, mass_ranges_field_halos, probabilities_subhalos, probabilities_field_halos, arcsecond_opening_angle=SIDM_constructor(zsource=zsource,zlens=zdeflector,M_host=m_Host,Host_gamma=slope_Host)
      return lens_model_list,lens_kwargs_list,lens_redshift_list,cosmology,Macro_model_list, Macro_kwargs_list, Macro_redshift_list, mass_ranges_subhalos, mass_ranges_field_halos, probabilities_subhalos, probabilities_field_halos, arcsecond_opening_angle,m_Host,slope_Host
  
  elif DM_type == 'Axion':
      lens_model_list,lens_kwargs_list,lens_redshift_list,cosmology,Macro_model_list, Macro_kwargs_list, Macro_redshift_list,arcsecond_opening_angle,M_axion,flucs_shape,flucs_args=Axion_constructor(zsource=zsource,zlens=zdeflector,M_host=m_Host,Host_gamma=slope_Host,M_axion=M_axion),M_axion
      return lens_model_list,lens_kwargs_list,lens_redshift_list,cosmology,Macro_model_list, Macro_kwargs_list, Macro_redshift_list,arcsecond_opening_angle, M_axion, flucs_shape, flucs_args,m_Host,slope_Host
  
  

