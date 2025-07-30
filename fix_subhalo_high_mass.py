# %%
#Sample subhalo mass percentage with m_sub_high/M_host = 10^(-1)
import pyHalo
import pyHalo.preset_models
from astropy.constants import M_sun
import numpy as np
import h5py

log_mass_host = [12,13,14]

subhalo_percentage_12= np.zeros(1000)
subhalo_percentage_13= np.zeros(1000)
subhalo_percentage_14= np.zeros(1000)

for log_host_mass in log_mass_host:
   sims = 0
   while sims < 1000:
        data_file = 'deeplens_update.hdf5'
        file = h5py.File(data_file, 'r') 

        #Deflector 
        idd_array = np.where((file['specz_redshift'] <= (np.max(file['specz_redshift']) - 0.25)))[0]
        idd = np.random.choice(idd_array)
        zdeflector = (file['specz_redshift'][idd])
                        
        #Source 
        ids_array = np.array(np.where(file['specz_redshift'] >= zdeflector + 0.25))[0]
        ids = np.random.choice(ids_array)
        zsource = (file['specz_redshift'][ids])
       
        good = False
        attempts = 0
        while good == False and attempts < 10:
            try:
                Subhalo_constructor= pyHalo.preset_models.preset_model_from_name('CDM')
                Subhalo_realization = Subhalo_constructor(z_lens=zdeflector,z_source=zsource,log_m_host=log_host_mass,cone_opening_angle_arcsec=10, LOS_normalization=1.0,log_mhigh=log_host_mass-1)     
                Subahlo_masses_sum = np.sum([halo.mass * M_sun.value for halo in Subhalo_realization.halos])
                Whole_Halo_Mass = (10**log_host_mass * M_sun.value) + Subahlo_masses_sum
                subhalo_mass_percentage = Subahlo_masses_sum/(Whole_Halo_Mass)
                if log_host_mass == 12:
                    subhalo_percentage_12[sims] = subhalo_mass_percentage
                if log_host_mass == 13:
                    subhalo_percentage_13[sims] = subhalo_mass_percentage
                if log_host_mass == 14:
                    subhalo_percentage_14[sims] = subhalo_mass_percentage  
                print(f'Sim {sims+1} done!')
                good = True
                sims += 1
            
            except Exception as e:
                zsource = zsource + np.random.uniform(0.25,0.5)
                attempts += 1
                good = False
       

#Histograms
#%%
import matplotlib.pyplot as plt
plt.hist(subhalo_percentage_12,1000)
plt.title(f'Subhalo Mass Percentage Distribution for log10M_halo = {log_mass_host[0]} M_sun')

#%%
plt.hist(subhalo_percentage_13,1000)
plt.title(f'Subhalo Mass Percentage Distribution for log10M_halo = {log_mass_host[1]} M_sun')

#%%
plt.hist(subhalo_percentage_14,1000)
plt.title(f'Subhalo Mass Percentage Distribution for log10M_halo = {log_mass_host[2]} M_sun')


#%%
#Most populated bins
hist, bin_edges = np.histogram(subhalo_percentage_12,1000)
max_index = np.where(hist == np.max(hist))
max_edge1 = bin_edges[max_index]
max_edge2 = bin_edges[max_index[0]+1]
print(f'Most highly popultaed bin for log10M_host = {log_mass_host[0]} is {max_edge1}-{max_edge2}')

#%%
hist, bin_edges = np.histogram(subhalo_percentage_13,1000)
max_index = np.where(hist == np.max(hist))
max_edge1 = bin_edges[max_index]
max_edge2 = bin_edges[max_index[0]+1]
print(f'Most highly popultaed bin for log10M_host = {log_mass_host[1]} is {max_edge1}-{max_edge2}')

#%%
hist, bin_edges = np.histogram(subhalo_percentage_14,1000)
max_index = np.where(hist == np.max(hist))
max_edge1 = bin_edges[max_index]
max_edge2 = bin_edges[max_index[0]+1]
print(f'Most highly popultaed bin for log10M_host = {log_mass_host[2]} is {max_edge1}-{max_edge2}')


#%%
#Mean value

print(f'Mean of subhalo percentage for log10M_host = {log_mass_host[0]} is {np.mean(subhalo_percentage_12)}')
print(f'Mean of subhalo percentage for log10M_host = {log_mass_host[1]} is {np.mean(subhalo_percentage_13)}')
print(f'Mean of subhalo percentage for log10M_host = {log_mass_host[2]} is {np.mean(subhalo_percentage_14)}')

# %%
