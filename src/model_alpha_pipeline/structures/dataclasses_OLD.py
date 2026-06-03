from dataclasses import dataclass
import numpy as np
from typing import Any


@dataclass

class sampled_values:
    redshifts:np.ndarray
    host_theta_E_arcsecond:float
    M_host:float
    max_subhalo_mass:float


@dataclass

class dlu_1_output:
    lens_model_list:list
    lens_kwargs_list:list
    lens_redshift_list:list
    cosmology:Any
    macro_model_list:list
    macro_kwargs_list:list
    macro_redshift_list:list
    arcsecond_opening_angle:int
    host_mass:float
    whole_halo_mass:float
    num_subhalos:int
    slope_Host:float
    type_kwargs:dict
    

@dataclass
class dlu_2_output:
    bands:list
    band_labels:list
    needed_hsc_bands:list
    source_images:np.ndarray
    source_mag:np.ndarray
    deflector_images:np.ndarray
    deflector_mag:np.ndarray
    raw_src:dict
    raw_dfr:dict
