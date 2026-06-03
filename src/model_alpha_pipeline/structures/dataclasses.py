from dataclasses import dataclass
import numpy as np
from typing import Any
from typing import Optional


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
    bands: list
    band_labels: list
    needed_hsc_bands: list
    # In INTERPOL mode these hold the per-band processed pixel cutouts.
    # In SERSIC mode they are None (the analytic profile carries the
    # spatial info, no pixel image needed).
    source_images: Optional[np.ndarray]
    source_mag: np.ndarray
    deflector_images: Optional[np.ndarray]
    deflector_mag: np.ndarray
    raw_src: Optional[dict]
    raw_dfr: Optional[dict]
    # Per-galaxy Sersic shape parameters (R_sersic, n_sersic, e1, e2).
    # Populated in SERSIC mode only; None in INTERPOL mode. Stored as a
    # dict so values can later be drawn per galaxy without changing the
    # downstream interface.
    source_sersic_params: Optional[dict] = None
    deflector_sersic_params: Optional[dict] = None