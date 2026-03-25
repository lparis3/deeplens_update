from dataclasses import dataclass
import numpy as np


@dataclass
class SampledParameters:
    """
    Output of the normalizing flow sampling stage.
    """
    z_lens: float
    z_source: float
    theta_E: float
    gamma: float
    e1: float
    e2: float
    center_x: float
    center_y: float


@dataclass
class LensSystem:
    """
    Output of stage 1: lens + subhalo realization.
    """
    z_lens: float
    z_source: float
    theta_E: float
    gamma: float
    e1: float
    e2: float
    center_x: float
    center_y: float

    # Substructure / halo information
    halo_model: object
    kwargs_lens: list
    kwargs_lens_light: list
    kwargs_ps: list


@dataclass
class ObservationData:
    """
    Output of stage 2: selected and processed galaxy image.
    """
    image: np.ndarray
    exposure_time: float
    background_rms: float
    pixel_scale: float

    # Optional metadata
    source_redshift: float
    lens_redshift: float
