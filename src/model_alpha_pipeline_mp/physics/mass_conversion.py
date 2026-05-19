import numpy as np
from scipy.optimize import brentq

from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.LensModel.Profiles.gnfw import GNFW


def concentration_param_dutton(m_200: float, z: float) -> float:
    """
    Return the halo concentration c200 using the Dutton & Maccio (2014)
    relation for an NFW halo.

    Parameters
    ----------
    m_200 : float
        Halo mass in solar masses.
    z : float
        Redshift.

    Returns
    -------
    float
        Concentration parameter c200.
    """
    planck_h = 0.671
    m_200_dutton = (m_200 / 10**12) * planck_h
    a = 0.520 + (0.905 - 0.520) * np.exp(-0.617 * z**1.21)
    b = -0.101 + 0.026 * z
    c200 = 10 ** (a + b * np.log10(m_200_dutton))
    return c200


def lensing_params_to_gnfw_m200(lenspop_params, gamma_in: float = 2.0) -> float:
    """
    Convert sampled lensing observables into an estimated GNFW M200.

    Parameters
    ----------
    lenspop_params : array-like
        Expected to contain (z_lens, z_source, theta_E).
        If a longer array is passed, only the first, second, and last
        entries are used.
    gamma_in : float, optional
        Inner slope parameter for the GNFW profile.

    Returns
    -------
    float
        Estimated M200 in solar masses.
    """
    zl = lenspop_params[0]
    zs = lenspop_params[1]
    theta_e = lenspop_params[-1]

    lens_cosmo = LensCosmo(zl, zs)
    gnfw = GNFW()

    def lensing_equation(m_200: float) -> float:
        c200 = concentration_param_dutton(m_200, z=zl)
        rs, rs_alpha = lens_cosmo.gnfw_physical2angle(m_200, c200, gamma_in=gamma_in)
        reduced_deflection_angle = gnfw.alpha(theta_e, rs, rs_alpha, gamma_in)
        return theta_e - reduced_deflection_angle

    solution = brentq(lensing_equation, 10**11, 10**14)
    return solution
