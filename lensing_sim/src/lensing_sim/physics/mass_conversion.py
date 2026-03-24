import numpy as np
from scipy.optimize import brentq
from lenstronomy.LensModel.Profiles.gnfw import GNFW
from lenstronomy.Cosmo.lens_cosmo import LensCosmo

def lensing_params_to_gnfw_M200(Lenspop_params, gamma_in=2):
    zl, zs, theta_E = Lenspop_params
    lens_cosmo = LensCosmo(zl, zs)
    gnfw = GNFW()

    def concentration_param_Dutton(M_200, z):
        Planck_h = 0.671
        M_200_Dutton = M_200 / 10**12 * Planck_h
        a = 0.520 + (0.905 - 0.520) * np.exp(-0.617 * z**(1.21))
        b = -0.101 + 0.026 * z
        return 10**(a + b * np.log10(M_200_Dutton))

    def lensing_eq(M_200):
        c200 = concentration_param_Dutton(M_200, zl)
        Rs, Rs_alpha = lens_cosmo.gnfw_physical2angle(M_200, c200, gamma_in=2)
        return theta_E - gnfw.alpha(theta_E, Rs, Rs_alpha, gamma_in)

    return brentq(lensing_eq, 10**11, 10**14)

