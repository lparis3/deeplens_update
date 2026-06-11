#%%
"""
Confirms that the band-shared rendering path used in image_generation._lensed_source_sb

        ra, dec      = imSim.ImageNumerics.coordinates_evaluate
        beta_x,beta_y = imSim.LensModel.ray_shooting(ra, dec, kwargs_lens)
        sb           = imSim.SourceModel.surface_brightness(beta_x, beta_y, kwargs_source)
                       * imSim._flux_scaling
        img          = imSim.ImageNumerics.re_size_convolve(sb, unconvolved=False)

reproduces lenstronomy's all-in-one

        imSim.image(kwargs_lens, kwargs_source, None,
                    source_add=True, lens_light_add=False, point_source_add=False)

exactly. The point of the optimization is that ray_shooting is band-independent, so
it can be computed once and reused; this script checks that splitting it out changes
nothing numerically.

Setup: SIE lens (no subhalos), elliptical Sersic source, random parameters, repeated
over several trials. Run:  python verify_rayshooting_equivalence.py
"""

import numpy as np
from lenstronomy.SimulationAPI.sim_api import SimAPI
from lenstronomy.SimulationAPI.ObservationConfig.LSST import LSST

# --------------------------------------------------------------------------
# Fixed setup: build the ImageModel once (data grid + PSF don't change between
# trials -- only the lens/source parameters do). Built via SimAPI so the imSim
# object is exactly the kind image_generation uses.
# --------------------------------------------------------------------------
NUMPIX = 127
N_TRIALS = 5
SEED = 0
ATOL = 1e-10                      # pass tolerance on max abs difference

kwargs_model = {
    "lens_model_list": ["SIE"],
    "source_light_model_list": ["SERSIC_ELLIPSE"],
    "z_source": 2.0,
}
kwargs_numerics = {"point_source_supersampling_factor": 1}

band = LSST(band="i", psf_type="GAUSSIAN", coadd_years=10).kwargs_single_band()
# positional args -> robust to the numpix/num_pix kwarg rename across lenstronomy versions
sim = SimAPI(NUMPIX, band, kwargs_model)
imSim = sim.image_model_class(kwargs_numerics)


def random_params(rng):
    """Random but sensible SIE + Sersic parameters."""
    kwargs_lens = [{
        "theta_E":  rng.uniform(1.0, 1.8),
        "e1":       rng.uniform(-0.2, 0.2),
        "e2":       rng.uniform(-0.2, 0.2),
        "center_x": rng.uniform(-0.1, 0.1),
        "center_y": rng.uniform(-0.1, 0.1),
    }]
    kwargs_source = [{
        "amp":      rng.uniform(5.0, 20.0),
        "R_sersic": rng.uniform(0.2, 0.6),
        "n_sersic": rng.uniform(1.0, 4.0),
        "e1":       rng.uniform(-0.3, 0.3),
        "e2":       rng.uniform(-0.3, 0.3),
        "center_x": rng.uniform(-0.3, 0.3),
        "center_y": rng.uniform(-0.3, 0.3),
    }]
    return kwargs_lens, kwargs_source


def reference_image(kwargs_lens, kwargs_source):
    """lenstronomy's all-in-one source render (what the pipeline currently calls)."""
    return imSim.image(
        kwargs_lens, kwargs_source, None,
        source_add=True, lens_light_add=False, point_source_add=False,
    )


def shared_rayshooting_image(kwargs_lens, kwargs_source):
    """The split path: ray-shoot, evaluate source at the deflected coords, convolve."""
    ra, dec = imSim.ImageNumerics.coordinates_evaluate
    beta_x, beta_y = imSim.LensModel.ray_shooting(ra, dec, kwargs_lens)
    sb = imSim.SourceModel.surface_brightness(beta_x, beta_y, kwargs_source) * imSim._flux_scaling
    return imSim.ImageNumerics.re_size_convolve(sb, unconvolved=False)


def main():
    rng = np.random.default_rng(SEED)
    print(f"SIE lens + elliptical Sersic source | {NUMPIX}x{NUMPIX} | {N_TRIALS} random trials\n")
    all_ok = True
    for t in range(1, N_TRIALS + 1):
        kwargs_lens, kwargs_source = random_params(rng)
        ref = reference_image(kwargs_lens, kwargs_source)
        new = shared_rayshooting_image(kwargs_lens, kwargs_source)

        max_abs = float(np.max(np.abs(ref - new)))
        max_rel = max_abs / float(np.max(np.abs(ref)))
        exact = np.array_equal(ref, new)
        close = np.allclose(ref, new, rtol=0, atol=ATOL)
        all_ok &= close

        status = "PASS" if close else "FAIL"
        print(f"trial {t}: theta_E={kwargs_lens[0]['theta_E']:.3f} "
              f"n_sersic={kwargs_source[0]['n_sersic']:.2f}  "
              f"max|Δ|={max_abs:.2e}  rel={max_rel:.2e}  "
              f"exact_equal={exact}  [{status}]")

    print()
    if all_ok:
        print("ALL TRIALS PASS: ray_shooting + surface_brightness + re_size_convolve == image()")
    else:
        raise SystemExit("MISMATCH DETECTED: the split path does not match image().")


if __name__ == "__main__":
    main()
# %%
