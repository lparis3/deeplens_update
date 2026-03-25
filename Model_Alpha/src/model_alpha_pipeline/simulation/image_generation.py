def simulate(...):
    # copied from dlu_3
    # run lenstronomy simulation for one case
    return image, exposure_time

def generate_images(setup_results, dlu_2_results):
    # call simulate(...) for full realization
    # call simulate(...) for nss realization
    # compute sns_diff
    # compute SNR
    # compute tot_exp_times

    return {
        "img": img,
        "img_nss": img_nss,
        "sns_diff": sns_diff,
        "SNR": SNR,
        "tot_exp_times": tot_exp_times,
    }
