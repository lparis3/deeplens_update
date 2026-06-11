import numpy as np
from scipy.ndimage import shift
from scipy.ndimage import gaussian_filter


def create_elliptical_gaussian(shape, center, sigma_x, sigma_y, angle=0):
    """Generate a 2D elliptical Gaussian mask."""
    y, x = np.indices(shape)
    x0, y0 = center
    x = x - x0
    y = y - y0

    # Apply rotation for the angle
    x_rot = x * np.cos(angle) - y * np.sin(angle)
    y_rot = x * np.sin(angle) + y * np.cos(angle)

    # Elliptical Gaussian function
    mask = np.exp(-((x_rot**2 / (2 * sigma_x**2)) + (y_rot**2 / (2 * sigma_y**2))))
    return mask


def src_process(band_image,center,sigma_x,sigma_y,angle):
    # Center source or deflector in image
    x_shift = np.shape(band_image)[0]/2 - center[0]
    y_shift = np.shape(band_image)[1]/2 - center[1]
    band_image = shift(band_image,[y_shift,x_shift],mode='constant')
    center = [np.shape(band_image)[0]/2,np.shape(band_image)[0]/2]

    # Background estimation (simple mean of edge pixels for example)
    background_level = np.mean(np.concatenate([band_image[0, :], band_image[-1, :], band_image[:, 0], band_image[:, -1]]))
    band_image -= background_level  # Subtract background

    #Remove all unphysical negative pixels
    band_image[band_image < 0] = 0

    # Apply Gaussian convolution to smooth the image
    band_image = gaussian_filter(band_image, sigma=1)  # What sigma is best for us?

    # Create elliptical Gaussian mask
    mask = create_elliptical_gaussian(band_image.shape, center, sigma_x, sigma_y, angle)

    # Apply mask to isolate the central galaxy
    masked_image = band_image * mask
    return masked_image
