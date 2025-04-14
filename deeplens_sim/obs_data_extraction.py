import h5py
from scipy.ndimage import gaussian_filter
import numpy as np
from scipy.ndimage import shift

def extraction(z_limits):
    '''Sample GalaxiesML Dataset for sources and deflectors.'''
 #Necessary functions for processing:
    def center_extraction(image):
        brightest_pixel_values = np.array([np.max(image[0,:,:]), np.max(image[1,:,:]),np.max(image[2,:,:])])
        brightest_band = brightest_pixel_values.argmax()
        center_rows_columns = np.unravel_index(image[brightest_band,:,:].argmax(),image[brightest_band,:,:].shape)
        center_pixel_coor = np.array([center_rows_columns[1], center_rows_columns[0]])
        return center_pixel_coor
        
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
        
        # Apply Gaussian convolution to smooth the image
        band_image = gaussian_filter(band_image, sigma=1)  # What sigma is best for us? 

        # Create elliptical Gaussian mask
        mask = create_elliptical_gaussian(band_image.shape, center, sigma_x, sigma_y, angle)

        # Apply mask to isolate the central galaxy
        masked_image = band_image * mask
        return masked_image

    def mag_booster(source_image,deflector_image,source_mag, deflector_mag):
        '''(maybe) Boosts the magnitude of lens until brightest pixel is brighter than corresponding pixel of deflector'''
        new_mag = source_mag
        if np.max(source_image)>deflector_image[np.unravel_index(np.argmax(source_image), np.shape(source_image))]:
            if np.random.uniform(0.0,1.0,None) < 0.5:
                new_mag = deflector_mag - 0.5
                return new_mag
            else:
                return new_mag
        else:
            new_mag = source_mag - np.random.uniform(0.0,0.5,None)
            return new_mag
     
    data_file = '5x127x127_testing_with_morphology.hdf5'
    with h5py.File(data_file, 'r') as file:
        if z_limits != None:
            #Deflector id
            idd_array = np.where((file['specz_redshift'] >= np.float64(z_limits[0])) & (file['specz_redshift']<= (np.float64(z_limits[1] - 0.25))))[0]
            idd = np.random.choice(idd_array)
            zdeflector = (file['specz_redshift'][idd])

            #Source id
            ids_array = np.array(np.where((file['specz_redshift'] >= np.float64(z_limits[0]))&(file['specz_redshift'] <= np.float64(z_limits[1]))&(file['specz_redshift'] >= np.float64(zdeflector + 0.25))))[0]
            ids = np.random.choice(ids_array)
        else:
            #Deflector id
            idd_array = np.where((file['specz_redshift'] <= (np.max(file['specz_redshift']) - 0.25)))[0]
            idd = np.random.choice(idd_array)
            zdeflector = (file['specz_redshift'][idd])
            
            #Source id
            ids_array = np.array(np.where(file['specz_redshift'] >= zdeflector + 0.25))[0]
            ids = np.random.choice(ids_array)


        #Deflector Info    
        #print(str(idd)+ " works!")
        deflector_morph = file['image'][idd]
        if np.min(deflector_morph.flatten()) < 0.0:
            deflector_morph[deflector_morph < 0] = 0.0
        center_d = center_extraction(deflector_morph)
        sigma_y_d = file['g_half_light_radius'][idd]*2. 
        sigma_x_d = file['g_half_light_radius'][idd]*2. * (1 - file['g_ellipticity'][idd])  
        angle_d = np.pi/2. - np.deg2rad(file['g_pos_angle'][idd]) 
        g_mag_d = file['g_cmodel_mag'][idd]
        r_mag_d = file['r_cmodel_mag'][idd]
        i_mag_d = file['i_cmodel_mag'][idd] 
        zdeflector = np.round(file['specz_redshift'][idd],2)
        
        
        #Source Info:
        source_morph = file['image'][ids]
        if np.min(source_morph.flatten()) < 0.0:
            source_morph[source_morph < 0] = 0.0
        center_s = center_extraction(source_morph)
        sigma_y_s = file['g_half_light_radius'][ids]*2. 
        sigma_x_s = file['g_half_light_radius'][ids]*2. * (1 - file['g_ellipticity'][ids])  
        angle_s = np.pi/2. - np.deg2rad(file['g_pos_angle'][ids]) 
        g_mag_s = file['g_cmodel_mag'][ids]
        r_mag_s = file['r_cmodel_mag'][ids]
        i_mag_s = file['i_cmodel_mag'][ids]
        zsource = np.round(file['specz_redshift'][ids],2)
       
    
    source_images = np.array([src_g,src_i,src_r])
    source_mag = np.array([g_mag_s,r_mag_s,i_mag_s])
    raw_info_src = {'raw_img': source_morph, 'pros_img': source_images, 'raw_center':center_s}
    
    deflector_images = np.array([dfr_g,dfr_r,dfr_i])
    deflector_mag = np.array([g_mag_d,r_mag_d,i_mag_d])
    raw_info_dfr = {'raw_img': deflector_morph, 'pros_img': deflector_images, 'raw_center':center_d}

    source_mag_boost = mag_booster(source_images,deflector_images,source_mag,deflector_mag)
    redshifts = [zdeflector,zsource]
    return source_images,source_mag_boost,deflector_images,deflector_mag, redshifts, raw_info_src, raw_info_dfr


    
   



