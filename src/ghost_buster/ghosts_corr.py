import numpy as np
from scipy.signal import correlate

version = "0.1"

def normalize(data):
    '''

    Parameters
    ----------
    data : np.array
        array of bins

    Returns
    -------
    np.array
        data normalize, values between 0 and 1

    '''
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def getCorr(image, ghosts):
    '''

    Parameters
    ----------
    image : np.array
        image.fits.getArray()
        Bin's values
    ghosts : np.array
        Simulated bins values of each ghosts

    Returns
    -------
    np.array
        Best match position found by cross-correlation between the image and each ghost

    '''
    image = np.nan_to_num(image, nan=0.0)
    image = normalize(image)
    correlation_results = []
    
    for i, ghost in enumerate(ghosts):
    
        if np.all(ghost == 0.0):
            print(f'Correlation {i+1} skip, nodata')
            correlation_results.append([0.0, 0.0])
            continue
            
        ghost = normalize(ghost)
        correlation_result = correlate(image, ghost, mode='same', method='fft')
        best_match_position = np.unravel_index(np.argmax(correlation_result), correlation_result.shape)
        
        #plt.imshow(correlation_result, cmap='hot')
        #plt.title(f"Correlation template {i+1}\nBest match found at position: ({best_match_position[0]}, {best_match_position[1]})")
        #plt.colorbar()
        #plt.show()
    
        correlation_results.append([best_match_position[0], best_match_position[1]])

    return np.array(correlation_results)

def getCorrGhosts(image, ghosts):
    '''

    Parameters
    ----------
    image : np.array
        image.fits.getArray()
        Bin's values
    ghosts : np.array
        Simulated bins values of each ghosts

    Returns
    -------
    ghosts_keep : np.array
        Array of ghosts index which pass the selection

    '''
    corr = getCorr(image, ghosts)
    image_center = np.array([image.shape[1] / 2, image.shape[0] / 2])
    ghosts_keep = corr - image_center
    mask = (np.abs(ghosts_keep[:, 0]) < 500) & (np.abs(ghosts_keep[:, 1]) < 500)
    ghosts_keep = np.where(mask)[0]
    return ghosts_keep

def selectGhosts(x, y, flux, ghosts_keep):
    '''

    Parameters
    ----------
    Parameters
    ----------
    x : np.array
        x coordinates of rays with a dimension for each ghost
    y : np.array
        y coordinates of rays with a dimension for each ghost
    flux : np.array
        flux of rays with a dimension for each ghost
    ghosts_keep : np.array
        Array of ghosts index which pass the selection

    Returns
    -------
    x_ghosts : np.array
        x coordinates of ghosts keep
    y_ghosts : np.array
        y coordinates of ghosts keep
    flux_ghosts : np.array
        flux of ghosts keep

    '''
    x_ghosts = np.concatenate([x[i] for i in ghosts_keep])
    y_ghosts = np.concatenate([y[i].y for i in ghosts_keep])
    flux_ghosts = np.concatenate([flux[i] for i in ghosts_keep])
    return x_ghosts, y_ghosts, flux_ghosts
