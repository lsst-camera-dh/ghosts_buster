import numpy as np
from scipy.signal import correlate

def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def getCorr(image, ghosts):
    image = np.nan_to_num(image, nan=0.0)
    image = normalize_image(image)
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
    corr = getCorr(image, ghosts)
    image_center = np.array([image.shape[1] / 2, image.shape[0] / 2])
    ghost_keep = correlation_results - image_center
    mask = (np.abs(ghost_keep[:, 0]) < 500) & (np.abs(ghost_keep[:, 1]) < 500)
    ghost_keep = np.where(mask)[0]
    return ghost_keep

def selectGhosts(x, y, flux):
    x_ghosts = np.concatenate([x[i] for i in ghost_keep])
    y_ghosts = np.concatenate([y[i].y for i in ghost_keep])
    flux_ghosts = np.concatenate([flux[i] for i in ghost_keep])
    return x_ghosts, y_ghosts, flux_ghosts
