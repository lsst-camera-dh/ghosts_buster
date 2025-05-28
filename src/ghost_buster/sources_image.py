import numpy as np
from astropy.stats import sigma_clipped_stats
from photutils.segmentation import detect_sources, SourceCatalog
from scipy.ndimage import gaussian_filter

version = "0.2"

def statsSmoothImage(image):
    '''

    Parameters
    ----------
    image : np.array
        image.fits.getArray()
        Bin's values
        
    Returns
    -------
    smoothed : np.array
        image smooth
    mean : float
        mean value of smoothed image
    median : float
        median value of smoothed image
    std : float
        std value of smoothed image

    '''
    image = np.nan_to_num(image, nan=1e-10)
    smoothed = gaussian_filter(image, sigma=3.0)
    mean, median, std = sigma_clipped_stats(smoothed, sigma=3.0)
    return smoothed, mean, median, std

def statsImage(image):
    '''

    Parameters
    ----------
    image : np.array
        image.fits.getArray()
        Bin's values
        
    Returns
    -------
    mean : float
        mean value of image
    median : float
        median value of image
    std : float
        std value of image

    '''
    image = np.nan_to_num(image, nan=1e-10)
    mean, median, std = sigma_clipped_stats(image, sigma=3.0)
    return mean, median, std
    
def removeSources(image):
    '''

    Parameters
    ----------
    image : np.array
        image.fits.getArray()
        Bin's values

    Returns
    -------
    ghosts_only : np.array
        image without brightness sources
        Bin's values

    '''
    smoothed, mean, median, std = statsSmoothImage(image)
    threshold = median + 3.0 * std
    segm = detect_sources(smoothed, threshold=threshold, npixels=5)
    source_mask = segm.make_source_mask()
    ghosts_only = np.where(source_mask, 0.0, image)
    return ghosts_only

def removeSourcesBoth(image, hist):
    '''

    Parameters
    ----------
    image : np.array
        image.fits.getArray()
        Bin's values

    Returns
    -------
    ghosts_only : np.array
        image without brightness sources
        Bin's values

    '''
    smoothed, mean, median, std = statsSmoothImage(image)
    threshold = median + 5.0 * std
    segm = detect_sources(smoothed, threshold=threshold, npixels=5)
    source_mask = segm.make_source_mask()
    image = np.where(source_mask, 0.0, image)
    hist = np.where(source_mask, 0.0, hist)
    return image, hist

def extractSources(image):
    '''
    Parameters
    ----------
    image : np.array
        image.fits.getArray()
        Bin's values

    Returns
    -------
    sources_only : np.array
        image with only brightness sources
        Bin's values
    '''
    smoothed, mean, median, std = statsSmoothImage(image)
    threshold = median + 3.0 * std
    segm = detect_sources(smoothed, threshold=threshold, npixels=5)
    source_mask = segm.make_source_mask()
    sources_only = np.where(source_mask, image, 0.0)
    return sources_only

def getCatalog(image):
    '''

    Parameters
    ----------
    image : np.array
        image.fits.getArray()
        Bin's values

    Returns
    -------
    catalog : SourceCatalog
        Catalog of brightness sources on the image

    '''
    smoothed, mean, median, std = statsSmoothImage(image)
    threshold = median + 8.0 * std
    segm = detect_sources(smoothed, threshold=threshold, npixels=3)
    catalog = SourceCatalog(image, segm)
    return catalog
