import numpy as np
from astropy.stats import sigma_clipped_stats
from photutils.segmentation import detect_sources, SourceCatalog
from scipy.ndimage import gaussian_filter

version = "0.1"

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
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
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
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
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

def getBrightnessCoord(catalog):
    '''

    Parameters
    ----------
    catalog : SourceCatalog
        Catalog of brightness sources on the image

    Returns
    -------
    x_star : float
        x coordinate of the brightness star in the catalog
    y_star : float
        y coordinate of the brightness star in the catalog

    '''
    fluxes = np.array([src.segment_flux for src in catalog])
    idx_max = np.argmax(fluxes)
    brightest_src = catalog[idx_max]
    x_star = brightest_src.xcentroid
    y_star = brightest_src.ycentroid
    # x_star, y_star = 304.0282151518103, 733.3845556158473 # Test with new origin, Gaia and Wcs LSST
    print(f"Coordonnées de l’étoile la plus brillante : x = {x_star:.2f}, y = {y_star:.2f}")
    return x_star, y_star

def getCoordBatoid(image, bins=8):
    '''

    Parameters
    ----------
    image : np.array
        image.fits.getArray()
        Bin's values
    bins : int, optional
        Binning of image's data
        The default is 8 (Because i currently work with this binning)

    Returns
    -------
    theta_x : float
        theta_x coordinate for Batoid
    theta_y : float
        theta_y coordinate for Batoid

    '''
    ny, nx = image.shape
    x0, y0 = nx / 2, ny / 2
    scale = 0.2003375
    catalog = getCatalog(image)
    x_star, y_star = getBrightnessCoord(catalog)
    # x_star = np.array([303.42590378617035, -129.52970241622484])
    # y_star = np.array([733.5446069859133, 730.8892532436698]) # Meilleur coordonnées trouver sur Gaia
    dx = (x_star - x0) * scale * bins
    dy = (y_star - y0) * scale * bins
    theta_x = dx / 3600
    theta_y = dy / 3600
    print(f"starPos Batoid = [{theta_x:.4f}, {theta_y:.4f}]")
    return theta_x, theta_y
    
#ghost_292 = afwimage.ImageF.readFits("../../notebooks/comcam_ghosts/ghost_292.fits", 0)
#img = ghost_292.getArray()
#ghost_img = removeSources(img)
#displayRemoveSources(img, ghost_img)
