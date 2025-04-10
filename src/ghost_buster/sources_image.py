import lsst.afw.image as afwimage
import matplotlib.pyplot as plt
import numpy as np
from astropy.stats import sigma_clipped_stats
from photutils.segmentation import detect_sources, SourceCatalog
from scipy.ndimage import gaussian_filter

def statsSmoothImage(image):
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    smoothed = gaussian_filter(image, sigma=3.0)
    mean, median, std = sigma_clipped_stats(smoothed, sigma=3.0)
    return smoothed, mean, median, std

def statsImage(image):
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    mean, median, std = sigma_clipped_stats(image, sigma=3.0)
    return mean, median, std
    
def removeSources(image):
    smoothed, mean, median, std = statsSmoothImage(image)
    threshold = median + 3.0 * std
    segm = detect_sources(smoothed, threshold=threshold, npixels=5)
    source_mask = segm.make_source_mask()
    ghosts_only = np.where(source_mask, 0.0, image)
    return ghosts_only

'''
A déplacé dans display_image.py (faudra faire un import)
'''
def displayRemoveSources(image, image_ghosts):
    mean, median, std = statsImage(image)
    fig = plt.figure(figsize=(16, 16))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray', origin='lower', vmin=median, vmax=median + 3*std)
    plt.title("Image originale")
    plt.subplot(1, 2, 2)
    plt.imshow(image_ghosts, cmap='gray', origin='lower', vmin=median, vmax=median + 3*std)
    plt.title("Image sans sources (ghosts potentiels)")
    plt.tight_layout()
    plt.show()
    return fig

def getCatalog(image):
    smoothed, mean, median, std = statsSmoothImage(image)
    threshold = median + 3.0 * std
    segm = detect_sources(smoothed, threshold=threshold, npixels=5)
    catalog = SourceCatalog(image, segm)
    return catalog

def getBrightnessCoord(catalog):
    fluxes = np.array([src.segment_flux for src in catalog])
    idx_max = np.argmax(fluxes)
    brightest_src = catalog[idx_max]
    x_star = brightest_src.xcentroid
    y_star = brightest_src.ycentroid
    print(f"Coordonnées de l’étoile la plus brillante : x = {x_star:.2f}, y = {y_star:.2f}")
    return x_star, y_star

def getCoordBatoid(image, bins=8):
    ny, nx = image.shape
    x0, y0 = nx / 2, ny / 2
    scale = 0.2
    catalog = getCatalog(image)
    x_star, y_star = getBrightnessCoord(catalog)
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
