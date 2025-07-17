import pylab as plt
import numpy as np
import lsst.afw.display as afwDisplay # type: ignore
from astropy.stats import sigma_clipped_stats
from .sources_image import statsImage, extractSources, removeSources

version = "0.3"

def displayImage(image, title=None, name=None):
    """
    Display an astronomical image using LSST's afwDisplay with asinh scaling and plasma colormap.

    Parameters
    ----------
    image : array-like
        The image data to display.
    title : str, optional
        Title for the plot.
    name : str, optional
        If provided, saves the figure to this filename.

    Returns
    -------
    afwDisplay.Display
        The display object for further manipulation.
    """
    afwDisplay.setDefaultBackend('matplotlib') 
    fig = plt.figure(figsize=(10,10))
    afw_display = afwDisplay.Display(1)
    afw_display.scale('asinh', 'zscale')
    afw_display.setImageColormap(cmap='plasma')
    afw_display.mtv(image)
    plt.title(title)
    plt.gca().axis('off')

    if name != None:
        plt.savefig(name, bbox_inches='tight')
    plt.show()
    return afw_display
    
def displayImageGhosts(image, zmin=0, zmax=5000, title=None, name=None):
    """
    Display an image with linear scaling and plasma colormap, for ghost visualization.

    Parameters
    ----------
    image : array-like
        The image data to display.
    zmin : float, optional
        Minimum value for scaling.
    zmax : float, optional
        Maximum value for scaling.
    title : str, optional
        Title for the plot.
    name : str, optional
        If provided, saves the figure to this filename.

    Returns
    -------
    afwDisplay.Display
        The display object for further manipulation.
    """
    afwDisplay.setDefaultBackend('matplotlib') 
    fig = plt.figure(figsize=(10,10))
    afw_display = afwDisplay.Display(1)
    afw_display.scale('linear', min=zmin, max=zmax)
    afw_display.setImageColormap(cmap='plasma')
    afw_display.mtv(image)
    plt.title(title)
    plt.gca().axis('off')
    if name != None:
        plt.savefig(name, bbox_inches='tight')
    plt.show()
    return afw_display

def displayImageGhostsBW(image, title=None, name=None):
    """
    Display an image using asinh scaling and grayscale colormap, for ghost visualization.

    Parameters
    ----------
    image : array-like
        The image data to display.
    title : str, optional
        Title for the plot.
    name : str, optional
        If provided, saves the figure to this filename.

    Returns
    -------
    afwDisplay.Display
        The display object for further manipulation.
    """
    afwDisplay.setDefaultBackend('matplotlib') 
    fig = plt.figure(figsize=(10,10))
    afw_display = afwDisplay.Display(1)
    afw_display.scale('asinh', 'zscale')
    afw_display.setImageColormap(cmap='grey')
    afw_display.mtv(image)
    plt.title(title)
    plt.gca().axis('off')
    if name != None:
        plt.savefig(name, bbox_inches='tight')
    plt.show()
    return afw_display

def displayImageGhostsFlux(image, minflux=955, maxflux=990, name=None):
    """
    Display only the pixels in the image whose flux is within a specified range.

    Parameters
    ----------
    image : array-like
        The image data to display.
    minflux : float, optional
        Minimum flux value to display.
    maxflux : float, optional
        Maximum flux value to display.
    name : str, optional
        If provided, saves the figure to this filename.
    """
    flux_mask = (image >= minflux) & (image <= maxflux)
    pixels_in_range = image[flux_mask]
    masked_data = np.where(flux_mask, image, 0.0)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    im = ax.imshow(masked_data, origin='lower', vmin=minflux, vmax=maxflux)
    fig.colorbar(im, ax=ax, label='Flux')
    ax.set_title(f"Pixels in [{minflux} <= flux <= {maxflux}]")
    if name != None:
        plt.savefig(name, bbox_inches='tight')
    plt.show()

def displayReal(image, name=None):
    """
    Display the original image with grayscale colormap and sigma-clipped scaling.

    Parameters
    ----------
    image : array-like
        The image data to display.
    name : str, optional
        If provided, saves the figure to this filename.
    """
    mean, median, std = sigma_clipped_stats(image, sigma=3.0)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    im = ax.imshow(image, cmap='gray', origin='lower', vmin=median, vmax=median + 3*std)
    plt.title("Original image")

    if name != None:
        plt.savefig(name, bbox_inches='tight')
    plt.show()

def displaySub(image, name=None, starpos=None):
    """
    Display a subimage with optional marker for a star position.

    Parameters
    ----------
    image : array-like
        The image data to display.
    name : str, optional
        If provided, saves the figure to this filename.
    starpos : tuple, optional
        Coordinates (x, y) of the star to mark.
    """
    mean, median, std = sigma_clipped_stats(image, sigma=3.0)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    im = ax.imshow(image, cmap='gray', origin='lower', vmin=median, vmax=median + 3*std)

    if starpos != None:
        ax.plot(starpos[0]-185, starpos[1]-140, '+', c='r')
    plt.title("Subimage")

    if name != None:
        plt.savefig(name, bbox_inches='tight')
    plt.show()

def displaySimu(hist, flux_update=False, name=None):
    """
    Display a simulated image with optional flux scaling.

    Parameters
    ----------
    hist : array-like
        The simulated image data to display.
    flux_update : bool, optional
        Whether to use high flux scaling.
    name : str, optional
        If provided, saves the figure to this filename.
    """
    if flux_update == True:
        vm = 4.0
    else:
        vm = 4e-2
        
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    ax.set_facecolor('black')
    plt.imshow(hist, origin='lower', vmax=vm)
    plt.xticks([])
    plt.yticks([])
    
    if name != None:
        plt.savefig(name, bbox_inches='tight')
    plt.show()

def displayClean(image, name=None):
    """
    Display a cleaned image with grayscale colormap and sigma-clipped scaling.

    Parameters
    ----------
    image : array-like
        The cleaned image data to display.
    name : str, optional
        If provided, saves the figure to this filename.
    """
    mean, median, std = sigma_clipped_stats(image, sigma=1.0)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    im = ax.imshow(image, cmap='gray', origin='lower', vmin=median, vmax=median + 1*std)
    plt.title("Clean image")
    fig.colorbar(im, ax=ax)
    plt.xticks([])
    plt.yticks([])
    
    if name != None:
        plt.savefig(name, bbox_inches='tight')
    plt.show()

def displayFit(image, temp, clean, flux_update=False, name=None):
    """
    Display original, template, and cleaned images side by side.

    Parameters
    ----------
    image : array-like
        The original image data.
    temp : array-like
        The template image data.
    clean : array-like
        The cleaned image data.
    flux_update : bool, optional
        Whether to use high flux scaling for the template.
    name : str, optional
        If provided, saves the figure to this filename.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    mean, median, std = sigma_clipped_stats(image, sigma=3.0)
    ax = axes[0]
    ax.imshow(image, cmap='gray', origin='lower', vmin=median, vmax=median + 3*std)
    ax.set_title("Original image")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axes[1]
    ax.set_facecolor('black')

    if flux_update == True:
        vm = 4.0
    else:
        vm = 4e-2
        
    ax.imshow(temp, origin='lower', vmax=vm)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Template")

    mean, median, std = sigma_clipped_stats(clean, sigma=3.0)
    ax = axes[2]
    ax.set_facecolor('white')
    ax.imshow(clean, cmap='gray', origin='lower', vmin=median, vmax=median + 3*std)
    ax.set_title("Clean image")
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    if name != None:
        plt.savefig(name, bbox_inches='tight')
    plt.show()

def displayRemoveSources(image, image_ghosts, name=None):
    """
    Display original image and image with sources removed side by side.

    Parameters
    ----------
    image : array-like
        The original image data.
    image_ghosts : array-like
        The image with sources removed.
    name : str, optional
        If provided, saves the figure to this filename.
    """
    mean, median, std = statsImage(image)
    fig = plt.figure(figsize=(16, 16))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray', origin='lower', vmin=median, vmax=median + 3*std)
    plt.title("Original image")
    plt.subplot(1, 2, 2)
    plt.imshow(image_ghosts, cmap='gray', origin='lower', vmin=median, vmax=median + 3*std)
    plt.title("Image without brightness sources (potential ghosts)")
    plt.tight_layout()
    if name != None:
        plt.savefig(name, bbox_inches='tight')
    plt.show()

def displayRemoveSourcesAndArtefacts(image, artefacts, name=None):
    """
    Display image without sources and artefacts side by side.

    Parameters
    ----------
    image : array-like
        The image with sources removed.
    artefacts : array-like
        The artefacts image data.
    name : str, optional
        If provided, saves the figure to this filename.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    mean, median, std = sigma_clipped_stats(image, sigma=3.0)
    axes[0].imshow(image, cmap='gray', origin='lower', vmin=median, vmax=median + 3*std)
    axes[0].set_title("Image without sources")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    mean, median, std = sigma_clipped_stats(artefacts, sigma=3.0)
    axes[1].imshow(artefacts, cmap='gray', origin='lower', vmin=median, vmax=median + 3*std)
    axes[1].set_title("Artefacts")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    plt.tight_layout()
    if name != None:
        plt.savefig(name, bbox_inches='tight')
    plt.show()

def displaySourcesAndGhosts(image, name=None):
    """
    Display images of brightness sources and potential ghosts side by side.

    Parameters
    ----------
    image : array-like
        The original image data.
    name : str, optional
        If provided, saves the figure to this filename.
    """
    mean, median, std = statsImage(image)
    image_source = extractSources(image)
    image_ghosts = removeSources(image)
    fig = plt.figure(figsize=(16, 16))
    plt.subplot(1, 2, 1)
    plt.imshow(image_source, cmap='gray', origin='lower', vmin=median, vmax=median + 3*std)
    plt.title("Image of brightness sources")
    plt.subplot(1, 2, 2)
    plt.imshow(image_ghosts, cmap='gray', origin='lower', vmin=median, vmax=median + 3*std)
    plt.title("Image of potential ghosts")
    plt.tight_layout()
    if name != None:
        plt.savefig(name, bbox_inches='tight')
    plt.show()

def displayCut(image, xpos, ypos, name=None):
    """
    Display vertical and horizontal cuts through the image at specified positions.

    Parameters
    ----------
    image : array-like
        The image data to display.
    xpos : int
        X position for vertical cut.
    ypos : int
        Y position for horizontal cut.
    name : str, optional
        If provided, saves the figure to this filename.
    """
    fig = plt.figure(figsize=(16, 6))

    plt.subplot(1, 2, 1)
    x = image[:, int(xpos)]
    x_idx = np.arange(len(x))
    plt.bar(x_idx, x, width=1.0)
    plt.xlabel("Pixels along y axis")
    plt.ylabel("Bin value")
    plt.ylim(-1000, 3000)
    plt.title(f"Cut at x = {xpos}")

    plt.subplot(1, 2, 2)
    y = image[int(ypos), :]
    y_idx = np.arange(len(y))
    plt.bar(y_idx, y, width=1.0)
    plt.xlabel("Pixels along x axis")
    plt.ylabel("Bin value")
    plt.ylim(-1000, 3000)
    plt.title(f"Cut at y = {ypos}")

    plt.tight_layout()
    
    if name is not None:
        plt.savefig(name, bbox_inches='tight')
    plt.show()

def displayCutTest(image, xpos, ypos, width=5, name=None):
    """
    Display mean vertical and horizontal cuts over a specified width around given positions.

    Parameters
    ----------
    image : array-like
        The image data to display.
    xpos : int
        X position for vertical cut.
    ypos : int
        Y position for horizontal cut.
    width : int, optional
        Number of pixels to average over for the cut.
    name : str, optional
        If provided, saves the figure to this filename.
    """
    fig = plt.figure(figsize=(16, 6))
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Coupe verticale (moyenne sur les colonnes autour de xpos)
    plt.subplot(1, 2, 1)
    x_start = max(0, int(xpos) - width)
    x_end = min(image.shape[1], int(xpos) + width + 1)
    x_profile = np.mean(image[:, x_start:x_end], axis=1)
    x_idx = np.arange(len(x_profile))
    plt.plot(x_idx, x_profile, color='blue')
    plt.xlabel("Pixels along y axis")
    plt.ylabel("Mean bin value")
    plt.ylim(np.min(x_profile)-50, np.max(x_profile)+50)
    plt.title(f"Mean cut over x ∈ [{x_start}, {x_end-1}]")

    # Coupe horizontale (moyenne sur les lignes autour de ypos)
    plt.subplot(1, 2, 2)
    y_start = max(0, int(ypos) - width)
    y_end = min(image.shape[0], int(ypos) + width + 1)
    y_profile = np.mean(image[y_start:y_end, :], axis=0)
    y_idx = np.arange(len(y_profile))
    plt.plot(y_idx, y_profile, color='green')
    plt.xlabel("Pixels along x axis")
    plt.ylabel("Mean bin value")
    plt.ylim(np.min(y_profile)-50, np.max(y_profile)+50)
    plt.title(f"Mean cut over y ∈ [{y_start}, {y_end-1}]")

    plt.tight_layout()
    
    if name is not None:
        plt.savefig(name, bbox_inches='tight')
    plt.show()

def displayFitTest(image, temp, clean, flux_update=False, name=None):
    """
    Display original, template, and cleaned images side by side with colorbars.

    Parameters
    ----------
    image : array-like
        The original image data.
    temp : array-like
        The template image data.
    clean : array-like
        The cleaned image data.
    flux_update : bool, optional
        Whether to use high flux scaling for the template.
    name : str, optional
        If provided, saves the figure to this filename.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Première image : image originale
    mean, median, std = sigma_clipped_stats(image, sigma=3.0)
    ax = axes[0]
    im0 = ax.imshow(image, cmap='gray', origin='lower', vmin=median, vmax=median + 3*std)
    ax.set_title("Original image")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im0, ax=ax, fraction=0.046, pad=0.04)

    # Deuxième image : template
    ax = axes[1]
    ax.set_facecolor('black')
    vm = 4.0 if flux_update else 4e-2
    im1 = ax.imshow(temp, origin='lower', vmax=vm)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Template")
    fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)

    # Troisième image : image nettoyée
    mean, median, std = sigma_clipped_stats(clean, sigma=3.0)
    ax = axes[2]
    ax.set_facecolor('white')
    im2 = ax.imshow(clean, cmap='gray', origin='lower', vmin=median, vmax=median + 3*std)
    ax.set_title("Clean image")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    if name is not None:
        plt.savefig(name, bbox_inches='tight')
    plt.show()

def displayRealTest(image, name=None):
    """
    Display the original image with grayscale colormap, sigma-clipped scaling, and colorbar.

    Parameters
    ----------
    image : array-like
        The image data to display.
    name : str, optional
        If provided, saves the figure to this filename.
    """
    mean, median, std = sigma_clipped_stats(image, sigma=3.0)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    
    im = ax.imshow(image, cmap='gray', origin='lower', vmin=median, vmax=median + 3*std)
    ax.set_title("Original image")
    ax.set_xticks([])
    ax.set_yticks([])

    # Ajout de la colorbar
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if name is not None:
        plt.savefig(name, bbox_inches='tight')
    plt.show()
