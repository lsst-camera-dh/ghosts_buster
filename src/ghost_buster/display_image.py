import pylab as plt
import numpy as np
import lsst.afw.display as afwDisplay
from astropy.stats import sigma_clipped_stats
from .sources_image import statsImage, extractSources, removeSources

version = "0.1"

def displayImage(image, title=None, name=None):
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
    mean, median, std = sigma_clipped_stats(image, sigma=3.0)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    im = ax.imshow(image, cmap='gray', origin='lower', vmin=median, vmax=median + 3*std)
    plt.title("Original image")

    if name != None:
        plt.savefig(name, bbox_inches='tight')
    plt.show()

def displaySub(image, name=None, starpos=None):
    mean, median, std = sigma_clipped_stats(image, sigma=3.0)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    im = ax.imshow(image, cmap='gray', origin='lower', vmin=median, vmax=median + 3*std)

    if starpos != None:
        ax.plot(starpos[0]-185, starpos[1]-140, '+', c='r')
    plt.title("Subimage")

    if name != None:
        plt.savefig(name, bbox_inches='tight')
    plt.show()

def displaySimu(hist, x, y, name=None):
    
    X, Y = np.meshgrid(x, y)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    im = ax.pcolormesh(X, Y, hist, vmax=4e-2)
    ax.set_aspect("equal")
    ax.set_facecolor('black')
    plt.xticks([])
    plt.yticks([])
    
    if name != None:
        plt.savefig(name, bbox_inches='tight')
    plt.show()

def displayClean(image, name=None):
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

def displayFit(image, hist, x, y, clean, name=None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    mean, median, std = sigma_clipped_stats(image, sigma=3.0)
    ax = axes[0]
    ax.imshow(image, cmap='gray', origin='lower', vmin=median, vmax=median + 3*std)
    ax.set_title("Original image")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axes[1]
    ax.set_facecolor('black')

    X, Y = np.meshgrid(x, y)

    pcm = ax.pcolormesh(X, Y, hist, vmax=4e-2, shading='auto', norm='log')
    
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Histogram 2D")

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

def displayFit2(image, temp, clean, name=None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    mean, median, std = sigma_clipped_stats(image, sigma=3.0)
    ax = axes[0]
    ax.imshow(image, cmap='gray', origin='lower', vmin=median, vmax=median + 3*std)
    ax.set_title("Original image")
    ax.set_xticks([])
    ax.set_yticks([])

    mean, median, std = sigma_clipped_stats(temp, sigma=3.0)
    ax = axes[1]
    ax.set_facecolor('black')
    ax.imshow(temp, cmap='gray', origin='lower', vmin=median, vmax=median + 3*std)
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

def displayRemoveSourcesBoth(image, image_ghosts, name=None):
    mean, median, std = sigma_clipped_stats(image, sigma=3.0)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(image, cmap='gray', origin='lower', vmin=median, vmax=median + 3*std)
    axes[0].set_title("Original image")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[1].imshow(image_ghosts, origin='lower', vmax=4e-2, norm='log')
    axes[1].set_title("Image without brightness sources (potential ghosts)")
    axes[1].set_facecolor('black')
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    plt.tight_layout()
    if name != None:
        plt.savefig(name, bbox_inches='tight')
    plt.show()

def displaySourcesAndGhosts(image, name=None):
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
