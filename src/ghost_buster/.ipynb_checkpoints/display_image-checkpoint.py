import pylab as plt
import lsst.afw.image as afwimage
import lsst.afw.display as afwDisplay

def displayImage(image,title=None):
    afwDisplay.setDefaultBackend('matplotlib') 
    fig = plt.figure(figsize=(10,10))
    afw_display = afwDisplay.Display(1)
    afw_display.scale('asinh', 'zscale')
    #afw_display.scale('linear', min=-5, max=10)
    afw_display.setImageColormap(cmap='plasma')
    afw_display.mtv(image)
    plt.title(title)
    plt.gca().axis('off')
    plt.show()
    return afw_display
    
def displayImageGhosts(image, zmin=0, zmax=5000, title=None):
    afwDisplay.setDefaultBackend('matplotlib') 
    fig = plt.figure(figsize=(10,10))
    afw_display = afwDisplay.Display(1)
    afw_display.scale('linear', min=zmin, max=zmax)
    afw_display.setImageColormap(cmap='plasma')
    afw_display.mtv(image)
    plt.title(title)
    plt.gca().axis('off')
    plt.show()
    return afw_display

def displayImageGhostsBW(image, title=None):
    afwDisplay.setDefaultBackend('matplotlib') 
    fig = plt.figure(figsize=(10,10))
    afw_display = afwDisplay.Display(1)
    afw_display.scale('asinh', 'zscale')
    afw_display.setImageColormap(cmap='grey')
    afw_display.mtv(ghost_292)
    plt.gca().axis('off')
    plt.show()
    return afw_display

#ghost_292=afwimage.ImageF.readFits("ghost_292.fits", 0)
#displayImageGhosts(ghost_292, zmin=850, zmax=1100, title="2024111100292")
#displayImageGhostsBW(ghost_292, title="2024111100292")

def displayImageGhostsFlux(image, minflux=955, maxflux=990):
    flux_mask = (image >= minflux) & (image <= maxflux)
    pixels_in_range = data_ghost[flux_mask]
    masked_data = np.where(flux_mask, image, 0.0)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    im = ax.imshow(masked_data, origin='lower', vmin=minflux, vmax=maxflux)
    fig.colorbar(im, ax=ax, label='Flux (dans le range)')
    ax.set_title(f"Pixels in {minflux} <= flux <= {maxflux}")
    plt.show()
    return fig

def displayReal(image):
    mean, median, std = sigma_clipped_stats(image, sigma=3.0)
    
    fig = plt.imshow(image, cmap='gray', origin='lower', vmin=median, vmax=median + 3*std)
    plt.title("Image originale")
    plt.xticks([])
    plt.yticks([])
    plt.show()
    return fig

def displaySimu(hist, x, y):
    CCD_DX = 43.333/1000.

    x_min, x_max, dx = -CCD_DX*1.5, CCD_DX*1.5, CCD_DX
    y_min, y_max, dy = -CCD_DX*1.5, CCD_DX*1.5, CCD_DX
    
    x_grid = np.arange(x_min, x_max + dx, dx)
    y_grid = np.arange(y_min, y_max + dy, dy)

    X, Y = np.meshgrid(x, y)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    im = ax.pcolormesh(X, Y, hist, vmax=4e-2, norm='log')
    
    for xi in x_grid:
        ax.plot([xi, xi], [y_min, y_max], c="r")
    for yi in y_grid:
        ax.plot([x_min, x_max], [yi, yi], c="r")
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_facecolor('black')
    plt.xticks([])
    plt.yticks([])
    #plt.savefig('simu.png', bbox_inches='tight')
    #plt.close()
    plt.show()
    return fig

def displayFit(image, hist, x , y, clean):
    mean, median, std = sigma_clipped_stats(image, sigma=3.0)

    plt.subplot(1, 3, 1)
    plt.imshow(data_ghost, cmap='gray', origin='lower', vmin=median, vmax=median + 3*std)
    plt.title("Image originale")
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(1, 3, 2, facecolor='black')
    CCD_DX = 43.333/1000.
    x_min, x_max, dx = -CCD_DX*1.5, CCD_DX*1.5, CCD_DX
    y_min, y_max, dy = -CCD_DX*1.5, CCD_DX*1.5, CCD_DX
    x_grid = np.arange(x_min, x_max + dx, dx)
    y_grid = np.arange(y_min, y_max + dy, dy)
    X, Y = np.meshgrid(x, y)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.pcolormesh(X, Y, hist, vmax=4e-2, norm='log')
    
    for xi in x_grid:
        ax.plot([xi, xi], [y_min, y_max], c="r")
    for yi in y_grid:
        ax.plot([x_min, x_max], [yi, yi], c="r")
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_facecolor('black')
    plt.xticks([])
    plt.yticks([])
    
    mean, median, std = sigma_clipped_stats(clean, sigma=3.0)
    
    plt.subplot(1, 3, 3)
    plt.imshow(clean, cmap='gray', origin='lower', vmin=median, vmax=median + 3*std)
    plt.title("Image nettoyée")
    plt.xticks([])
    plt.yticks([])
    #plt.savefig('real.png', bbox_inches='tight')
    #plt.close()
    plt.show()
