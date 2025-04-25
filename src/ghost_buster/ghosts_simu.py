import pylab as plt
import numpy as np
import batoid
from scipy.ndimage import gaussian_filter
from .sources_image import getCoordBatoid

version = "0.1"

def rotBeforeBatoid(data, theta):
    '''

    Parameters
    ----------
    data : list
        initial data for Batoid's calculus
    theta : float
        angle of the telescope with sky

    Returns
    -------
    data : list
        initial data for Batoid's calculus

    '''
    x_prime, y_prime = [], []
    
    for point in data:
        x = point[0]
        y = point[1]
        x_rot = x * np.cos(theta) - y * np.sin(theta)
        y_rot = x * np.sin(theta) + y * np.cos(theta)
        x_prime.append(x_rot)
        y_prime.append(y_rot)
        
    for i in range(len(data)):
        data[i] = (x_prime[i], y_prime[i], data[i][2], data[i][3], data[i][4], data[i][5])
        
    return data

def rotAfterBatoid(x, y, metadata):
    '''

    Parameters
    ----------
    x : np.array
        x coordinates of rays in Batoid referentiel
    y : np.array
        y coordinates of rays in Batoid referentiel
    metadata : dict
        meta-data of the image

    Returns
    -------
    x_prime : np.array
        x coordinates of rays in Telescope referentiel
    y_prime : np.array
        y coordinates of rays in Telescope referentiel

    '''
    theta = np.deg2rad(-metadata["ROTPA"]) # Changer angle
    x_prime = x*np.cos(theta) - y*np.sin(theta)
    y_prime = x*np.sin(theta) + y*np.cos(theta)
    return x_prime, y_prime

def initTelescope():
    '''

    Returns
    -------
    telescope : batoid.Optic
        Construct telescope for Batoid's simulation

    '''
    telescope = batoid.Optic.fromYaml("ComCamSpiders_r.yaml")

    for surface in telescope.itemDict.values():
            if isinstance(surface, batoid.RefractiveInterface):
                if surface.name.split('_')[0] in ['L1']:
                    surface.forwardCoating = batoid.SimpleCoating(1-0.9525945248652419, 0.9525945248652419)
                    surface.reverseCoating = batoid.SimpleCoating(1-0.9525945248652419, 0.9525945248652419)
                elif surface.name.split('_')[0] in ['L2']:
                    surface.forwardCoating = batoid.SimpleCoating(1-0.9486521132432222, 0.9486521132432222)
                    surface.reverseCoating = batoid.SimpleCoating(1-0.9486521132432222, 0.9486521132432222)
                elif surface.name.split('_')[0] in ['L3']:
                    surface.forwardCoating = batoid.SimpleCoating(1-0.9535605093540551, 0.9535605093540551)
                    surface.reverseCoating = batoid.SimpleCoating(1-0.9535605093540551, 0.9535605093540551)
                elif surface.name.split('_')[0] in ['Filter']:
                    surface.forwardCoating = batoid.SimpleCoating(1-0.9529014593685005, 0.9529014593685005)
                    surface.reverseCoating = batoid.SimpleCoating(1-0.9529014593685005, 0.9529014593685005)
            if isinstance(surface, batoid.Detector):
                surface.forwardCoating = batoid.SimpleCoating(1-0.9018076800000383, 0.9018076800000383)
    
    return telescope

def initParams(image, metadata, bins=8, nrad=300, naz=900, maxflux=1.0, minflux=1e-4):
    '''

    Parameters
    ----------
    image : np.array
        image.fits.getArray()
        Bin's values
    metadata : dict
        meta-data of the image
    nrad : int, optional
        number of circle for asPolar simulation 
        The default is 300.
    naz : int, optional
        number of point on each circle (see naz)
        The default is 900.
    maxflux : float, optional
        initial flux value
        The default is 1.0.
    minflux : float, optional
        minimum final flux value
        The default is 1e-5.

    Returns
    -------
    init_simu : list
        initial data for Batoid's calculus

    '''
    theta_x, theta_y = getCoordBatoid(image, bins=bins)
    init_simu = [(theta_x, theta_y, nrad, naz, maxflux, minflux)]
    theta = np.deg2rad(metadata["ROTPA"]) # Changer angle
    init_simu = rotBeforeBatoid(init_simu, theta)
    return init_simu

def batoidCalcul(telescope, init_simu, debug=False):
    '''

    Parameters
    ----------
    telescope : Batoid.Optic
        telescope use for Batoid's simulation
    init_simu : list
        initial data for Batoid's calculus
    debug : bool, optional
        print more informations for debug
        The default is False.

    Returns
    -------
    x : np.array
        x coordinates of rays in Batoid's referentiel
    y : np.array
        y coordinates of rays in Batoid's referentiel
    flux : np.array
        flux values of each rays

    '''
    x, y, flux = [], [], []
    for dat in init_simu:
    
        rays = batoid.RayVector.asPolar(
            optic=telescope, wavelength=630e-9,
            theta_x=np.deg2rad(dat[0]), theta_y=np.deg2rad(dat[1]),
            nrad=dat[2], naz=dat[3], flux=dat[4]
        )
    
        rForward, rReverse = telescope.traceSplit(rays, minFlux=dat[5], _verbose=False) # _verbose = log calculus
    
        for i, rr in enumerate(rForward):
            x.append([ix for ix in rr.x])
            y.append([iy for iy in rr.y])
            flux.append([iflux for iflux in rr.flux])
    
    if debug:
        print("# input rays          = {}".format(len(rays)))
        print("# forward output rays = {}".format(sum(len(rr) for rr in rForward)))
        print("# reverse output rays = {}".format(sum(len(rr) for rr in rReverse)))
        print("input flux          = {}".format(np.sum(rays.flux)))
        forwardFlux = np.sum([np.sum(rr.flux) for rr in rForward])
        reverseFlux = np.sum([np.sum(rr.flux) for rr in rReverse])
        print("forward output flux = {}".format(forwardFlux))
        print("reverse output flux = {}".format(reverseFlux))
        print("destroyed flux      = {}".format(
            np.sum(rays.flux) - forwardFlux - reverseFlux
        ))

    return x, y, flux

def groupData(x, y, flux):
    '''

    Parameters
    ----------
    x : np.array
        x coordinates of rays with a dimension for each ghost
    y : np.array
        y coordinates of rays with a dimension for each ghost
    flux : np.array
        flux of rays with a dimension for each ghost

    Returns
    -------
    x : np.array
        x coordinates of rays
    y : np.array
        y coordinates of rays
    flux : np.array
        flux of rays

    '''
    x = np.concatenate([ix for ix in x])
    y = np.concatenate([iy for iy in y])
    flux = np.concatenate([iflux for iflux in flux])
    return x, y, flux

def getSimuImage(px, py, x, y, flux, binning):
    '''
    
    Parameters
    ----------
    px : int
        number of pixels on x axe of the image
    py : int
        number of pixels on y axe of the image
    x : np.array
        x coordinates of rays with a dimension for each ghost
    y : np.array
        y coordinates of rays with a dimension for each ghost
    flux : np.array
        flux of rays with a dimension for each ghost

    Returns
    -------
    fig : plt.fig
        subplot of the both images

    '''
    # On suppose que x_prime, y_prime et flux sont des arrays 1D avec les positions et le flux associé

    scale = binning*1e-5
    X, Y = px*scale, py*scale
    
    x_min, x_max = -X/2.0, X/2.0
    y_min, y_max = -Y/2.0, Y/2.0
    
    mask = (x > x_min) & (x < x_max) & (y > y_min) & (y < y_max)
    
    x = x[mask]
    y = y[mask]
    flux = flux[mask]
    
    H, xedges, yedges = np.histogram2d(x, y, bins=[px, py], weights=flux)

    return H.T, xedges, yedges

def getSmoothSimu(simu, sigma=1.0):
    simu = np.nan_to_num(simu, nan=0.0, posinf=0.0, neginf=0.0)
    smoothed = gaussian_filter(simu, sigma=sigma)
    return smoothed

def getNoiseSimu(image, mu, sigma):
    # Taille de l'image simulée
    nx, ny = image.shape[1], image.shape[0]
    
    # Génération d'une "image" de bruit 2D
    image_bruit = np.random.normal(loc=mu, scale=sigma, size=(nx, ny))
    
    return image_bruit.T

def showNoiseSimu(image, mu, sigma):
    image_bruit = getNoiseSimu(image=image, mu=mu, sigma=sigma)

    plt.imshow(image_bruit, origin='lower', cmap='viridis', aspect='auto')
    plt.colorbar(label='Valeur du pixel')
    plt.title("Image simulée - Bruit gaussien")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
