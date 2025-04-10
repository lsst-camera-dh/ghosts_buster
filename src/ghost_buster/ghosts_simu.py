import pyplot as plt
import numpy as np
import batoid
from sources_image import getCoordBatoid

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
    theta = np.radians(-metadata["ROTPA"])
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
                if surface.name.split('_')[0] in ['L1', 'L2', 'L3']:
                    surface.forwardCoating = batoid.SimpleCoating(0.05, 0.95)
                    surface.reverseCoating = batoid.SimpleCoating(0.05, 0.95)
                elif surface.name.split('_')[0] in ['Filter']:
                    surface.forwardCoating = batoid.SimpleCoating(0.05, 0.95)
                    surface.reverseCoating = batoid.SimpleCoating(0.05, 0.95)
            if isinstance(surface, batoid.Detector):
                surface.forwardCoating = batoid.SimpleCoating(0.15, 0.85)
    
    return telescope

def initParams(image, metadata, nrad=300, naz=900, maxflux=1.0, minflux=1e-5):
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
    theta_x, theta_y = getCoordBatoid(image)
    init_simu = [(theta_x, theta_y, nrad, naz, maxflux, minflux)]
    theta = np.radians(metadata["ROTPA"])
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

def getSimuImage(px, py, x, y, flux):
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
    
    CCD_DX = 43.333/1000.

    x_min, x_max, dx = -CCD_DX*1.5, CCD_DX*1.5, CCD_DX
    y_min, y_max, dy = -CCD_DX*1.5, CCD_DX*1.5, CCD_DX
    
    x_grid = np.arange(x_min, x_max + dx, dx)
    y_grid = np.arange(y_min, y_max + dy, dy)
    
    
    mask = (x > x_min) & (x < x_max) & (y > y_min) & (y < y_max)
    
    x = x[mask]
    y = y[mask]
    flux = flux[mask]
    
    H, xedges, yedges = np.histogram2d(x, y, bins=[px, py], weights=flux)

    return H.T, xedges, yedges
    
    fig, ax = plt.subplots(figsize=(8, 8))
    X, Y = np.meshgrid(xedges, yedges)
    ax.pcolormesh(X, Y, H.T, vmax=4e-2, norm='log')
    
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
