import pylab as plt
import numpy as np
import batoid
import math
from scipy.ndimage import gaussian_filter
from scipy.special import j1
from scipy.ndimage import rotate
from scipy.signal import fftconvolve
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

def rotAfterBatoid(x, y, rot):
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
    theta = np.deg2rad(-rot) # Changer angle
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
    '''
    for surface in telescope.itemDict.values():
            if isinstance(surface, batoid.RefractiveInterface):
                if surface.name.split('_')[0] in ['L1']:
                    surface.forwardCoating = batoid.SimpleCoating(1-0.9519450133010235, 0.9519450133010235)
                    surface.reverseCoating = batoid.SimpleCoating(1-0.95194501330102359, 0.9519450133010235)
                elif surface.name.split('_')[0] in ['L2']:
                    surface.forwardCoating = batoid.SimpleCoating(1-0.9510109078711367, 0.9510109078711367)
                    surface.reverseCoating = batoid.SimpleCoating(1-0.9510109078711367, 0.9510109078711367)
                elif surface.name.split('_')[0] in ['L3']:
                    surface.forwardCoating = batoid.SimpleCoating(1-0.9528543599768664, 0.9528543599768664)
                    surface.reverseCoating = batoid.SimpleCoating(1-0.9528543599768664, 0.9528543599768664)
                elif surface.name.split('_')[0] in ['Filter']:
                    surface.forwardCoating = batoid.SimpleCoating(1-0.9441339435556122, 0.9441339435556122)
                    surface.reverseCoating = batoid.SimpleCoating(1-0.9441339435556122, 0.9441339435556122)
            if isinstance(surface, batoid.Detector):
                surface.forwardCoating = batoid.SimpleCoating(1-0.9213684000000424, 0.9213684000000424)
    ''' # Pour 666nm au haut et 622.17nm en bas
    
    for surface in telescope.itemDict.values():
            if isinstance(surface, batoid.RefractiveInterface):
                if surface.name.split('_')[0] in ['L1']:
                    surface.forwardCoating = batoid.SimpleCoating(1-0.9516768353277874, 0.9516768353277874)
                    surface.reverseCoating = batoid.SimpleCoating(1-0.9516768353277874, 0.9516768353277874)
                elif surface.name.split('_')[0] in ['L2']:
                    surface.forwardCoating = batoid.SimpleCoating(1-0.9469600169240043, 0.9469600169240043)
                    surface.reverseCoating = batoid.SimpleCoating(1-0.9469600169240043, 0.9469600169240043)
                elif surface.name.split('_')[0] in ['L3']:
                    surface.forwardCoating = batoid.SimpleCoating(1-0.9531593356044364, 0.9531593356044364)
                    surface.reverseCoating = batoid.SimpleCoating(1-0.9531593356044364, 0.9531593356044364)
                elif surface.name.split('_')[0] in ['Filter']:
                    surface.forwardCoating = batoid.SimpleCoating(1-0.9542022542740184, 0.9542022542740184)
                    surface.reverseCoating = batoid.SimpleCoating(1-0.9542022542740184, 0.9542022542740184)
            if isinstance(surface, batoid.Detector):
                surface.forwardCoating = batoid.SimpleCoating(1-0.8973497700000442, 0.8973497700000442)
    
    return telescope

def initParams(image, rot, bins=8, nrad=300, naz=900, maxflux=1.0, minflux=1e-4, pos=None, thetapos=None):
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
    if thetapos != None:
        theta_x = thetapos[0]
        theta_y = thetapos[1]
        
    elif pos != None:
        ny, nx = image.shape
        x0, y0 = nx / 2, ny / 2
        scale = 0.2
        dx = (pos[0] - x0) * scale * bins
        dy = (pos[1] - y0) * scale * bins
        theta_x = dx / 3600
        theta_y = dy / 3600   
    else:
        theta_x, theta_y = getCoordBatoid(image, bins=bins)
        
    init_simu = [(theta_x, theta_y, nrad, naz, maxflux, minflux)]
    theta = np.deg2rad(rot) # Changer angle
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
    x, y, flux, paths = [], [], [], []
    
    for dat in init_simu:
    
        rays = batoid.RayVector.asPolar(
            optic=telescope, wavelength=622.2e-9,
            theta_x=np.deg2rad(dat[0]), theta_y=np.deg2rad(dat[1]),
            nrad=dat[2], naz=dat[3], flux=dat[4]
        )
    
        rForward, rReverse = telescope.traceSplit(rays, minFlux=dat[5], _verbose=False) # _verbose = log calculus
    
        for i, rr in enumerate(rForward):
            x.append([ix for ix in rr.x])
            y.append([iy for iy in rr.y])
            flux.append([iflux for iflux in rr.flux])
            paths.append(rr.path)
            
    
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

    return x, y, flux, paths

def batoidCalcul2(init_simu):
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
    # wavelenght = np.array([540.0e-9,580.0e-9,620.0e-9,660.0e-9,700.0e-9])
    
    # L1 = np.array([0.9473656178595908, 0.9487472881118438, 0.9515581750487883, 0.9521221952988791, 0.9495472617064802])
    # L2 = np.array([0.938148973589068, 0.9401417728987096, 0.9466205595400999, 0.9507516177855074, 0.9503512461572875])
    # L3 = np.array([0.9566516937772196, 0.9544130930928716, 0.9531789255651926, 0.9528765064914722, 0.9524670910461178])
    # Filter = np.array([0.035443222206936145, 0.4975571325762805, 0.5314882302960491, 0.5670596511840295, 0.04828171062891763])
    # Detector = np.array([0.862944000000011, 0.8753575200000265, 0.8961033600000373, 0.9184341600000414, 0.9347832000000273])
    
    # factor_flux = np.array([1.826506e-13, 1.97499e-13, 1.911044e-13, 1.919171e-13, 1.80519e-13])
    # factor_flux = factor_flux / np.max(factor_flux)

    wavelenght = np.array([634.0e-9])
    
    L1 = np.array([0.9523404062480814])
    L2 = np.array([0.9487676941242609])
    L3 = np.array([0.9532206045692587])
    Filter = np.array([0.9454090388657778])
    Detector = np.array([0.9041446800000388])
    
    factor_flux = np.array([1.974203e-13])
    factor_flux = factor_flux / np.max(factor_flux)
    
    x, y, flux = [], [], []
    
    for i in range(len(wavelenght)):

        telescope = batoid.Optic.fromYaml("ComCamSpiders_r.yaml")
        
        for surface in telescope.itemDict.values():
            if isinstance(surface, batoid.RefractiveInterface):
                if surface.name.split('_')[0] in ['L1']:
                    surface.forwardCoating = batoid.SimpleCoating(1-L1[i], L1[i])
                    surface.reverseCoating = batoid.SimpleCoating(1-L1[i], L1[i])
                elif surface.name.split('_')[0] in ['L2']:
                    surface.forwardCoating = batoid.SimpleCoating(1-L2[i], L2[i])
                    surface.reverseCoating = batoid.SimpleCoating(1-L2[i], L2[i])
                elif surface.name.split('_')[0] in ['L3']:
                    surface.forwardCoating = batoid.SimpleCoating(1-L3[i], L3[i])
                    surface.reverseCoating = batoid.SimpleCoating(1-L3[i], L3[i])
                elif surface.name.split('_')[0] in ['Filter']:
                    surface.forwardCoating = batoid.SimpleCoating(1-Filter[i], Filter[i])
                    surface.reverseCoating = batoid.SimpleCoating(1-Filter[i], Filter[i])
            if isinstance(surface, batoid.Detector):
                surface.forwardCoating = batoid.SimpleCoating(1-Detector[i], Detector[i])
                
        for dat in init_simu:
        
            rays = batoid.RayVector.asPolar(
                optic=telescope, wavelength=wavelenght[i],
                theta_x=np.deg2rad(dat[0]), theta_y=np.deg2rad(dat[1]),
                nrad=dat[2], naz=dat[3], flux=factor_flux[i]
            )
        
            rForward, rReverse = telescope.traceSplit(rays, minFlux=dat[5], _verbose=False) # _verbose = log calculus
        
            for i, rr in enumerate(rForward):
                x.append([ix for ix in rr.x])
                y.append([iy for iy in rr.y])
                flux.append([iflux for iflux in rr.flux])

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

    extent = [-X/2.0, X/2.0, -Y/2.0, Y/2.0]
    x_min, x_max, y_min, y_max = extent
    
    mask = (x > x_min) & (x < x_max) & (y > y_min) & (y < y_max)
    
    x = x[mask]
    y = y[mask]
    flux = flux[mask]
    
    H, xedges, yedges = np.histogram2d(x, y, bins=[px, py], weights=flux, range=[[x_min, x_max], [y_min, y_max]])

    return H.T, xedges, yedges

def getGhosts(telescope, init_simu):
    xsep, ysep, flux, path = batoidCalcul(telescope, init_simu)
    nghost = len(xsep)
    ref = ['L1_entrance', 'L1_exit', 'L2_entrance', 'L2_exit', 'Filter_entrance', 'Filter_exit', 'L3_entrance', 'L3_exit', 'Detector']
    paths = []

    for i in range(nghost):
        paths.append([ipath for ipath in path[i] if ipath in ref])
        
    

    if nghost == 0:
        return [], [], []
    
    idx_keep = []
    flux_update = []
    
    fig, axes = plt.subplots(nghost, 1, figsize=(4 * nghost, 4), constrained_layout=True)
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    hexbin_collection = [None] * nghost
    
    for i in range(nghost):
        hexbin_collection[i] = axes[i].hexbin(
            xsep[i], ysep[i], extent=[-0.25, 0.25, -0.25, 0.25],
            gridsize=400
        )    
        offsets = hexbin_collection[i].get_offsets()
        counts = hexbin_collection[i].get_array()
        
        hex_size = 0.5 / 400
        hex_area = (3 * np.sqrt(3) / 2) * (hex_size ** 2)
    
        mask = counts > 0
        individual_areas = [hex_area for _ in offsets[mask]]
        flux_update.append(flux[i][0]/(len(individual_areas)*hex_area))
        # print(f"Flux/m2 for ghost {i+1} : {flux_update}")

        if flux_update[i] > 1e-1:
            idx_keep.append(i)
    
    plt.close()

    path = paths.copy()
    x, y, f, paths = [], [], [], []
    
    for i in idx_keep:
        if flux_update[i]==np.max(flux_update):
            continue
            
        x.append([ix for ix in xsep[i]])
        y.append([iy for iy in ysep[i]])
        f.append([iflux for iflux in flux[i]])
        paths.append([ipath for ipath in path[i]])

    def get_subplot_grid(num_subplots):
        n_cols = math.ceil(math.sqrt(num_subplots))
        n_rows = math.ceil(num_subplots / n_cols)
        return n_rows, n_cols

    nrows, ncols = get_subplot_grid(len(idx_keep))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), constrained_layout=True)

    if nrows == 1 or ncols == 1:
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    else:
        axes = np.array(axes).flatten()
    
    hexbin_collection = [None] * len(idx_keep)
    
    for i in range(len(idx_keep)):
        xi, yi = rotAfterBatoid(np.array(xsep[idx_keep[i]]), np.array(ysep[idx_keep[i]]), 77.89209882814671)
        hexbin_collection[i] = axes[i].hexbin(
            xi, yi, extent=[-0.25, 0.25, -0.25, 0.25],
            gridsize=400
        )
        th = np.linspace(0, 2*np.pi, 1000)
        axes[i].plot(0.32/5*np.cos(th), 0.32/5*np.sin(th), c='r', label="Cercle optique")
    
        # w = np.argmin([len(ipath) for ipath in paths])
        # axes[i].plot(np.mean(x[w]), np.mean(y[w]), marker='+', color='m')
    
        offsets = hexbin_collection[i].get_offsets()
        counts = hexbin_collection[i].get_array()
        
        hex_size = 0.5 / 400
        hex_area = (3 * np.sqrt(3) / 2) * (hex_size ** 2)
    
        mask = counts > 0
        individual_areas = [hex_area for _ in offsets[mask]]
        
        axes[i].set_xlabel(f"{np.array(path[idx_keep[i]])}", fontsize=8, color="gray", labelpad=5)
        axes[i].set_title(f"Ghost {i+1}, Flux : {flux[idx_keep[i]][0]/(len(individual_areas)*hex_area):.5f}", fontsize=10)
        axes[i].axis("equal")
        
    for i in range(len(idx_keep), len(axes)):
        axes[i].axis("off")
    
    # Add a description under the figure
    plt.savefig('all_ghosts.png', bbox_inches='tight')
    plt.show()
    
    x, y, f = groupData(x, y, f)

    return x, y, f
        
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

def getAiry(xstar, ystar, rot, D=8.36, f=9.8648):
    binning = 8.0
    wavelength = 666e-9  # m
    
    theta = 1.20 * wavelength / D
    rayon_image_m = f * theta
    
    Ny, Nx = 1577, 1586
    pixel_scale_arcsec = 0.2*binning
    pixel_scale_rad = pixel_scale_arcsec * np.pi / (180 * 3600)
    pixel_scale_m = f * pixel_scale_rad
    
    FOV_x = Nx * pixel_scale_m
    FOV_y = Ny * pixel_scale_m
    
    x = np.linspace(-FOV_x / 2, FOV_x / 2, Nx)
    y = np.linspace(-FOV_y / 2, FOV_y / 2, Ny)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    
    k = 2 * np.pi / wavelength
    alpha = k * D * R / (2 * f)
    airy = (2 * j1(alpha) / alpha)**2
    airy = airy/airy.max()
    airy_rotated = rotate(airy, angle=rot, reshape=False, order=3)
    
    def fourier_shift(image, dx, dy):
        ny, nx = image.shape
        fx = np.fft.fftfreq(nx)
        fy = np.fft.fftfreq(ny)
        FX, FY = np.meshgrid(fx, fy)
        shift_phase = np.exp(-2j * np.pi * (FX * dx + FY * dy))
        f_image = np.fft.fft2(image)
        f_shifted = f_image * shift_phase
        return np.abs(np.fft.ifft2(f_shifted))

    shift_x = xstar - Nx / 2.0
    shift_y = ystar - Ny / 2.0
    
    image = fourier_shift(airy_rotated, shift_x, shift_y) # airy_rotated > airy

    theta_x = shift_x * pixel_scale_arcsec / 3600.0
    theta_y = shift_y * pixel_scale_arcsec / 3600.0
    psf_norm = psfNorm(theta_x, theta_y)
    
    image_psf = fftconvolve(image, psf_norm, mode='same') # image > airy
    image_psf = image_psf/image_psf.max()
    
    return image_psf # image_psf > psf_norm

def psfNorm(theta_x, theta_y):
    target_Ny, target_Nx = 1577, 1586
    nx = max(target_Ny, target_Nx)

    telescope = initTelescope()
    wavelength = 666e-9

    fftpsf = batoid.fftPSF(
        telescope,
        np.deg2rad(theta_x), np.deg2rad(theta_y),
        wavelength,
        nx=nx,
        pad_factor=1
    )

    psf = fftpsf.array
    psf = psf / psf.sum()

    # Rognage central (crop) si trop grand
    start_y = (psf.shape[0] - target_Ny) // 2
    start_x = (psf.shape[1] - target_Nx) // 2
    psf_cropped = psf[start_y:start_y+target_Ny, start_x:start_x+target_Nx]
    psf_cropped = psf_cropped / psf_cropped.sum()

    return psf_cropped
