import pylab as plt
import numpy as np
import pandas as pd
import batoid
import math
import os
from scipy.special import j1
from scipy.ndimage import rotate
from scipy.signal import fftconvolve

version = "0.2"

def pixeltotheta(dimension, starpos, binning):
    '''

    Parameters
    ----------
    dimension : tuple
        data.shape for the changement of coordinates.
    starpos : tuple
        (xstar, ystar), position of the star.
    binning : int
        Binng of the data.

    Returns
    -------
    theta_x : float
        theta_x value for Batoid.
    theta_y : float
        theta_y value for Batoid.

    '''
    ny, nx = dimension
    x0, y0 = nx / 2, ny / 2
    scale = 0.2
    dx = (starpos[0] - x0) * scale * binning
    dy = (starpos[1] - y0) * scale * binning
    theta_x = dx / 3600
    theta_y = dy / 3600
    return theta_x, theta_y

def thetatopixel(dimension, thetapos, binning):
    '''

    Parameters
    ----------
    dimension : tuple
        data.shape for the changement of coordinates.
    thetapos : tuple
        (theta_x, theta_y), position of the star in Batoid's coordinates.
    binning : int
        Binng of the data.

    Returns
    -------
    x_star : float
        x position in the data (in pixel).
    y_star : float
        y position in the data (in pixel).

    '''
    ny, nx = dimension
    x0, y0 = nx / 2, ny / 2
    scale = 0.2
    dx = thetapos[0] * 3600 
    dy = thetapos[1] * 3600
    xstar = x0 + dx / scale / binning
    ystar = y0 + dy / scale / binning
    return xstar, ystar

def rotBeforeBatoid(data, rot):
    '''

    Parameters
    ----------
    data : list
        initial data for Batoid's calculus
    rot : float
        angle of the telescope with sky in degrees

    Returns
    -------
    data : list
        initial data for Batoid's calculus

    '''
    x_prime, y_prime = [], []
    theta = np.deg2rad(rot)
    
    for point in data:
        x = point[0]
        y = point[1]
        x_rot = x * np.cos(theta) - y * np.sin(theta)
        y_rot = x * np.sin(theta) + y * np.cos(theta)
        x_prime.append(x_rot)
        y_prime.append(y_rot)
        
    for i in range(len(data)):
        data[i] = (x_prime[i], y_prime[i], data[i][2], data[i][3], data[i][4])
        
    return data

def rotAfterBatoid(x, y, rot):
    '''

    Parameters
    ----------
    x : np.array
        x coordinates of rays in Batoid referentiel
    y : np.array
        y coordinates of rays in Batoid referentiel
    rot : float
        angle of the telescope with sky in degrees
        
    Returns
    -------
    x_prime : np.array
        x coordinates of rays in Telescope referentiel
    y_prime : np.array
        y coordinates of rays in Telescope referentiel

    '''
    theta = np.deg2rad(-rot)
    x_prime = x*np.cos(theta) - y*np.sin(theta)
    y_prime = x*np.sin(theta) + y*np.cos(theta)
    return x_prime, y_prime

def getTransmissionRate(band, wavelength=None):
    '''

    Parameters
    ----------
    band : string
        'u', 'g', 'r', 'i', 'z', or 'y'
    wavelength : float in nm, optional
        Wavlength to obtain transmission and reflection rate for the telescope.
        If is None, we take the medium wavelength of the band.

    Returns
    -------
    t : array
        Transmission's rate for each optic element.
    wavelength : float
        Wavelength (return because of cause it was on None).

    '''
    
    t = []
    
    if wavelength == None:
        bands = {"u": 355.0, "g": 475.0, "r": 622.0, "i": 763.0, "z": 905.0, "y": 1000.0}
        wavelength = bands[band]

    base_path = os.path.dirname(__file__)
    data_path = os.path.abspath(os.path.join(base_path, '..', '..', 'data'))
    
    filenames = [
        "lens1.dat",
        "lens2.dat",
        "lens3.dat",
        f"filter_{band}.dat",
        "detector.dat"
    ]
    files = [os.path.join(data_path, name) for name in filenames]

    for file in files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Missing file: {file}")
        
        df = pd.read_csv(file, comment="#", sep=r'\s+', names=["wavelength", "throughput"])
            
        wavelengths = df["wavelength"].values
        throughput = df["throughput"].values
            
        index = np.argmin(np.abs(wavelengths - wavelength))
        t.append(throughput[index])

    return t, wavelength

def initTelescope(band, t=None, r=None, wavelength=None):
    '''

    Parameters
    ----------
    band : string
        'u', 'g', 'r', 'i', 'z', or 'y'
    t : array, optional
        Transmission's rate for each optic element.
        The default is None.
    r : array, optional
        Reflection's rate for each optic element.
        The default is None.
    wavelength : float, optional
        Wavelength for the work.
        The default is None.

    Returns
    -------
    telescope : Batoid.Optic
        ComCam in the band give with new transmission/reflection rate. Need for Batoid's simulation
    wavelength : float
        Wavelength used to obtain transmission/reflection rate.
        return because in case is was on None.

    '''
    file_yaml = "ComCamSpiders_" + band + ".yaml"
    telescope = batoid.Optic.fromYaml(file_yaml)
    
    if t == None and r == None:
        if wavelength == None:
            t, wavelength = getTransmissionRate(band, wavelength)
        else:
            t, _ = getTransmissionRate(band, wavelength)

        for surface in telescope.itemDict.values():
            if isinstance(surface, batoid.RefractiveInterface):
                if surface.name.split('_')[0] in ['L1']:
                    surface.forwardCoating = batoid.SimpleCoating(1-t[0], t[0])
                    surface.reverseCoating = batoid.SimpleCoating(1-t[0], t[0])
                elif surface.name.split('_')[0] in ['L2']:
                    surface.forwardCoating = batoid.SimpleCoating(1-t[1], t[1])
                    surface.reverseCoating = batoid.SimpleCoating(1-t[1], t[1])
                elif surface.name.split('_')[0] in ['L3']:
                    surface.forwardCoating = batoid.SimpleCoating(1-t[2], t[2])
                    surface.reverseCoating = batoid.SimpleCoating(1-t[2], t[2])
                elif surface.name.split('_')[0] in ['Filter']:
                    surface.forwardCoating = batoid.SimpleCoating(1-t[3], t[3])
                    surface.reverseCoating = batoid.SimpleCoating(1-t[3], t[3])
            if isinstance(surface, batoid.Detector):
                surface.forwardCoating = batoid.SimpleCoating(1-t[4], t[4])
                
    elif t != None and r != None:
        for surface in telescope.itemDict.values():
            if isinstance(surface, batoid.RefractiveInterface):
                if surface.name.split('_')[0] in ['L1']:
                    surface.forwardCoating = batoid.SimpleCoating(r[0], t[0])
                    surface.reverseCoating = batoid.SimpleCoating(r[0], t[0])
                elif surface.name.split('_')[0] in ['L2']:
                    surface.forwardCoating = batoid.SimpleCoating(r[1], t[1])
                    surface.reverseCoating = batoid.SimpleCoating(r[1], t[1])
                elif surface.name.split('_')[0] in ['L3']:
                    surface.forwardCoating = batoid.SimpleCoating(r[2], t[2])
                    surface.reverseCoating = batoid.SimpleCoating(r[2], t[2])
                elif surface.name.split('_')[0] in ['Filter']:
                    surface.forwardCoating = batoid.SimpleCoating(r[3], t[3])
                    surface.reverseCoating = batoid.SimpleCoating(r[3], t[3])
            if isinstance(surface, batoid.Detector):
                surface.forwardCoating = batoid.SimpleCoating(r[4], t[4])
                    
    elif t != None:
        for surface in telescope.itemDict.values():
            if isinstance(surface, batoid.RefractiveInterface):
                if surface.name.split('_')[0] in ['L1']:
                    surface.forwardCoating = batoid.SimpleCoating(1-t[0], t[0])
                    surface.reverseCoating = batoid.SimpleCoating(1-t[0], t[0])
                elif surface.name.split('_')[0] in ['L2']:
                    surface.forwardCoating = batoid.SimpleCoating(1-t[1], t[1])
                    surface.reverseCoating = batoid.SimpleCoating(1-t[1], t[1])
                elif surface.name.split('_')[0] in ['L3']:
                    surface.forwardCoating = batoid.SimpleCoating(1-t[2], t[2])
                    surface.reverseCoating = batoid.SimpleCoating(1-t[2], t[2])
                elif surface.name.split('_')[0] in ['Filter']:
                    surface.forwardCoating = batoid.SimpleCoating(1-t[3], t[3])
                    surface.reverseCoating = batoid.SimpleCoating(1-t[3], t[3])
            if isinstance(surface, batoid.Detector):
                surface.forwardCoating = batoid.SimpleCoating(1-t[4], t[4])
                    
    elif r != None:
        for surface in telescope.itemDict.values():
            if isinstance(surface, batoid.RefractiveInterface):
                if surface.name.split('_')[0] in ['L1']:
                    surface.forwardCoating = batoid.SimpleCoating(r[0], 1-r[0])
                    surface.reverseCoating = batoid.SimpleCoating(r[0], 1-r[0])
                elif surface.name.split('_')[0] in ['L2']:
                    surface.forwardCoating = batoid.SimpleCoating(r[1], 1-r[1])
                    surface.reverseCoating = batoid.SimpleCoating(r[1], 1-r[1])
                elif surface.name.split('_')[0] in ['L3']:
                    surface.forwardCoating = batoid.SimpleCoating(r[2], 1-r[2])                        
                    surface.reverseCoating = batoid.SimpleCoating(r[2], 1-r[2])
                elif surface.name.split('_')[0] in ['Filter']:
                    surface.forwardCoating = batoid.SimpleCoating(r[3], 1-r[3])
                    surface.reverseCoating = batoid.SimpleCoating(r[3], 1-r[3])
            if isinstance(surface, batoid.Detector):
                surface.forwardCoating = batoid.SimpleCoating(r[4], 1-r[4])
                    
    else:
        print("No change of transmission and reflection rate.")
    
    return telescope, wavelength

def initParams(thetapos, rot, nrad=600, naz=1200, minflux=1e-3):
    '''

    Parameters
    ----------
    thetapos : (theta_x, theta_y)
        Position of the source in Batoid's coordinates.
    rot : float
        Rotation of ComCam with the sky.
    nrad : int, optional
        Number of circle for AsPolar simulation.
        The default is 300.
    naz : int, optional
        Number of points on each circle for AsPolar simulation.
        Must be a multiple of 6.
        The default is 900.
    minflux : float, optional
        Cutoff for Batoid's calculus. The default is 1e-3.

    Returns
    -------
    init_simu : array
        Parameters to start the ray tracing simulation.

    '''
    theta_x = thetapos[0]
    theta_y = thetapos[1]
        
    init_simu = [(theta_x, theta_y, nrad, naz, minflux)]
    init_simu = rotBeforeBatoid(init_simu, rot)
    return init_simu

def batoidCalcul(telescope, init_simu, wavelength, debug=False):
    '''

    Parameters
    ----------
    telescope : Batoid.Optic
        telescope use for Batoid's simulation
    init_simu : array
        initial data for Batoid's calculus
    wavelength : float
        Wavelength to simulate.
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
            optic=telescope, wavelength=wavelength*1e-9,
            theta_x=np.deg2rad(dat[0]), theta_y=np.deg2rad(dat[1]),
            nrad=dat[2], naz=dat[3], flux=1.0
        )
    
        rForward, rReverse = telescope.traceSplit(rays, minFlux=dat[4], _verbose=False) # _verbose = log calculus
    
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

def getSimuImage(px, py, x, y, flux, binning, pixelsize=1e-5):
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
    binning : int
        Binning used on the data.

    Returns
    -------
    H.T : Histogram2D of all ghosts.

    '''
    # On suppose que x_prime, y_prime et flux sont des arrays 1D avec les positions et le flux associé

    scale = binning*pixelsize
    X, Y = px*scale, py*scale

    extent = [-X/2.0, X/2.0, -Y/2.0, Y/2.0]
    x_min, x_max, y_min, y_max = extent
    
    mask = (x > x_min) & (x < x_max) & (y > y_min) & (y < y_max)
    
    x = x[mask]
    y = y[mask]
    flux = flux[mask]
    
    H, _, _ = np.histogram2d(x, y, bins=[px, py], weights=flux, range=[[x_min, x_max], [y_min, y_max]])
    H = H.T

    return H

def getGhosts(telescope, init_simu, wavelength, nbghost=None, ghostmap=False, update_flux=False, name=None, debug=False):
    '''

    Parameters
    ----------
    telescope : Batoid.Optic
        telescope use for Batoid's simulation
    init_simu : array
        initial data for Batoid's calculus
    wavelength : float
        Wavelength to simulate.
    nbghost : int, optional
        Number of ghost you need.
        It will return the n most brightness ghosts in surface.
        The default is 5.
    ghostmap : bool, optional
        If on True, it will plot a map of ghosts on informations on them.
        The default is False.
    name : string, optional
        Save the ghostmap as name.png.
        The default is None.

    Returns
    -------
    tuple
        x, y and the flux of each rays regroup.

    '''
    xsep, ysep, flux, path = batoidCalcul(telescope, init_simu, wavelength, debug)
    nghost = len(xsep)
    ref = ['L1_entrance', 'L1_exit', 'L2_entrance', 'L2_exit', 'Filter_entrance', 'Filter_exit', 'L3_entrance', 'L3_exit', 'Detector']
    paths = []

    if nghost == 0:
        return [], [], []
        
    for i in range(nghost):
        paths.append([ipath for ipath in path[i] if ipath in ref])
    
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
        flux_update.append(flux[i]/(len(individual_areas)*hex_area))
        
        if debug == True:
            print(f"Flux/m2 for ghost {i+1} : {flux_update[i][0]}")
    
    plt.close()

    path = paths.copy()
    x, y, f, paths = [], [], [], []

    if nbghost == None:
        nbghost = nghost

    else:
        nbghost += 1
    
    idx_in_cercle = []
    
    for i in range(len(xsep)):
        r = np.sqrt(np.array(xsep[i])**2 + np.array(ysep[i])**2)
        if np.any(r <= 0.32/5.0):
            idx_in_cercle.append(i)
    
    idx_sorted = sorted(
        idx_in_cercle,
        key=lambda i: flux_update[i][0],
        reverse=True
        )
    
    idx_keep = idx_sorted[:nbghost]

    flat_flux = [f[0] for f in flux_update]  # si tous ont au moins un élément
    max_flux = max(flat_flux)

    for i in idx_keep:
        if flux_update[i][0]==max_flux:
            continue
            
        x.append([ix for ix in xsep[i]])
        y.append([iy for iy in ysep[i]])
        if update_flux==False:
            f.append([iflux for iflux in flux[i]])
        else:
            f.append([iflux for iflux in flux_update[i]])
        paths.append([ipath for ipath in path[i]])

    if ghostmap == True:
        
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
            axes[i].plot(0.32/5.0*np.cos(th), 0.32/5.0*np.sin(th), c='r', label="Cercle optique")
        
            offsets = hexbin_collection[i].get_offsets()
            counts = hexbin_collection[i].get_array()
            
            hex_size = 0.5 / 400
            hex_area = (3 * np.sqrt(3) / 2) * (hex_size ** 2)
        
            mask = counts > 0
            individual_areas = [hex_area for _ in offsets[mask]]
            
            axes[i].set_xlabel(f"{np.array(path[idx_keep[i]])}", fontsize=8, color="gray", labelpad=5)
            if update_flux == True:
                axes[i].set_title(f"Ghost {i+1}, Flux : {flux[idx_keep[i]][0]/(len(individual_areas)*hex_area):.5f}", fontsize=10)
            else:
                axes[i].set_title(f"Ghost {i+1}, Flux : {flux[idx_keep[i]][0]:.5f}", fontsize=10)
            axes[i].axis("equal")
            
        for i in range(len(idx_keep), len(axes)):
            axes[i].axis("off")
        
        if name != None:
            plt.savefig(name, bbox_inches='tight')
        plt.show()
    
    x, y, f = groupData(x, y, f)

    return x, y, f

def ccd_extractor(image):
    '''

    Parameters
    ----------
    image : image : array
        ImageFits.getArray().

    Returns
    -------
    array
        ccd[n][0] = image on CCD n.
        ccd[n][1] = origin (left bottom corner) of CCD n on the original data (in pixel).

    '''
    mask = ~np.isnan(image)
    validate_lines = np.any(mask, axis=1)
    line_blocs = _blocs_detector(validate_lines)

    ccd = []

    for i_start, i_end in line_blocs:
        sub_line = image[i_start:i_end, :]
        mask_col = ~np.isnan(sub_line)
        validate_cols = np.any(mask_col, axis=0)
        col_blocs = _blocs_detector(validate_cols)
        
        for j_start, j_end in col_blocs:
            bloc = image[i_start:i_end, j_start:j_end]
            if not np.isnan(bloc).any():
                ccd.append((bloc.copy(), (i_start, j_start)))  # On copie pour éviter les effets de bord

    return ccd

def _blocs_detector(binary_vector):
    blocs = []
    in_bloc = False
    for i, val in enumerate(binary_vector):
        if val and not in_bloc:
            start = i
            in_bloc = True
        elif not val and in_bloc:
            end = i
            blocs.append((start, end))
            in_bloc = False
    if in_bloc:
        blocs.append((start, len(binary_vector)))
    return blocs



##################################################
#          The work below is not finish          #
##################################################



def getAiry(xstar, ystar, rot, D=8.36, f=9.8648):
    '''

    Work in progress, do not use in this form.

    '''
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
    '''

    Work in progress, do not use in this form.

    '''
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
