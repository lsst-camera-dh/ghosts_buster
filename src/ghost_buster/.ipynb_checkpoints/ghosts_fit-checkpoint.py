import numpy as np
import pylab as plt
from astropy.stats import sigma_clipped_stats
from scipy.optimize import curve_fit

version = "0.1"

def modelghosts(x, a):
    '''

    Parameters
    ----------
    x : np.array
        bin's sumlation
    a : float
        amplitude of ghosts fit

    Returns
    -------
    np.array
        bin's simulated (Batoid/ghosts) with amplitude a

    '''
    return a * x

def modelfit(x, a, b):
    '''

    Parameters
    ----------
    x : np.array
        bin's sumlation
    a : float
        amplitude of ghosts fit
    b : float
        sky background fit

    Returns
    -------
    np.array
        bin's simulated (Batoid/ghost) with amplitude a and skyground b

    '''
    return a * x + b

def gaussian(x, mu, sigma):
    '''

    Parameters
    ----------
    x : np.array
        x coordinates
    mu : float
        mean value
    sigma : float
        std value

    Returns
    -------
    np.array
        y coordinates (gausian)

    '''
    return 1 / (sigma*np.sqrt(2*np.pi)) * np.exp(-(x - mu)**2 / (2 * sigma**2))

def showNoise(image, x=(1150, 1200), y=(450, 500)):
    '''

    Parameters
    ----------
    image : np.array
        image.fits.getArray()
        Bin's values
    x : tuple, optional
        x range to capture noise 
        The default is (1150, 1200). (Optimal for the currently work)
    y : tuple, optional
        y range to capture noise 
        The default is (450, 500). (Optimal for the currently work)

    Returns
    -------
    None.

    '''
    image = np.nan_to_num(image, nan=0.0)
    sub_image = image[x[0]:x[1], y[0]:y[1]]
    mean, median, std = sigma_clipped_stats(sub_image, sigma=3.0)
    x_hist = sub_image.flatten()
    x_fit = np.linspace(mean - 5*std, mean + 5*std, 100)
    plt.hist(x_hist, bins=100, range=(mean - 5*std, mean + 5*std), density=True, alpha=0.6, label='Histogramme')
    plt.plot(x_fit, gaussian(x_fit, mean, std), 'r-', label='Fit gaussien')
    plt.show()

def getFit(image, hist):
    '''

    Parameters
    ----------
    image : np.array
        image.fits.getArray()
        Bin's values
    hist : np.array
        Simulated bins values

    Returns
    -------
    params : tuple
        Best parameters find by the fit

    '''
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    hist = np.nan_to_num(hist, nan=0.0, posinf=0.0, neginf=0.0)
    y_data = image.flatten()
    x_data = hist.flatten()

    params, cov = curve_fit(modelfit, x_data, y_data, bounds=[(0.0, 900.027),(np.inf, 1000)])
    
    print(f"Best fit: a = {params[0]:.3f}, b = {params[1]:.3f}")
    return params

def applyFit(image, hist):
    '''

    Parameters
    ----------
    image : np.array
        image.fits.getArray()
        Bin's values
    hist : np.array
        Simulated bins values

    Returns
    -------
    clean : np.array
        Try to remove ghosts on image by the fit

    '''
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    hist = np.nan_to_num(hist, nan=0.0, posinf=0.0, neginf=0.0)
    a, b = getFit(image, hist)
    ghosts = modelghosts(hist, a)
    clean = image - ghosts
    return clean

def applyFit2(image, hist, a):
    '''

    Parameters
    ----------
    image : np.array
        image.fits.getArray()
        Bin's values
    hist : np.array
        Simulated bins values

    Returns
    -------
    clean : np.array
        Try to remove ghosts on image by the fit

    '''
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    hist = np.nan_to_num(hist, nan=0.0, posinf=0.0, neginf=0.0)
    ghosts = modelghosts(hist, a)
    clean = image - ghosts
    return clean

def applyGrid(image, hist):
    hist[np.isnan(image)] = np.nan
    return hist
