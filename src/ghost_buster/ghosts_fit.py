import numpy as np
import pylab as plt
from astropy.stats import sigma_clipped_stats
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic
from iminuit import Minuit
from .sources_image import removeSources, removeSourcesBoth

version = "0.2"

def sum_two_gaussian(x, A1, mu1, sigma1, A2, mu2, sigma2):
    '''

    Parameters
    ----------
    x : array
        Array of x coordinates.
    A1 : float
        Amplitude of the first gaussian.
    mu1 : float
        Mean of the first gaussian.
    sigma1 : float
        Standard deviation of the first gaussian.
    A2 : float
        Amplitude of the second gaussian.
    mu2 : float
        Mean of the second gaussian.
    sigma2 : float
        Standard deviation of the second gaussian.

    Returns
    -------
    array
        Sum of the two gaussian.

    '''
    gauss1 = A1 * np.exp(-0.5 * ((x - mu1) / sigma1)**2)
    gauss2 = A2 * np.exp(-0.5 * ((x - mu2) / sigma2)**2)
    return gauss1 + gauss2

def two_gaussian(x, A1, mu1, sigma1, A2, mu2, sigma2):
    '''

    Parameters
    ----------
    x : array
        Array of x coordinates.
    A1 : float
        Amplitude of the first gaussian.
    mu1 : float
        Mean of the first gaussian.
    sigma1 : float
        Standard deviation of the first gaussian.
    A2 : float
        Amplitude of the second gaussian.
    mu2 : float
        Mean of the second gaussian.
    sigma2 : float
        Standard deviation of the second gaussian.

    Returns
    -------
    array, array
        gaussian_1, gaussain_2.

    '''
    gauss1 = A1 * np.exp(-0.5 * ((x - mu1) / sigma1)**2)
    gauss2 = A2 * np.exp(-0.5 * ((x - mu2) / sigma2)**2)
    return gauss1, gauss2

def getNoise(image, bounds=None, sigma=10.0, name=None):
    '''

    Parameters
    ----------
    image : array
        ImageFits.getArray().
    bounds : tuple, optionnal
        Bounds interval as (xmin, ymin, xmax, ymax). The default is None
    sigma : float, optional
        Number of sigma for the selection of data. The default is 10.0.
    name : string, optional
        Will save the plot in 'name.png' if it is different of None.
        The default is None.

    Returns
    -------
    result : float
        Mean value of the noise.

    '''

    if bounds != None:
        sub_image = image[bounds[0]:bounds[3], bounds[2]:bounds[4]].ravel()
    else:
        sub_image = removeSources(image, np.nan).ravel()
    sub_image = sub_image[~np.isnan(sub_image)]
    
    mean, median, std = sigma_clipped_stats(sub_image, sigma=3.0)
    
    mask = (sub_image > mean - sigma*std) & (sub_image < mean + sigma*std)
    x_hist = sub_image[mask]

    hist_vals, bin_edges = np.histogram(x_hist, bins=100, density=True)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    p0 = [1, np.mean(x_hist)-10, 5, 1, np.mean(x_hist)+10, 5]  # A1, mu1, sigma1, A2, mu2, sigma2

    params, cov = curve_fit(sum_two_gaussian, bin_centers, hist_vals, p0=p0)

    if params[1] < params[4]:
        print("\n--- Paramètres du fit double gaussienne ---")
        print("Gaussienne 1 (noise) :")
        print(f"  Amplitude   = {params[0]:.3f}")
        print(f"  Moyenne     = {params[1]:.3f}")
        print(f"  Sigma       = {params[2]:.3f}")
        
        print("\nGaussienne 2 (ghosts) :")
        print(f"  Amplitude   = {params[3]:.3f}")
        print(f"  Moyenne     = {params[4]:.3f}")
        print(f"  Sigma       = {params[5]:.3f}")

    else:
        print("\n--- Paramètres du fit double gaussienne ---")
        print("Gaussienne 1 (noise) :")
        print(f"  Amplitude   = {params[3]:.3f}")
        print(f"  Moyenne     = {params[4]:.3f}")
        print(f"  Sigma       = {params[5]:.3f}")
        
        print("\nGaussienne 2 (ghosts) :")
        print(f"  Amplitude   = {params[0]:.3f}")
        print(f"  Moyenne     = {params[1]:.3f}")
        print(f"  Sigma       = {params[2]:.3f}")

    x_fit = np.linspace(bin_centers.min(), bin_centers.max(), 1000)
    y_fit = two_gaussian(x_fit, *params)
    
    plt.hist(x_hist, bins=100, density=True, alpha=0.6, label='Histogramme')

    if params[1] < params[4]:
        plt.plot(x_fit, y_fit[0], 'g-', label='Fit gaussienne Noise')
        plt.plot(x_fit, y_fit[1], 'r-', label='Fit gaussienne Ghosts')
        result = params[1], params[2]
    else:
        plt.plot(x_fit, y_fit[1], 'g-', label='Fit gaussienne Noise')
        plt.plot(x_fit, y_fit[0], 'r-', label='Fit gaussienne Ghosts')
        result = params[4], params[5]
        
    plt.plot(x_fit, sum_two_gaussian(x_fit, *params), 'b-', label='Fit1 + Fit2')
    plt.legend()
    
    if name != None:
        plt.savefig(name, bbox_inches='tight')
    plt.show()
    
    return result

def applyGrid(image, hist):
    '''

    Parameters
    ----------
    image : array
        ImageFits.getArray().
    hist : array
        Histogram2D, simulation of ghosts.

    Returns
    -------
    hist : array
        Histogram2D, simulation of ghosts with the grid of Nan which separe the different CCD.

    '''
    hist[np.isnan(image)] = np.nan
    return hist

def fitGhosts(image, hist):
    '''

    Parameters
    ----------
    image : array
        ImageFits.getArray().
    hist : array
        Histogram2D, simulation of ghosts with the grid of Nan which separe the different CCD.

    Returns
    -------
    array
        Image withut ghosts.
    Minuit object
        Contain a lot of information about the fit.

    '''
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    hist = np.nan_to_num(hist, nan=0.0, posinf=0.0, neginf=0.0)

    noise, sigma_val = getNoise(image=image)
    sigma = np.ones_like(image) * sigma_val

    image_fit, hist_fit = removeSourcesBoth(image, hist)

    image_flat = image_fit.ravel()
    hist_flat = hist_fit.ravel()
    sigma_flat = sigma.ravel()
    
    

    def chi2(amp):
        model = amp * hist_flat + noise
        return np.sum(((image_flat - model) / sigma_flat) ** 2)

    m = Minuit(chi2, amp=300.0)
    m.errordef = Minuit.LEAST_SQUARES
    m.migrad()
    m.hesse()

    print("Amplitude optimale :", m.values['amp'])
    print("Erreur sur amp     :", m.errors['amp'])
    
    model_fit = m.values['amp'] * hist + noise

    return image - model_fit, m

def fitAiry(image, hist, temp):
    '''
    
    Work in progress, please do not use this in this form.

    Parameters
    ----------
    image : array
        ImageFits.getArray().
    hist : array
        Histogram2D, simulation of ghosts with the grid of Nan which separe the different CCD.
    temp : array
        Simulation of Airy rings.

    Returns
    -------
    array
        Image withut ghosts and airy rings.
    Minuit object
        Contain a lot of information about the fit.

    '''
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    hist = np.nan_to_num(hist, nan=0.0, posinf=0.0, neginf=0.0)
    temp = np.nan_to_num(temp, nan=0.0, posinf=0.0, neginf=0.0)
    
    sigma_val = getNoise(image)[1]
    sigma = np.ones_like(image) * sigma_val

    image_fit, hist_fit = removeSourcesBoth(image, hist)
    _, temp_fit = removeSourcesBoth(image, temp)

    image_flat = image_fit.ravel()
    hist_flat = hist_fit.ravel()
    temp_flat = temp_fit.ravel()
    sigma_flat = sigma.ravel()

    noise = getNoise(image=image)

    def chi2(AmpG, AmpA):
        model = AmpG * hist_flat + AmpA * temp_flat + noise
        return np.sum(((image_flat - model) / sigma_flat) ** 2)

    m = Minuit(chi2, AmpG=300.0, AmpA=100.0)
    m.errordef = Minuit.LEAST_SQUARES
    m.migrad()
    m.hesse()

    print("Amplitude optimale Ghosts   :", m.values['AmpG'])
    print("Amplitude optimale Airy     :", m.values['AmpA'])
    print("Erreur sur Amplitude Ghosts :", m.errors['amp'])
    print("Erreur sur Amplitude Airy   :", m.errors['offset'])
    
    model_fit = m.values['AmpG'] * hist + m.values['AmpA'] * temp + noise

    return image - model_fit, m

def profil_radial(image_sub, starpos, bins=50, bin_width=None):
    '''

    Parameters
    ----------
    image_sub : array
        ImageFits.getArray() on only one CCD.
    starpos : tuple
        Position of the star in the CCD coordinates.
    bins : int, optional
        Number of radial bins to compute the profile (used if bin_width is None).
        Default is 50.
    bin_width : float, optional
        Width of each radial bin. If provided, overrides the `bins` parameter.

    Returns
    -------
    r_centers : array
        Array of radial distances corresponding to the center of each bin.
    profil : array
        Mean pixel value in each radial bin, representing the radial profile.

    '''
    py, px = image_sub.shape
    y_indices, x_indices = np.indices((py, px))

    r = np.sqrt((x_indices - starpos[0])**2 + (y_indices - starpos[1])**2)
    values = image_sub.flatten()
    r = r.flatten()

    if bin_width is not None:
        r_max = r.max()
        bins_edges = np.arange(0, r_max + bin_width, bin_width)
    else:
        bins_edges = bins

    profil, edges, _ = binned_statistic(r, values, statistic='mean', bins=bins_edges)

    r_centers = 0.5 * (edges[1:] + edges[:-1])

    return r_centers, profil
