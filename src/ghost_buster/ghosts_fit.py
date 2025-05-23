import numpy as np
import pylab as plt
from astropy.stats import sigma_clipped_stats
from scipy.optimize import curve_fit
from iminuit import Minuit
from .sources_image import extractSources, removeSourcesBoth

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

def getNoise(image, x=[(1100, 1400)], y=[(100, 400)]):

    n = len(x)
    sub_image = []
    
    sub_image = np.concatenate([image[x[i][0]:x[i][1], y[i][0]:y[i][1]].flatten() for i in range(len(x))])
    
    mean, median, std = sigma_clipped_stats(sub_image, sigma=3.0)
        
    return mean, std
    
def showNoise(image, x=[(1100, 1400)], y=[(100, 400)], sigma=5.0, name=None):
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
    
    n = len(x)
    sub_image = []
    
    sub_image = np.concatenate([image[x[i][0]:x[i][1], y[i][0]:y[i][1]].flatten() for i in range(len(x))])
    
    mean, median, std = sigma_clipped_stats(sub_image, sigma=3.0)
    x_hist = sub_image.flatten()
    mask = (x_hist > mean - sigma*std) & (x_hist < mean + sigma*std)
    x_hist = x_hist[mask]
    x_fit = np.linspace(mean - sigma*std, mean + sigma*std, 100)
    plt.hist(x_hist, bins=100, range=(mean - sigma*std, mean + sigma*std), density=True, alpha=0.6, label='Histogramme')
    plt.plot(x_fit, gaussian(x_fit, mean, std), 'r-', label='Fit gaussien')
    plt.legend()
    if name != None:
        plt.savefig(name, bbox_inches='tight')
    plt.show()

def double_gaussian(x, A1, mu1, sigma1, A2, mu2, sigma2):
    gauss1 = A1 * np.exp(-0.5 * ((x - mu1) / sigma1)**2)
    gauss2 = A2 * np.exp(-0.5 * ((x - mu2) / sigma2)**2)
    return gauss1 + gauss2

def double_gaussian2(x, A1, mu1, sigma1, A2, mu2, sigma2):
    gauss1 = A1 * np.exp(-0.5 * ((x - mu1) / sigma1)**2)
    gauss2 = A2 * np.exp(-0.5 * ((x - mu2) / sigma2)**2)
    return gauss1, gauss2

def triple_gaussian(x, A1, mu1, sigma1, A2, mu2, sigma2, A3, mu3, sigma3):
    gauss1 = A1 * np.exp(-0.5 * ((x - mu1) / sigma1)**2)
    gauss2 = A2 * np.exp(-0.5 * ((x - mu2) / sigma2)**2)
    gauss3 = A3 * np.exp(-0.5 * ((x - mu3) / sigma3)**2)
    return gauss1 + gauss2 + gauss3

def triple_gaussian2(x, A1, mu1, sigma1, A2, mu2, sigma2, A3, mu3, sigma3):
    gauss1 = A1 * np.exp(-0.5 * ((x - mu1) / sigma1)**2)
    gauss2 = A2 * np.exp(-0.5 * ((x - mu2) / sigma2)**2)
    gauss3 = A3 * np.exp(-0.5 * ((x - mu3) / sigma3)**2)
    return gauss1, gauss2, gauss3

def getNoise2(image, x=[(1100, 1400)], y=[(100, 400)], sigma=10.0):
    n = len(x)
    sub_image = []
    
    sub_image = np.concatenate([image[x[i][0]:x[i][1], y[i][0]:y[i][1]].flatten() for i in range(len(x))])

    mean, median, std = sigma_clipped_stats(sub_image, sigma=3.0)
    
    x_hist = sub_image.flatten()
    mask = (x_hist > mean - sigma*std) & (x_hist < mean + sigma*std)
    x_hist = x_hist[mask]

    hist_vals, bin_edges = np.histogram(x_hist, bins=100, density=True)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    # Paramètres initiaux (à ajuster selon ton jeu de données)
    p0 = [1, np.mean(x_hist)-10, 5, 1, np.mean(x_hist)+10, 5]  # A1, mu1, sigma1, A2, mu2, sigma2

    params, cov = curve_fit(double_gaussian, bin_centers, hist_vals, p0=p0)

    print("\n--- Paramètres du fit double gaussienne ---")
    print(f"Gaussienne 1 (noise) :")
    print(f"  Amplitude   = {params[0]:.3f}")
    print(f"  Moyenne     = {params[1]:.3f}")
    print(f"  Sigma       = {params[2]:.3f}")
    
    print(f"\nGaussienne 2 (ghosts) :")
    print(f"  Amplitude   = {params[3]:.3f}")
    print(f"  Moyenne     = {params[4]:.3f}")
    print(f"  Sigma       = {params[5]:.3f}")

    return params

def showNoise2(image, x=[(1100, 1400)], y=[(100, 400)], sigma=10.0, name=None):    
    n = len(x)
    sub_image = []
    
    sub_image = np.concatenate([image[x[i][0]:x[i][1], y[i][0]:y[i][1]].flatten() for i in range(len(x))])

    mean, median, std = sigma_clipped_stats(sub_image, sigma=3.0)
    
    x_hist = sub_image.flatten()
    mask = (x_hist > mean - sigma*std) & (x_hist < mean + sigma*std)
    x_hist = x_hist[mask]

    hist_vals, bin_edges = np.histogram(x_hist, bins=100, density=True)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    # Paramètres initiaux (à ajuster selon ton jeu de données)
    p0 = [1, np.mean(x_hist)-10, 5, 1, np.mean(x_hist)+10, 5]  # A1, mu1, sigma1, A2, mu2, sigma2

    params, cov = curve_fit(double_gaussian, bin_centers, hist_vals, p0=p0)

    if params[1] < params[4]:
        print("\n--- Paramètres du fit double gaussienne ---")
        print(f"Gaussienne 1 (noise) :")
        print(f"  Amplitude   = {params[0]:.3f}")
        print(f"  Moyenne     = {params[1]:.3f}")
        print(f"  Sigma       = {params[2]:.3f}")
        
        print(f"\nGaussienne 2 (ghosts) :")
        print(f"  Amplitude   = {params[3]:.3f}")
        print(f"  Moyenne     = {params[4]:.3f}")
        print(f"  Sigma       = {params[5]:.3f}")

    else:
        print("\n--- Paramètres du fit double gaussienne ---")
        print(f"Gaussienne 1 (noise) :")
        print(f"  Amplitude   = {params[3]:.3f}")
        print(f"  Moyenne     = {params[4]:.3f}")
        print(f"  Sigma       = {params[5]:.3f}")
        
        print(f"\nGaussienne 2 (ghosts) :")
        print(f"  Amplitude   = {params[0]:.3f}")
        print(f"  Moyenne     = {params[1]:.3f}")
        print(f"  Sigma       = {params[2]:.3f}")

    x_fit = np.linspace(bin_centers.min(), bin_centers.max(), 1000)
    y_fit = double_gaussian2(x_fit, *params)
    
    plt.hist(x_hist, bins=100, density=True, alpha=0.6, label='Histogramme')

    if params[1] < params[4]:
        plt.plot(x_fit, y_fit[0], 'g-', label='Fit gaussienne Noise')
        plt.plot(x_fit, y_fit[1], 'r-', label='Fit gaussienne Ghosts')
    else:
        plt.plot(x_fit, y_fit[1], 'g-', label='Fit gaussienne Noise')
        plt.plot(x_fit, y_fit[0], 'r-', label='Fit gaussienne Ghosts')
        
    plt.plot(x_fit, double_gaussian(x_fit, *params), 'b-', label='Fit1 + Fit2')
    plt.legend()
    
    if name != None:
        plt.savefig(name, bbox_inches='tight')
    plt.show()

def showOverlapStar(image, x=[(20, 500)], y=[(580, 1000)], name=None):    
    n = len(x)
    sub_image = []
    
    sub_image = np.concatenate([image[x[i][0]:x[i][1], y[i][0]:y[i][1]].flatten() for i in range(len(x))])

    # mean, median, std = sigma_clipped_stats(sub_image, sigma=3.0)
    
    x_hist = sub_image.flatten()

    hist_vals, bin_edges = np.histogram(x_hist, bins=100, density=True)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    # Paramètres initiaux (à ajuster selon ton jeu de données)
    p0 = [1, np.mean(x_hist)-10, 5, 1, np.mean(x_hist)+10, 5, 1, np.mean(x_hist)-10, 5]  # A1, mu1, sigma1, A2, mu2, sigma2, A3, mu3, sigma3

    params, cov = curve_fit(triple_gaussian, bin_centers, hist_vals, p0=p0)

    print("\n--- Paramètres du fit double gaussienne ---")
    print(f"Gaussienne 1 (noise) :")
    print(f"  Amplitude   = {params[0]:.3f}")
    print(f"  Moyenne     = {params[1]:.3f}")
    print(f"  Sigma       = {params[2]:.3f}")
    
    print(f"\nGaussienne 2 (ghosts) :")
    print(f"  Amplitude   = {params[3]:.3f}")
    print(f"  Moyenne     = {params[4]:.3f}")
    print(f"  Sigma       = {params[5]:.3f}")

    x_fit = np.linspace(bin_centers.min(), bin_centers.max(), 1000)
    y_fit = triple_gaussian2(x_fit, *params)
    
    plt.hist(x_hist, bins=100, density=True, alpha=0.6, label='Histogramme')
    plt.plot(x_fit, y_fit[0], 'r-', label='Fit gaussienne Star')
    plt.plot(x_fit, y_fit[1], 'g-', label='Fit gaussienne Noise')
    plt.plot(x_fit, y_fit[2], 'b-', label='Fit gaussienne Ghosts')
    plt.plot(x_fit, triple_gaussian(x_fit, *params), label='Fit1 + Fit2 + Fit3')
    plt.legend()
    
    if name != None:
        plt.savefig(name, bbox_inches='tight')
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
    mean, std = getNoise(image)
    y_data = image.flatten()
    x_data = hist.flatten()

    params, cov = curve_fit(modelfit, x_data, y_data) # bounds=[(0.0, mean),(np.inf, mean + 1)]
    
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
    ghosts = modelfit(hist, a, b)
    clean = image - ghosts
    return clean

def applyFit2(image, hist, a, b):
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
    ghosts = modelfit(hist, a, b)
    clean = image - ghosts
    return clean

def applyGrid(image, hist):
    hist[np.isnan(image)] = np.nan
    return hist

def getChi2(image, clean):
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    clean = np.nan_to_num(clean, nan=0.0, posinf=0.0, neginf=0.0)

    image_src = extractSources(image)
    
    image_src = image_src.flatten()
    clean = clean.flatten()
    
    chi2 = np.sum((image_src - clean)**2 / (1/np.sqrt(np.abs(clean)))**2)
    # print(f'chi2 = {chi2}')

    dof = clean.size  # or image_obs.size - n_paramètres_fittés
    chi2_reduce = chi2 / dof
    # print(f'chi2_r = {chi2_reduce}')
    
    return chi2, chi2_reduce

def testMinuit(image, hist):
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    hist = np.nan_to_num(hist, nan=0.0, posinf=0.0, neginf=0.0)

    sigma_val = getNoise(image)[1]
    sigma = np.ones_like(image) * sigma_val

    image_fit, hist_fit = removeSourcesBoth(image, hist)

    image_flat = image_fit.ravel()
    hist_flat = hist_fit.ravel()
    sigma_flat = sigma.ravel()

    def chi2(amp, offset):
        model = amp * hist_flat + offset
        return np.sum(((image_flat - model) / sigma_flat) ** 2)

    m = Minuit(chi2, amp=300.0, offset=950.0)
    m.errordef = Minuit.LEAST_SQUARES
    m.migrad()
    m.hesse()

    # print("Amplitude optimale :", m.values['amp'])
    # print("Offset optimal     :", m.values['offset'])
    # print("Erreur sur amp     :", m.errors['amp'])
    # print("Erreur sur offset     :", m.errors['offset'])
    
    model_fit = m.values['amp'] * hist + m.values['offset']

    return image - model_fit, m

def testMinuitBis(image, hist):
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    hist = np.nan_to_num(hist, nan=0.0, posinf=0.0, neginf=0.0)

    sigma_val = getNoise(image)[1]
    sigma = np.ones_like(image) * sigma_val

    image_fit, hist_fit = removeSourcesBoth(image, hist)

    image_flat = image_fit.ravel()
    hist_flat = hist_fit.ravel()
    sigma_flat = sigma.ravel()

    def chi2(amp):
        model = amp * hist_flat + 955.241
        return np.sum(((image_flat - model) / sigma_flat) ** 2)

    m = Minuit(chi2, amp=300.0)
    m.errordef = Minuit.LEAST_SQUARES
    m.migrad()
    m.hesse()

    # print("Amplitude optimale :", m.values['amp'])
    # print("Offset optimal     :", m.values['offset'])
    # print("Erreur sur amp     :", m.errors['amp'])
    # print("Erreur sur offset     :", m.errors['offset'])
    
    model_fit = m.values['amp'] * hist + 955.241

    return image - model_fit, m

def showChi2vsA(image, simu, a, b):
    a_list = np.linspace(a-100, a+100, 20)
    
    chi2a = []
    
    for a_i in a_list:
        clean2 = applyFit2(image, simu, a_i, b)
        chi2a.append(getChi2(extractSources(image), clean2)[0])

    plt.scatter(a_list, np.array(chi2a))
    plt.title('Chi2 vs a')
    plt.show()

def testMinuit2(image, simu, noise, sigma=5.0):
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    simu = np.nan_to_num(simu, nan=0.0, posinf=0.0, neginf=0.0)
    noise = np.nan_to_num(noise, nan=0.0, posinf=0.0, neginf=0.0)
    
    image_fit, simu_fit = removeSourcesBoth(image, simu)

    image_flat = image_fit.ravel()
    simu_flat = simu_fit.ravel()
    noise_flat = noise.ravel()

    mean, median, std = sigma_clipped_stats(image_flat, sigma=3.0)
    
    mask = (image_flat > mean - sigma*std) & (image_flat < mean + sigma*std)
    image_flat = image_flat[mask]

    image_vals, _ = np.histogram(image_flat, bins=100, density=True)
    
    eps = 1e-8
    err = 1 / np.sqrt(np.abs(image_vals) + eps)
    
    def chi2(amp):
        model = amp * simu_flat + noise_flat
        model = np.nan_to_num(model, nan=0.0, posinf=0.0, neginf=0.0)
        model_vals, _ = np.histogram(model, bins=100, density=True)
        return np.sum(((image_vals - model_vals) / err) ** 2)

    m = Minuit(chi2, amp=900.0)
    m.errordef = Minuit.LEAST_SQUARES
    m.limits["amp"] = (0.0, None)
    m.migrad()
    m.hesse()

    print("Amplitude optimale :", m.values['amp'])
    print("Erreur sur amp     :", m.errors['amp'])
    
    model_fit = m.values['amp'] * simu

    return image - model_fit, m

def showChi2vsAmp(image, simu, noise, sigma=5.0):
    a_list = np.linspace(0, 12000, 1200)
    
    chi2a = []
    
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    simu = np.nan_to_num(simu, nan=0.0, posinf=0.0, neginf=0.0)
    noise = np.nan_to_num(noise, nan=0.0, posinf=0.0, neginf=0.0)
    
    image_fit, simu_fit = removeSourcesBoth(image, simu)

    image_flat = image_fit.ravel()
    simu_flat = simu_fit.ravel()
    noise_flat = noise.ravel()

    mean, median, std = sigma_clipped_stats(image_flat, sigma=3.0)
    
    mask = (image_flat > mean - sigma*std) & (image_flat < mean + sigma*std)
    image_flat = image_flat[mask]

    image_vals, _ = np.histogram(image_flat, bins=100, density=True)
    
    eps = 1e-8
    err = 1 / np.sqrt(np.abs(image_vals) + eps)
    
    for amp in a_list:
        model = amp * simu_flat + noise_flat
        model = np.nan_to_num(model, nan=0.0, posinf=0.0, neginf=0.0)
        model_vals, _ = np.histogram(model, bins=100, density=True)
        chi2a.append(np.sum(((image_vals - model_vals) / err) ** 2))

    plt.plot(a_list, np.array(chi2a))
    plt.title('Chi2 vs Amp')
    plt.show()

'''
Fit au ghost/ghost
-> faire un histogramme2D par ghost (attention aux histogrammes vides)
-> passer par les listes de compréhension pour automatiser en N-body
-> fixer des limites (un fort flux doit être 'prioriser' devant un faible)
    -> surement une carte à jouer avec un tri sur les flux et faire au i-1, i+1...

Problème à cause du lissage, les densités du template de ghost est sous-évaluée, il faut vraiment trouver un moyen pour lisser sans perdre les spyders
Et se renseigner sur Batoid en ce qui concerne les interférence constructives/destructives (par sûr que ce soit implémenter, peut jouer sur la superposition des ghosts si je ne me trompe pas)
Avant de commencer cette méthode, regarder à nouveau la méthode template de iMinuit, surement + efficace que mon code
'''
def testMinuit3(image, simu, noise, sigma=5.0):
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    simu = np.nan_to_num(simu, nan=0.0, posinf=0.0, neginf=0.0)
    noise = np.nan_to_num(noise, nan=0.0, posinf=0.0, neginf=0.0)
    
    image_fit, simu_fit = removeSourcesBoth(image, simu)

    image_flat = image_fit.ravel()
    simu_flat = simu_fit.ravel()
    noise_flat = noise.ravel()

    mean, median, std = sigma_clipped_stats(image_flat, sigma=3.0)
    
    mask = (image_flat > mean - sigma*std) & (image_flat < mean + sigma*std)
    image_flat = image_flat[mask]
    '''
    mean, median, std = sigma_clipped_stats(simu_flat, sigma=3.0)
    
    mask = (simu_flat > mean - sigma*std) & (simu_flat < mean + sigma*std)
    simu_flat = simu_flat[mask]
    '''
    image_vals, _ = np.histogram(image_flat, bins=100, density=True)
    
    err = np.sqrt(np.maximum(image_vals, 1))
    
    def chi2(amp):
        model = amp * simu_flat + noise_flat
        model = np.nan_to_num(model, nan=0.0, posinf=0.0, neginf=0.0)
        model_vals, _ = np.histogram(model, bins=100, density=True)
        return np.sum(((image_vals - model_vals) / err) ** 2)

    m = Minuit(chi2, amp=900.0)
    m.errordef = Minuit.LEAST_SQUARES
    m.limits["amp"] = (0.0, None)
    m.migrad()
    m.hesse()

    print("Amplitude optimale :", m.values['amp'])
    print("Erreur sur amp     :", m.errors['amp'])
    
    model_fit = m.values['amp'] * simu

    return image - model_fit, m

def testMinuitAiry(image, hist, temp):
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

    def chi2(AmpG, AmpA):
        model = AmpG * hist_flat + AmpA * temp_flat + 955.241
        return np.sum(((image_flat - model) / sigma_flat) ** 2)

    m = Minuit(chi2, AmpG=300.0, AmpA=100.0)
    m.errordef = Minuit.LEAST_SQUARES
    m.migrad()
    m.hesse()

    # print("Amplitude optimale :", m.values['amp'])
    # print("Offset optimal     :", m.values['offset'])
    # print("Erreur sur amp     :", m.errors['amp'])
    # print("Erreur sur offset     :", m.errors['offset'])
    
    model_fit = m.values['AmpG'] * hist + m.values['AmpA'] * temp + 955.241

    return image - model_fit, m