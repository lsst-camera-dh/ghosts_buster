import numpy as np
from astropy.stats import sigma_clipped_stats
from scipy.optimize import curve_fit

def modelghosts(x, a):
    return a * x

# Fonction modèle : ici un simple scale + offset
def modelfit(x, a, b):
    return a * x + b

def gaussian(x, mu, sigma):
    return 1 / (sigma*np.sqrt(2*np.pi)) * np.exp(-(x - mu)**2 / (2 * sigma**2))

def showNoise(image, x=(1150, 1200), y=(450, 500))
    image = np.nan_to_num(image, nan=0.0)
    sub_image = image[x[0]:x[1], y[0]:y[1]]
    mean, median, std = sigma_clipped_stats(sub_image, sigma=3.0)
    x_hist = sub_image.flatten()
    x_fit = np.linspace(mean - 5*std, mean + 5*std, 100)
    plt.hist(x_hist, bins=100, range=(mean - 5*std, mean + 5*std), density=True, alpha=0.6, label='Histogramme')
    plt.plot(x_fit, gaussian(x_fit, mean, std), 'r-', label='Fit gaussien')
    plt.show()

def getFit(image, hist):
    y_data = image.flatten()
    x_data = hist.flatten()

    params, cov = curve_fit(modelfit, x_data, y_data)
    
    print(f"Best fit: a = {params[0]:.3f}, b = {params[1]:.3f}")
    return params

def applyFit(image, hist):
    a, b = getFit(image, hist)
    ghosts = modelghosts(hist, a)
    clean = image - ghosts
    return clean
