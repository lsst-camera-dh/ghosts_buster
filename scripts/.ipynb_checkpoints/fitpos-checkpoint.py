import sys
import os
sys.path.insert(0, os.path.abspath('../src'))

from ghost_buster import sources_image as sim
from ghost_buster import ghosts_simu as gsim
from ghost_buster import ghosts_fit as gfit
from ghost_buster import sources_image as sim

import matplotlib.pyplot as plt
import numpy as np
from iminuit import Minuit

import lsst.afw.image as afwimage
import lsst.afw.fits as afwfits

ghost=afwimage.ImageF.readFits("../notebooks/comcam_ghosts/ghost_054.fits", 0)
md=afwfits.readMetadata("../notebooks/comcam_ghosts/ghost_054.fits")
real = ghost.getArray()

def fit(real):
    real = np.nan_to_num(real, nan=0.0, posinf=0.0, neginf=0.0)
    
    txgaia = -0.22494359089904253
    tygaia = -0.008266610482656549
    
    bins = 8.0

    txpos, typos = txgaia, tygaia
    
    ComCam, wavelength = gsim.initTelescope("r", wavelength=622.2)

    print(f"txpos = {txpos}, typos = {typos}")

    real_flat = real.ravel()
    
    def chi2(tx, ty):
        Params = gsim.initParams(thetapos=(tx, ty), rot=md["ROTPA"])
        x_grp, y_grp, flux_grp = gsim.getGhosts(ComCam, Params, wavelength, nbghost=5)
        
        if len(x_grp) == 0:
            return np.sum((real_flat)**2)
        
        x_grp, y_grp = gsim.rotAfterBatoid(x_grp, y_grp, md["ROTPA"])
        px, py = real.shape[1], real.shape[0]
        hist = gsim.getSimuImage(px, py, x_grp, y_grp, flux_grp, binning=bins)
        hist = gfit.applyGrid(real, hist)

        real_wtnan = np.nan_to_num(real, nan=0.0, posinf=0.0, neginf=0.0)
        hist_wtnan = np.nan_to_num(hist, nan=0.0, posinf=0.0, neginf=0.0)
    
        real_fit, hist_fit = sim.removeSourcesBoth(real_wtnan, hist_wtnan)
    
        real_flat = real_fit.ravel()
        hist_flat = hist_fit.ravel()
    
        clean_flat = 1144.764573552056 * hist_flat + 954.143
        chi2 = np.sum((real_flat - clean_flat) ** 2)
        
        print(f"tx = {tx}, ty = {ty}, chi2 = {chi2}")
        return chi2

    m = Minuit(chi2, tx=txpos, ty=typos)
    m.errordef = Minuit.LEAST_SQUARES
    m.precision = 1e-8
    m.limits['tx'] = (txpos - 0.05, txpos + 0.05)
    m.limits['ty'] = (typos - 0.05, typos + 0.05)
    m.migrad()
    m.hesse()
# n_calls, iterations, matrix_cov
    print("Theta_X optimal        :", m.values['tx'])
    print("Theta_Y optimal        :", m.values['ty'])
    print("Erreur sur Theta_X     :", m.errors['tx'])
    print("Erreur sur Theta_Y     :", m.errors['ty'])
    print("Theta_X Gaia           :", txgaia)
    print("Theta_Y Gaia           :", tygaia)

    # Ploting part
    
    x, y = m.profile("tx", subtract_min=True)
    plt.plot(x, y)
    plt.title("Profil du chi² en fonction de tx")
    plt.xlabel("tx")
    plt.ylabel("Δchi²")
    plt.savefig("txvschi2")
    plt.close()
    
    x, y = m.profile("ty", subtract_min=True)
    plt.plot(x, y)
    plt.title("Profil du chi² en fonction de ty")
    plt.xlabel("ty")
    plt.ylabel("Δchi²")
    plt.savefig("tyvschi2")
    plt.close()

    m.draw_contour("tx", "ty")
    plt.title("Contours chi² : tx vs ty")
    plt.xlabel("tx")
    plt.ylabel("ty")
    plt.savefig("txvsty")
    plt.close()

fit(real)
