import sys
import os
sys.path.insert(0, os.path.abspath('../src'))

from ghost_buster import sources_image as sim
from ghost_buster import ghosts_simu as gsim
from ghost_buster import ghosts_fit as gfit
from ghost_buster import sources_image as sim

import pylab as plt
import numpy as np
from iminuit import Minuit

import lsst.afw.image as afwimage
import lsst.afw.fits as afwfits

ghost_292=afwimage.ImageF.readFits("../notebooks/comcam_ghosts/ghost_054.fits", 0)
md_292=afwfits.readMetadata("../notebooks/comcam_ghosts/ghost_054.fits")
real = ghost_292.getArray()

def fit(real):
    real = np.nan_to_num(real, nan=0.0, posinf=0.0, neginf=0.0)
    
    xgaia = 1104.27
    ygaia = 799.73
    
    xpos = int(xgaia)
    ypos = int(ygaia)
    
    ny, nx = real.shape
    x0, y0 = nx / 2, ny / 2
    scale = 0.2
    bins = 8.0
    dx = (xpos - x0) * scale * bins
    dy = (ypos - y0) * scale * bins
    txpos = dx / 3600
    typos = dy / 3600

    txgaia = (xgaia - x0) * scale * bins / 3600
    tygaia = (ygaia - y0) * scale * bins / 3600
    
    telescope = gsim.initTelescope()

    print(f"txpos = {txpos}, typos = {typos}")
    
    def chi2(tx, ty):
        init_params = gsim.initParams(real, md_292, bins=8, nrad=600, naz=1200, minflux=1e-3, thetapos=(tx, ty))
        x_grp, y_grp, flux_grp = gsim.getGhosts(telescope, init_params)

        real_flat = real.ravel()
        
        if len(x_grp) == 0:
            return np.sum((real_flat)**2)
        
        x_grp, y_grp = gsim.rotAfterBatoid(x_grp, y_grp, md_292)
        px, py = real.shape[1], real.shape[0]
        hist, x_hist, y_hist = gsim.getSimuImage(px, py, x_grp, y_grp, flux_grp, binning=8)
        hist = gfit.applyGrid(real, hist)
        #clean, _ = gfit.testMinuit(real, hist)

        real_wtnan = np.nan_to_num(real, nan=0.0, posinf=0.0, neginf=0.0)
        hist_wtnan = np.nan_to_num(hist, nan=0.0, posinf=0.0, neginf=0.0)
    
        real_fit, hist_fit = sim.removeSourcesBoth(real_wtnan, hist_wtnan)
    
        real_flat = real_fit.ravel()
        hist_flat = hist_fit.ravel()
    
        clean_flat = 11493.169 * hist_flat + 1321.599
        chi2 = np.sum((real_flat - clean_flat) ** 2)
        
        print(f"tx = {tx}, ty = {ty}, chi2 = {chi2}")
        return chi2

    m = Minuit(chi2, tx=txpos, ty=typos)
    m.errordef = Minuit.LEAST_SQUARES
    m.precision = 1e-5
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

fit(real)
