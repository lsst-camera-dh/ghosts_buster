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

ghost_292=afwimage.ImageF.readFits("../notebooks/comcam_ghosts/ghost_292.fits", 0)
md_292=afwfits.readMetadata("../notebooks/comcam_ghosts/ghost_292.fits")
real = ghost_292.getArray()

def fit(real):
    real = np.nan_to_num(real, nan=0.0, posinf=0.0, neginf=0.0)
    
    tx = -0.22494359089904253
    ty = -0.008266610482656549
    
    telescope = gsim.initTelescope()

    print(f"tx = {tx}, ty = {ty}")
    
    def chi2(rot):
        init_params = gsim.initParams(real, rot, bins=8, nrad=600, naz=1200, minflux=1e-3, thetapos=(tx, ty))
        x_grp, y_grp, flux_grp = gsim.getGhosts(telescope, init_params)

        real_flat = real.ravel()
        
        if len(x_grp) == 0:
            return np.sum((real_flat)**2)
        
        x_grp, y_grp = gsim.rotAfterBatoid(x_grp, y_grp, rot)
        px, py = real.shape[1], real.shape[0]
        hist, x_hist, y_hist = gsim.getSimuImage(px, py, x_grp, y_grp, flux_grp, binning=8)
        hist = gfit.applyGrid(real, hist)
        #clean, _ = gfit.testMinuit(real, hist)

        real_wtnan = np.nan_to_num(real, nan=0.0, posinf=0.0, neginf=0.0)
        hist_wtnan = np.nan_to_num(hist, nan=0.0, posinf=0.0, neginf=0.0)
    
        real_fit, hist_fit = sim.removeSourcesBoth(real_wtnan, hist_wtnan)
    
        real_flat = real_fit.ravel()
        hist_flat = hist_fit.ravel()
    
        clean_flat = 6248.169920405888 * hist_flat + 822.6034506966913                     
        chi2 = np.sum((real_flat - clean_flat) ** 2)
        
        print(f"rot = {rot}, chi2 = {chi2}")
        return chi2

    m = Minuit(chi2, rot=md_292["ROTPA"])
    m.errordef = Minuit.LEAST_SQUARES
    m.precision = 1e-5
    m.limits['rot'] = (md_292["ROTPA"] - 5.0, md_292["ROTPA"] + 5.0)
    m.migrad()
    m.hesse()
# n_calls, iterations, matrix_cov
    print("Rotation optimal        :", m.values['rot'])
    print("Erreur sur Rotation     :", m.errors['rot'])
    print("Rotation meta-data      :", md_292["ROTPA"])

fit(real)
