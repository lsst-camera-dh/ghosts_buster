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
    tx = -0.22494359089904253
    ty = -0.008266610482656549
    rot = 82.32264284072077
    
    ComCam, wavelength = gsim.initTelescope("r", wavelength=622.2)

    print(f"tx = {tx}, ty = {ty}")

    real_ccd = gsim.ccd_extractor(real)
    real_ccd = real_ccd[3][0]
    real_flat = real_ccd.ravel()
    real_flat = np.nan_to_num(real_flat, nan=0.0, posinf=0.0, neginf=0.0)

    Params = gsim.initParams(thetapos=(tx, ty), rot=rot)
    x_grp, y_grp, flux_grp = gsim.getGhosts(ComCam, Params, wavelength, nbghost=5)
    x_grp, y_grp = gsim.rotAfterBatoid(x_grp, y_grp, rot)
    px, py = real.shape[1], real.shape[0]
    PS = 1e-5
    
    def chi2(ps):
        hist = gsim.getSimuImage(px, py, x_grp, y_grp, flux_grp, binning=8.0, pixelsize=ps)
        hist = gfit.applyGrid(real, hist)

        real_wtnan = np.nan_to_num(real, nan=0.0, posinf=0.0, neginf=0.0)
        hist_wtnan = np.nan_to_num(hist, nan=0.0, posinf=0.0, neginf=0.0)
    
        hist_ccd = gsim.ccd_extractor(hist)
        hist_ccd = hist_ccd[3][0]
        hist_flat = hist_ccd.ravel()
    
        clean_flat = 1144.764573552056 * hist_flat + 954.143
        chi2 = np.sum((real_flat - clean_flat) ** 2)
        
        print(f"scale = {ps}, chi2 = {chi2}")
        return chi2

    m = Minuit(chi2, ps=PS)
    m.errordef = Minuit.LEAST_SQUARES
    m.precision = 1e-10
    m.limits['ps'] = (0.9e-5, 1.01e-5)
    m.migrad()
    m.hesse()
# n_calls, iterations, matrix_cov
    print("Scale optimal (m/pixel)       :", m.values['ps'])
    print("Erreur sur le scale (m/pixel) :", m.errors['ps'])
    print("Scale dans la littérature     :", PS)

fit(real)
