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
from itertools import combinations

import lsst.afw.image as afwimage
import lsst.afw.fits as afwfits

ghost_292=afwimage.ImageF.readFits("../notebooks/comcam_ghosts/ghost_292.fits", 0)
md_292=afwfits.readMetadata("../notebooks/comcam_ghosts/ghost_292.fits")
real = ghost_292.getArray()

def fit(real):    
    tx = -0.22494359089904253
    ty = -0.008266610482656549
    rot = md_292["ROTPA"]

    t, wavelength = gsim.getTransmissionRate('r', wavelength=622.2)

    print(f"tx = {tx}, ty = {ty}")

    real_ccd = gsim.ccd_extractor(real)
    real_ccd = real_ccd[3][0]
    real_flat = real_ccd.ravel()
    real_flat = np.nan_to_num(real_flat, nan=0.0, posinf=0.0, neginf=0.0)

    def chi2(tl1, tl2, tl3, tf, td):
        transm = [tl1, tl2, tl3, tf, td]
        ComCam, _ = gsim.initTelescope('r', t=transm)
        Params = gsim.initParams(thetapos=(tx, ty), rot=rot)
        x_grp, y_grp, flux_grp = gsim.getGhosts(ComCam, Params, wavelength, nbghost=5)
        
        if len(x_grp) == 0:
            return np.sum((real_flat)**2)
        
        x_grp, y_grp = gsim.rotAfterBatoid(x_grp, y_grp, rot)
        px, py = real.shape[1], real.shape[0]
        hist = gsim.getSimuImage(px, py, x_grp, y_grp, flux_grp, binning=8.0)
        hist = gfit.applyGrid(real, hist)

        real_wtnan = np.nan_to_num(real, nan=0.0, posinf=0.0, neginf=0.0)
        hist_wtnan = np.nan_to_num(hist, nan=0.0, posinf=0.0, neginf=0.0)
    
        real_fit, hist_fit = sim.removeSourcesBoth(real_wtnan, hist_wtnan)
    
        real_fit_flat = real_fit.ravel()
        hist_flat = hist_fit.ravel()
    
        clean_flat = 1144.764573552056 * hist_flat + 954.143
        chi2 = np.sum((real_fit_flat - clean_flat) ** 2)
        
        t_formatted = ', '.join(f'{x:.4f}' for x in transm)
        print(f"t = [{t_formatted}], chi2 = {chi2:.4e}")
        return chi2

    m = Minuit(chi2, tl1=t[0], tl2=t[1], tl3=t[2], tf=t[3], td=t[4])
    m.errordef = Minuit.LEAST_SQUARES
    m.precision = 1e-8
    m.limits['tl1'] = (0.8, 1.0)
    m.limits['tl2'] = (0.8, 1.0)
    m.limits['tl3'] = (0.8, 1.0)
    m.limits['tf'] = (0.8, 1.0)
    m.limits['td'] = (0.8, 1.0)
    m.migrad()
    m.hesse()
# n_calls, iterations, matrix_cov
    print("Transmission L1 optimal          :", m.values['tl1'])
    print("Transmission L2 optimal          :", m.values['tl2'])
    print("Transmission L3 optimal          :", m.values['tl3'])
    print("Transmission Filter optimal      :", m.values['tf'])
    print("Transmission Detector optimal    :", m.values['td'])
    print("Erreur sur Transmission L1       :", m.errors['tl1'])
    print("Erreur sur Transmission L2       :", m.errors['tl2'])
    print("Erreur sur Transmission L3       :", m.errors['tl3'])
    print("Erreur sur Transmission Filter   :", m.errors['tf'])
    print("Erreur sur Transmission Detector :", m.errors['td'])
    print("Transmission Meta-Data L1        :", t[0])
    print("Transmission Meta-Data L2        :", t[1])
    print("Transmission Meta-Data L3        :", t[2])
    print("Transmission Meta-Data Filter    :", t[3])
    print("Transmission Meta-Data Detector  :", t[4])

    # Ploting part

    for par in ["tl1", "tl2", "tl3", "tf", "td"]:
        plt.figure()
        x, y = m.profile(par, subtract_min=True)
        plt.plot(x, y)
        plt.title(f"Profil du chi² pour {par}")
        plt.xlabel(par)
        plt.ylabel("Δchi²")
        plt.savefig(f"{par}vschi2")
        plt.close()

    param_names = ["tl1", "tl2", "tl3", "tf", "td"]
    combs = list(combinations(param_names, 2))
    
    for p1, p2 in combs:
        plt.figure()
        m.draw_contour(p1, p2)
        plt.title(f"Contours de chi² : {p1} vs {p2}")
        plt.xlabel(p1)
        plt.ylabel(p2)
        plt.savefig(f"{p1}vs{p2}")
        plt.close()

fit(real)
