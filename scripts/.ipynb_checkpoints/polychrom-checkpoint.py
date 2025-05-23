import sys
import os
sys.path.insert(0, os.path.abspath('../src'))

from ghost_buster import sources_image as sim
from ghost_buster import ghosts_simu as gsim
from ghost_buster import ghosts_fit as gfit

import pylab as plt
import numpy as np

with open('test.npy', 'rb') as f:
    real = np.load(f)

flux_mask = (real >= 30000) & (real <= 70000)
pixels_in_range = real[flux_mask]
real_test = np.where(flux_mask, real, 0.0)

init_params = gsim.initParams(real_test, bins=8, nrad=600, naz=1200, minflux=1e-5)

x_sep, y_sep, flux_sep = gsim.batoidCalcul2(init_params)
x_grp, y_grp, flux_grp = gsim.groupData(x_sep, y_sep, flux_sep)
x_grp, y_grp = gsim.rotAfterBatoid(x_grp, y_grp)
px, py = real.shape[1], real.shape[0]
hist, x_hist, y_hist = gsim.getSimuImage(px, py, x_grp, y_grp, flux_grp, binning=8)
hist = gfit.applyGrid(real, hist)

X, Y = np.meshgrid(x_hist, y_hist)

with open('result_634.npy', 'wb') as f:
    np.save(f, hist)
    np.save(f, x_hist)
    np.save(f, y_hist)

fig, ax = plt.subplots(figsize=(8, 8))
im = ax.pcolormesh(X, Y, hist, vmax=4e-2)
ax.set_aspect("equal")
ax.set_facecolor('black')
plt.xticks([])
plt.yticks([])
plt.savefig('test_poly', bbox_inches='tight')
plt.close()
