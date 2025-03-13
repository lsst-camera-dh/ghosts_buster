import batoid
import numpy as np
import pandas as pd
import sys
import os

'''
On peut changer le filtre suivant pour la simulation.
'''
telescope = batoid.Optic.fromYaml("LSST_g.yaml")

for surface in telescope.itemDict.values():
    if isinstance(surface, batoid.RefractiveInterface):
        surface.forwardCoating = batoid.SimpleCoating(0.02, 0.98)
        surface.reverseCoating = batoid.SimpleCoating(0.02, 0.98)
    if isinstance(surface, batoid.Detector):
        surface.forwardCoating = batoid.SimpleCoating(0.30, 0.70)
        
'''
Nx, Ny sont les nombres de points que l'on souhaite avoir pour nos itérations'
min_x est l'angle minimal selon l'axe x
min_y est l'angle minimal selon l'axe y
max_x est l'angle maximal selon l'axe x
max_y est l'angle maximal selon l'axe y

one_ghost est un booléen (option), permet de garder la data pour les ghost d'un chemin optique en particulier'
'''

'''
min_x = 1.0
max_x = 2.0
min_y = 1.0
max_y = 2.0

Nx = int((max_x - min_x)*10)
Ny = int((max_y - min_y)*10)

theta_x = np.linspace(min_x, max_x, Nx)
theta_y = np.linspace(min_y, max_y, Ny)
'''

r = np.linspace(0, 5, 10)
theta = np.linspace(0, 8*np.pi, 10)

theta_x = r*np.cos(theta)
theta_y = r*np.sin(theta)

one_ghost = False

if one_ghost :
    path_check = [['M1', 'M2', 'M3', 'L1_entrance', 'L1_exit', 'L2_entrance', 'L2_exit', 'Filter_entrance', 'Filter_exit', 'L3_entrance', 'L3_exit', 'Detector', 'L3_exit', 'Detector']]

data_x, data_y = [], []
data_tx, data_ty = [], []

progress, keep_steps = 0, 0  # Compteur de progression
#total_steps = Nx*Ny  # Nombre total d'itérations
total_steps = len(theta_x)*len(theta_y)
print(f"{total_steps} simulations à faire")

for tx in theta_x :
    for ty in theta_y:
        progress+=1
        rays = batoid.RayVector.asPolar(
            telescope, wavelength=620e-9,
            theta_x=np.deg2rad(tx), theta_y=np.deg2rad(ty),
            naz=1000, nrad=300, flux=1.0
        )
        rForward, rReverse = telescope.traceSplit(rays, minFlux=1e-3)
                
        sys.stdout.write(f"\rProgress: {progress / total_steps * 100:.2f} %")
        sys.stdout.flush()
        
        if one_ghost:
            rForward = [rr for rr in rForward if rr.path in path_check]
        
        if len(rForward) == 0:
            continue
        
        keep_steps+=1

        for i, rr in enumerate(rForward[:]):
            data_x.append(rr.x.tolist())
            data_y.append(rr.y.tolist())
            data_tx.append(tx)
            data_ty.append(ty)

print(f"\n{keep_steps} / {total_steps} simulations garder")

# Vérifier si le dossier ../data/ existe, sinon le créer
save_dir = "../data/"
os.makedirs(save_dir, exist_ok=True)

# Préparer un DataFrame à partir des données
data_dict = {
    "x_data": [",".join(map(str, x)) for x in data_x],
    "y_data": [",".join(map(str, y)) for y in data_y],
    "theta_x": data_tx,
    "theta_y": data_ty
}

df = pd.DataFrame(data_dict)

# Sauvegarde dans le fichier CSV
csv_path = os.path.join(save_dir, "ray_data.csv")
df.to_csv(csv_path, index=False)

print("Données sauvegardées dans ../data/ray_data.csv")
