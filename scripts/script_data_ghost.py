import batoid
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import sys
import os

# Chargement du télescope LSST
telescope = batoid.Optic.fromYaml("LSST_g.yaml")

for surface in telescope.itemDict.values():
    if isinstance(surface, batoid.RefractiveInterface):
        surface.forwardCoating = batoid.SimpleCoating(0.02, 0.98)
        surface.reverseCoating = batoid.SimpleCoating(0.02, 0.98)
    if isinstance(surface, batoid.Detector):
        surface.forwardCoating = batoid.SimpleCoating(0.30, 0.70)

# Paramètres de simulation
one_ghost = False

theta_x = 0
theta_y = np.linspace(0, 2, 10)

if one_ghost:
    path_check = [['M1', 'M2', 'M3', 'L1_entrance', 'L1_exit', 'L2_entrance', 'L2_exit', 
                   'Filter_entrance', 'Filter_exit', 'L3_entrance', 'L3_exit', 'Detector', 
                   'L3_exit', 'Detector']]

# Chemin de stockage
save_dir = "../data/"
os.makedirs(save_dir, exist_ok=True)
parquet_path = os.path.join(save_dir, "ray_data.parquet")

# Supprimer le fichier s'il existe déjà (optionnel)
if os.path.exists(parquet_path):
    os.remove(parquet_path)

# Fonction d'ajout au fichier Parquet
def append_to_parquet(data_list, parquet_path):
    """Ajoute les données accumulées dans un fichier Parquet."""
    table = pa.Table.from_pandas(pd.DataFrame(data_list))  # Conversion en format Parquet
    
    if not os.path.exists(parquet_path):  
        # Créer un nouveau fichier s'il n'existe pas
        with pq.ParquetWriter(parquet_path, table.schema) as writer:
            writer.write_table(table)
    else:
        # Ajouter au fichier existant
        existing_table = pq.read_table(parquet_path)  # Lire les données existantes
        new_table = pa.concat_tables([existing_table, table])  # Fusionner les tables
        pq.write_table(new_table, parquet_path)  # Réécrire le fichier complet

# Simulation et écriture progressive
progress, keep_steps = 0, 0
total_steps = len(theta_y)

data_list = []

print(f"{total_steps} simulations à faire")

for ty in theta_y:
    
    progress += 1

    rays = batoid.RayVector.asPolar(
        telescope, wavelength=620e-9,
        theta_x=np.deg2rad(ty), theta_y=np.deg2rad(ty),
        naz=1000, nrad=300, flux=1.0
    )
    rForward, rReverse = telescope.traceSplit(rays, minFlux=1e-4)

    sys.stdout.write(f"\rProgress: {progress / total_steps * 100:.2f} %")
    sys.stdout.flush()

    if one_ghost:
        rForward = [rr for rr in rForward if rr.path in path_check]

    if len(rForward) == 0:
        continue

    keep_steps += 1

    # Stocker temporairement les données
    for i, rr in enumerate(rForward[:]):
        data_list.append({
            "theta_x": ty,
            "theta_y": ty,
            "x_data": rr.x.tolist(),
            "y_data": rr.y.tolist(),
            "path": rr.path  # Stocker les chemins sous forme de liste
        })
    append_to_parquet(data_list, parquet_path)
    data_list.clear()  # Réinitialiser le buffer

print(f"\n{keep_steps} / {total_steps} simulations gardées")
print(f"Données sauvegardées dans {parquet_path}")
