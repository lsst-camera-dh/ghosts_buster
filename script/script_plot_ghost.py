import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
from collections import Counter
import pyarrow.parquet as pq
import pyarrow as pa
import shutil
import os
import sys
import math
    
one_ghost = False
'''
Si l'option au dessus est False, cela signifie qu'il existe des groupes x_data, y_data qui possède les memes theta_x, theta_y.
Dans ce cas, je souhaite tous les avoir sur la meme image.
'''

# Vérifier si les dossiers ../data/ et ../data/images existe, sinon les créer
data_dir = "../data/"
images_dir = "../data/images/"

os.makedirs(data_dir, exist_ok=True)

if os.path.exists(images_dir):
    shutil.rmtree(images_dir)
os.makedirs(images_dir)

# Charger les données depuis le CSV
parquet_path = "../data/ray_data.parquet"
df = pd.read_parquet(parquet_path)
#print(df.head())
#print(df.shape[0])  # Nombre de lignes dans le DataFrame

# Conversion des colonnes contenant des listes
df["x_data"] = df["x_data"].apply(lambda x: list(x) if isinstance(x, (list, np.ndarray)) else eval(x))
df["y_data"] = df["y_data"].apply(lambda y: list(y) if isinstance(y, (list, np.ndarray)) else eval(y))

# Séparer les données
x_data = df["x_data"].tolist()
y_data = df["y_data"].tolist()
theta_x = df["theta_x"].tolist()
theta_y = df["theta_y"].tolist()
path = df["path"].tolist()

def get_subplot_grid(num_subplots):
    """Retourne (n_rows, n_cols) pour organiser num_subplots en subplots équilibrés."""
    n_cols = math.ceil(math.sqrt(num_subplots))
    n_rows = math.ceil(num_subplots / n_cols)
    return n_rows, n_cols
print("Start with a map of ghost")
# Compter les occurrences de chaque (theta_x, theta_y)
theta_pairs = list(zip(theta_x, theta_y))
theta_counter = Counter(theta_pairs)

# Trouver le couple (theta_x, theta_y) le plus fréquent
most_common_theta, _ = theta_counter.most_common(1)[0]

# Extraire les indices correspondant à ce couple
indices = [i for i, (tx, ty) in enumerate(theta_pairs) if (tx, ty) == most_common_theta]
nrows, ncols = get_subplot_grid(len(indices))

# Récupérer les x_data et y_data associés
selected_x_data = [x_data[i] for i in indices]
selected_y_data = [y_data[i] for i in indices]
selected_paths = [path[i] for i in indices]

fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), constrained_layout=True)

if nrows == 1 or ncols == 1:
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
else:
    axes = np.array(axes).flatten()

# Initialiser la collection de hexbin (vide au début)
hexbin_collection = [None] * len(indices)

# Fonction d'ajout au fichier Parquet
def append_to_parquet(data_list, bin_path):
    """Ajoute les données accumulées dans un fichier Parquet."""
    table = pa.Table.from_pandas(pd.DataFrame(data_list))  # Conversion en format Parquet
    
    if not os.path.exists(bin_path):  
        # Créer un nouveau fichier s'il n'existe pas
        with pq.ParquetWriter(bin_path, table.schema) as writer:
            writer.write_table(table)
    else:
        # Ajouter au fichier existant
        existing_table = pq.read_table(bin_path)  # Lire les données existantes
        new_table = pa.concat_tables([existing_table, table])  # Fusionner les tables
        pq.write_table(new_table, bin_path)  # Réécrire le fichier complet

data_list = []

# Chemin de référence
ref = ['M1', 'M2', 'M3', 'L1_entrance', 'L1_exit', 'L2_entrance', 'L2_exit', 'Filter_entrance', 'Filter_exit', 'L3_entrance', 'L3_exit', 'Detector']

# Remplir les subplots avec les données
for i in range(len(indices)):
    hexbin_collection[i] = axes[i].hexbin(
        x_data[i], y_data[i], extent=[-0.35, 0.35, -0.35, 0.35],
        gridsize=500
    )
    mean_x, mean_y = np.mean(x_data[i]), np.mean(y_data[i])
    axes[i].plot(mean_x, mean_y, marker='+', color='m', linestyle='None', markersize=8, alpha=0.8, label="Centre")  # Marqueur pour la moyenne
    if str(selected_paths[i]) == str(ref):
        axes[i].set_xlabel(f"{selected_paths[i]}", fontsize=8, color="indianred", labelpad=5)  # Ajout du path sous chaque subplot
    else :
        axes[i].set_xlabel(f"{selected_paths[i]}", fontsize=8, color="gray", labelpad=5)  # Ajout du path sous chaque subplot
    axes[i].axis("equal")  # Rendre les plots carrés
    axes[i].set_title(f"Ghost {i+1}", fontsize=10)
    
# Masquer les subplots vides
for i in range(len(indices), len(axes)):
    axes[i].axis("off")

# Ajouter une description sous la figure
fig.text(0.5, -0.05, f"Visualisation des ghosts pour θ_x={most_common_theta[0]:.2f}, θ_y={most_common_theta[1]:.2f}", ha="center", fontsize=10)

# Sauvegarde
subimages_path = os.path.join(images_dir, "ghosts_plot.png")
plt.savefig(subimages_path, bbox_inches="tight")

print("Subplots save")

if not one_ghost:
    from collections import defaultdict

    grouped_data = defaultdict(lambda: ([], [], []))  # Dictionnaire qui stocke les listes de x et y

    for i in range(len(theta_x)):
        key = (theta_x[i], theta_y[i])  # Utilisation des angles comme clé
        grouped_data[key][0].extend(x_data[i])  # Ajouter les x
        grouped_data[key][1].extend(y_data[i])  # Ajouter les y
        grouped_data[key][2].extend(path[i])    # Ajouter les path

    # Convertir les clés en listes
    theta_x, theta_y = zip(*grouped_data.keys())  # Séparer les angles en deux listes distinctes
    x_data = [grouped_data[key][0] for key in grouped_data.keys()]  # X fusionnés
    y_data = [grouped_data[key][1] for key in grouped_data.keys()]  # Y fusionnés
    path = [grouped_data[key][2] for key in grouped_data.keys()]    # Path fusionnés
    
# Vérification des dimensions
N = len(x_data)
print(f"Données chargées : {N} étapes de simulation")

# Création de la figure
fig, ax = plt.subplots()
th = np.linspace(0, 2*np.pi, 1000)
ax.plot(0.32*np.cos(th), 0.32*np.sin(th), c='r', label="Cercle optique")  # Cercle rouge
marker, = ax.plot([], [], marker='+', color='m', linestyle='None', markersize=8, alpha=0.8, label="Centre")

ax.set_xlim(-0.35, 0.35)
ax.set_ylim(-0.35, 0.35)
ax.set_aspect("equal")

# Stocker la collection hexbin pour mise à jour
hexbin_collection = [None]

# Stockage des valeurs des bins interessants
bin_path = os.path.join(data_dir, "bin_data.parquet")

# Supprimer le fichier s'il existe déjà (optionnel)
if os.path.exists(bin_path):
    os.remove(bin_path)

# Fonction d'animation
def update(frame):
    # Supprimer l'ancien hexbin
    if hexbin_collection[0] is not None:
        hexbin_collection[0].remove()

    # Calcul de la moyenne
    mean_x = np.mean(x_data[frame])
    mean_y = np.mean(y_data[frame])

    # Ajouter la nouvelle hexbin
    hexbin_collection[0] = ax.hexbin(x_data[frame], y_data[frame], extent=[-0.35, 0.35, -0.35, 0.35], gridsize=500, norm=colors.AsinhNorm())
    marker.set_data([mean_x], [mean_y])
    ax.set_title(f"Ray Tracing (color normalize with asinh function) - θx: {theta_x[frame]:.2f}°, θy: {theta_y[frame]:.2f}°")

    counts = hexbin_collection[0].get_array()  # Nombre de points par bin
    verts = hexbin_collection[0].get_offsets()  # Coordonnées des bins

    # Sélection des bins sur les axes de la moyenne
    bins_x = verts[:, 0][np.isclose(verts[:, 1], mean_y, atol=0.02)]
    counts_x = counts[np.isclose(verts[:, 1], mean_y, atol=0.02)]
    
    bins_y = verts[:, 1][np.isclose(verts[:, 0], mean_x, atol=0.02)]
    counts_y = counts[np.isclose(verts[:, 0], mean_x, atol=0.02)]

    image_path = os.path.join(images_dir, f"tx_{theta_x[frame]:.2f}_ty_{theta_y[frame]:.2f}.png")
    plt.savefig(image_path)
    
    sys.stdout.write(f"\rProgress: {(frame+1) / N * 100:.2f} %")
    sys.stdout.flush()
    
    data_list.append({
            "theta_x": theta_x[frame],
            "theta_y": theta_y[frame],
            "x_bins": bins_x,
            "x_counts": counts_x,
            "y_bins": bins_y,
            "y_counts": counts_y,
            "paths": path[frame]  # Stocker les chemins sous forme de liste
        })
    append_to_parquet(data_list, bin_path)
    data_list.clear()  # Réinitialiser le buffer
    
    return marker,

# Création de l'animation
ani = animation.FuncAnimation(fig, update, frames=N, interval=500, blit=False)

# Sauvegarde en GIF dans ../data/
gif_path = os.path.join(data_dir, "ray_tracing.gif")
ani.save(gif_path, writer="pillow", fps=5)

print(f"\nGIF enregistré dans : {gif_path}")
