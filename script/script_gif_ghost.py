import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
import os

one_ghost = False
'''
Si l'option au dessus est False, cela signifie qu'il existe des groupes x_data, y_data qui possède les memes theta_x, theta_y.
Dans ce cas, je souhaite tous les avoir sur la meme image.
'''

# Charger les données depuis le CSV
csv_path = "../data/ray_data.csv"
df = pd.read_csv(csv_path)

# Séparer les données
x_data = [list(map(float, x.split(','))) for x in df["x_data"]]
y_data = [list(map(float, y.split(','))) for y in df["y_data"]]
theta_x = df["theta_x"].tolist()
theta_y = df["theta_y"].tolist()

'''
Rajouter ici un subplot pour le couple theta_x, theta_y ayant le plus de data (de ghost)
Subplot car on va tracer tout les ghosts de maniere separer
Appeler chaque subplot par une lettre et donner le path associé en description sous la figure
'''

if not one_ghost:
    from collections import defaultdict

    grouped_data = defaultdict(lambda: ([], []))  # Dictionnaire qui stocke les listes de x et y

    for i in range(len(theta_x)):
        key = (theta_x[i], theta_y[i])  # Utilisation des angles comme clé
        grouped_data[key][0].extend(x_data[i])  # Ajouter les x
        grouped_data[key][1].extend(y_data[i])  # Ajouter les y

    # Convertir les clés en listes
    theta_x, theta_y = zip(*grouped_data.keys())  # Séparer les angles en deux listes distinctes
    x_data = [grouped_data[key][0] for key in grouped_data.keys()]  # X fusionnés
    y_data = [grouped_data[key][1] for key in grouped_data.keys()]  # Y fusionnés
    
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

# Fonction d'animation
def update(frame):
    # Supprimer l'ancien hexbin
    if hexbin_collection[0] is not None:
        hexbin_collection[0].remove()
    
    # Ajouter la nouvelle hexbin
    hexbin_collection[0] = ax.hexbin(x_data[frame], y_data[frame], extent=[-0.35, 0.35, -0.35, 0.35], gridsize=500, norm=colors.AsinhNorm())
    marker.set_data([np.mean(x_data[frame])], [np.mean(y_data[frame])])
    ax.set_title(f"Ray Tracing - θx: {theta_x[frame]:.2f}°, θy: {theta_y[frame]:.2f}°")

    '''
    Rajouter ici un plt.save avec comme nom le couple theta_x theta_y
    Rajouter donc une methode pour lechemin comme pour le gif
    Dans ../data/images/(theta_x:{theta_x}, theta_y:{theta_y}).png
    '''
    
    '''
    Rajouter ensuite une methode pour recuperer les hexbins sur les axes x et y du marker
    Une fois recuperer, on fera un histogramme du nombre de photon suivant x et y (subplot (2,1))
    '''
    return marker,

# Création de l'animation
ani = animation.FuncAnimation(fig, update, frames=N, interval=200, blit=False)

# Vérifier si le dossier ../data/ existe, sinon le créer
save_dir = "../data/"
os.makedirs(save_dir, exist_ok=True)

# Sauvegarde en GIF dans ../data/
gif_path = os.path.join(save_dir, "ray_tracing.gif")
ani.save(gif_path, writer="pillow", fps=5)

print(f"GIF enregistré dans : {gif_path}")
