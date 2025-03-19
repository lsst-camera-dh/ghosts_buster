import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil 
import sys
import os

# Vérifier si les dossiers ../data/ et ../data/images existe, sinon les créer
data_dir = "../data/"
hists_dir = "../data/hists/"

os.makedirs(data_dir, exist_ok=True)

if os.path.exists(hists_dir):
    shutil.rmtree(hists_dir)
os.makedirs(hists_dir)

# Charger les données depuis le CSV
bin_path = "../data/bin_data.parquet"
df = pd.read_parquet(bin_path)

Nimg = df.shape[0]
print(f"{Nimg} données chargées")

# Conversion des colonnes contenant des listes
df["x_bins"] = df["x_bins"].apply(lambda x: list(x) if isinstance(x, (list, np.ndarray)) else eval(x))
df["x_counts"] = df["x_counts"].apply(lambda x: list(x) if isinstance(x, (list, np.ndarray)) else eval(x))
df["y_bins"] = df["y_bins"].apply(lambda y: list(y) if isinstance(y, (list, np.ndarray)) else eval(y))
df["y_counts"] = df["y_counts"].apply(lambda y: list(y) if isinstance(y, (list, np.ndarray)) else eval(y))

# Séparer les données
x_bins = df["x_bins"].tolist()
x_counts = df["x_counts"].tolist()
y_bins = df["y_bins"].tolist()
y_counts = df["y_counts"].tolist()
theta_x = df["theta_x"].tolist()
theta_y = df["theta_y"].tolist()
paths = df["paths"].tolist()

# Affichage des histogrammes
for i in range(Nimg):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    
    ax[0].bar(x_bins[i], x_counts[i], width=0.001, color='blue')
    ax[0].set_title("Histogramme des bins alignés sur y moyen")
    ax[0].set_yscale('log')
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("Nombre de rayons")
    
    ax[1].bar(y_bins[i], y_counts[i], width=0.01, color='green')
    ax[1].set_title("Histogramme des bins alignés sur x moyen")
    ax[1].set_yscale('log')
    ax[1].set_xlabel("y")
    ax[1].set_ylabel("Nombre de rayons")

    sys.stdout.write(f"\rProgress: {(i+1) / Nimg * 100:.2f} %")
    sys.stdout.flush()
    
    plt.tight_layout()
    hist_path = os.path.join(hists_dir, f"tx_{theta_x[i]:.2f}_ty_{theta_y[i]:.2f}.png")
    plt.savefig(hist_path)

print("\n Work is finish")
