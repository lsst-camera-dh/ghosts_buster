#!/bin/bash
# -*- coding: utf-8 -*-

# SLURM options:

# SBATCH --job-name=fitrotationbatoid    # Nom du job
# SBATCH --output=fitrotationbatoid_%j.log   # Standard output et error log

# SBATCH --partition=htc               # Choix de partition (htc par défaut)

# SBATCH --ntasks=1                    # Exécuter une seule tâche
# SBATCH --mem=60000                  # Mémoire en MB par défaut
# SBATCH --time=1-00:00:00             # Délai max = 7 jours

# SBATCH --mail-user=dimitri.buffat@etu.univ-lyon1.fr         # Où envoyer l'e-mail
# SBATCH --mail-type=END,FAIL          # Événements déclencheurs (NONE, BEGIN, END, FAIL, ALL)

# SBATCH --licenses=sps                # Déclaration des ressources de stockage et/ou logicielles

# Commandes à soumettre :

source /cvmfs/sw.lsst.eu/almalinux-x86_64/lsst_distrib/w_2025_18/loadLSST.bash
setup lsst_distrib

# Lancer le script Python
python3 $HOME/lsst/scripts/fitrot.py
