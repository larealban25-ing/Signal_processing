"""
Module io.py : générateur d'ECG synthétique

Ce module crée un signal ECG artificiel pour tester nos algorithmes
sans avoir besoin de vraies données patient.
"""

import numpy as np


def synthetic_ecg(duration_s=60, fs=250, heart_rate=60):
    """
    Génère un signal ECG synthétique simple.
    
    Paramètres:
        duration_s : durée du signal en secondes (ex: 60 = 1 minute)
        fs : fréquence d'échantillonnage en Hz (ex: 250 Hz = 250 points/seconde)
        heart_rate : fréquence cardiaque moyenne en bpm (battements par minute)
    
    Retourne:
        ecg : array NumPy 1D du signal ECG (amplitude en microvolts)
        fs : fréquence d'échantillonnage (on la retourne pour ne pas l'oublier)
    
    Ce qu'on simule:
    - Des R-peaks (les pics principaux de l'ECG) à intervalles réguliers
    - Du bruit gaussien (électrodes, muscle, etc.)
    - Une baseline wander (dérive lente du signal, respiration)
    """
    
    # ========== ÉTAPE 1: Préparer les échantillons et le temps ==========
    
    # Calculer combien d'échantillons on a au total
    # Si on veut 10 secondes à 250 Hz → 10 * 250 = 2500 points
    n_samples = int(duration_s * fs)
    
    # Créer un vecteur temps: [0, 1/fs, 2/fs, ..., duration_s]
    # Exemple: si fs=250, on a [0, 0.004, 0.008, ..., 9.996] pour 10 secondes
    t = np.arange(n_samples) / fs
    
    
    # ========== ÉTAPE 2: Calculer où placer les R-peaks ==========
    
    # L'intervalle RR = temps entre deux battements (en secondes)
    # Formule: RR = 60 / heart_rate
    # Exemple: si HR = 60 bpm → RR = 60/60 = 1 seconde entre chaque battement
    # Exemple: si HR = 80 bpm → RR = 60/80 = 0.75 seconde
    rr_mean = 60.0 / heart_rate
    
    # Combien de battements on aura dans toute la durée?
    # Exemple: 60 secondes avec RR=1s → 60 battements
    n_beats = int(duration_s / rr_mean)
    
    # Placer les R-peaks dans le temps
    # On commence à 0.5s (décalage initial) puis on espace de rr_mean
    # Exemple: [0.5, 1.5, 2.5, 3.5, ...] si rr_mean=1
    r_peak_times = np.arange(n_beats) * rr_mean + 0.5
    
    # Convertir les temps (secondes) en indices d'échantillons
    # Exemple: 0.5s * 250 Hz = échantillon 125
    r_peak_samples = (r_peak_times * fs).astype(int)
    
    # Garder seulement les R-peaks qui tombent dans notre signal
    # (au cas où on dépasse la durée)
    r_peak_samples = r_peak_samples[r_peak_samples < n_samples]
    
    
    # ========== ÉTAPE 3: Initialiser le signal avec du bruit ==========
    
    # Créer un signal de bruit gaussien (moyenne=0, écart-type=0.1)
    # np.random.randn génère des valeurs aléatoires (distribution normale)
    # On multiplie par 0.1 pour avoir un petit bruit
    ecg = 0.1 * np.random.randn(n_samples)
    
    
    # ========== ÉTAPE 4: Ajouter les R-peaks (forme gaussienne) ==========
    
    # Pour chaque position de R-peak, on ajoute une gaussienne (courbe en cloche)
    for r_sample in r_peak_samples:
        
        # Largeur du pic: ~40 ms (millisecondes)
        # À 250 Hz, 0.04s * 250 = 10 échantillons
        width = int(0.04 * fs)
        
        # Créer un intervalle autour du R-peak
        # De -3*width à +3*width (pour capturer toute la gaussienne)
        x = np.arange(-3*width, 3*width)
        
        # Formule gaussienne: exp(-0.5 * (x/width)²)
        # Plus x est loin de 0, plus la valeur est faible
        # width contrôle la largeur de la courbe
        gaussian = np.exp(-0.5 * (x / width) ** 2)
        
        # Calculer les positions absolues dans le signal
        # Exemple: si r_sample=250 et x=[-30, -29, ..., 29, 30]
        # alors pos = [220, 221, ..., 279, 280]
        pos = r_sample + x
        
        # Vérifier que les positions sont valides (dans les limites du signal)
        mask = (pos >= 0) & (pos < n_samples)
        
        # Ajouter la gaussienne au signal (amplitude = 1.0)
        # On utilise le mask pour éviter de sortir des limites
        ecg[pos[mask]] += 1.0 * gaussian[mask]
    
    
    # ========== ÉTAPE 5: Ajouter une baseline wander (dérive) ==========
    
    # Simuler une oscillation lente (respiration, mouvement)
    # Fréquence: 0.33 Hz = environ 20 cycles par minute (respiration normale)
    # Amplitude: 0.2 µV
    # Formule sinus: amplitude * sin(2π * fréquence * temps)
    baseline = 0.2 * np.sin(2 * np.pi * 0.33 * t)
    
    # Ajouter la baseline au signal
    ecg += baseline
    
    
    # ========== ÉTAPE 6: Retourner le signal et la fréquence ==========
    
    return ecg, fs
