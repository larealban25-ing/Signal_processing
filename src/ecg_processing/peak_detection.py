"""
Module peak_detection.py : détection des R-peaks dans un signal ECG

Les R-peaks sont les pics principaux de l'ECG (dépolarisation ventriculaire).
Leur détection est essentielle pour calculer:
- Les intervalles RR (temps entre battements)
- La fréquence cardiaque
- La variabilité de la fréquence cardiaque (HRV)

Équivalence MATLAB → Python:
- findpeaks() → scipy.signal.find_peaks()
"""

import numpy as np
from scipy.signal import find_peaks


def detect_r_peaks(ecg, fs, method='simple', min_distance_ms=200, prominence_factor=2):
    """
    Détecte les R-peaks dans un signal ECG prétraité.
    
    Paramètres:
        ecg : signal ECG prétraité (array NumPy 1D)
              ⚠️ Le signal doit être filtré avant (bandpass + notch)
        fs : fréquence d'échantillonnage en Hz (ex: 250 Hz)
        method : méthode de détection (défaut: 'simple')
                 'simple' → utilise scipy.signal.find_peaks
        min_distance_ms : distance minimale entre deux R-peaks en millisecondes
                          (défaut: 200 ms = fréquence cardiaque max ~300 bpm)
        prominence_factor : facteur pour la proéminence minimale
                            (défaut: 0.5 → prominence = 0.5 * std(ecg))
    
    Retourne:
        r_peaks : array NumPy contenant les indices des R-peaks
                  (positions en échantillons, pas en temps)
    
    Équivalent MATLAB:
        [pks, locs] = findpeaks(ecg, 'MinPeakDistance', 0.2*fs, ...
                                     'MinPeakProminence', 0.5*std(ecg));
        r_peaks = locs;
    
    Exemple:
        ecg_clean = preprocess_ecg(ecg_raw, fs)
        peaks = detect_r_peaks(ecg_clean, fs)
        # peaks contient [125, 375, 625, 875, ...] (indices d'échantillons)
    """
    
    # ========== ÉTAPE 1: Calculer la distance minimale en échantillons ==========
    
    # Convertir la distance minimale de millisecondes en nombre d'échantillons
    # Exemple: min_distance_ms=200 ms, fs=250 Hz
    #          → min_distance = 0.2 * 250 = 50 échantillons
    # 
    # Pourquoi 200 ms?
    # - Fréquence cardiaque normale: 60-100 bpm → RR = 600-1000 ms
    # - Fréquence cardiaque max physiologique: ~220 bpm → RR = 273 ms
    # - 200 ms évite les fausses détections (onde T, bruit)
    min_distance = int((min_distance_ms / 1000.0) * fs)
    
    
    # ========== ÉTAPE 2: Calculer la proéminence minimale ==========
    
    # La proéminence = hauteur du pic par rapport aux vallées environnantes
    # Un R-peak doit se démarquer clairement du bruit et des autres ondes
    #
    # prominence_min = prominence_factor * écart-type du signal
    # Exemple: si std(ecg)=0.8 et prominence_factor=0.5 → prominence_min=0.4
    #
    # Pourquoi std(ecg)?
    # - S'adapte automatiquement à l'amplitude du signal
    # - Plus robuste que seuil absolu (varie selon patient/électrodes)
    prominence_min = prominence_factor * np.std(ecg)
    
    
    # ========== ÉTAPE 3: Détecter les pics avec find_peaks() ==========
    
    # find_peaks() trouve tous les maxima locaux qui respectent les contraintes
    #
    # Paramètres:
    #   ecg : le signal (array 1D)
    #   distance : nombre d'échantillons minimum entre deux pics
    #   prominence : hauteur minimale du pic par rapport aux vallées
    #   height : (optionnel) seuil absolu minimum (on ne l'utilise pas ici)
    #
    # Retourne:
    #   peaks : array des indices où se trouvent les pics
    #   properties : dict avec infos sur chaque pic (hauteur, largeur, etc.)
    #
    # Équivalent MATLAB exact:
    #   [pks, locs] = findpeaks(ecg, 'MinPeakDistance', min_distance, ...
    #                                 'MinPeakProminence', prominence_min);
    peaks, properties = find_peaks(
        ecg,
        distance=min_distance,
        prominence=prominence_min
    )
    
    
    # ========== ÉTAPE 4: Vérification et retour ==========
    
    # Vérifier qu'on a trouvé au moins quelques pics
    # (si aucun pic trouvé, le signal est peut-être mal prétraité)
    if len(peaks) == 0:
        print(" Attention: aucun R-peak détecté!")
        print("   Vérifiez que le signal est bien prétraité (bandpass + notch)")
        print(f"   Paramètres: min_distance={min_distance}, prominence={prominence_min:.3f}")
    
    return peaks


# ========== Fonction bonus: calculer la fréquence cardiaque ==========

def compute_heart_rate(r_peaks, fs):
    """
    Calcule la fréquence cardiaque moyenne à partir des R-peaks.
    
    Paramètres:
        r_peaks : array des indices des R-peaks (résultat de detect_r_peaks)
        fs : fréquence d'échantillonnage en Hz
    
    Retourne:
        hr_mean : fréquence cardiaque moyenne en bpm (battements par minute)
        hr_std : écart-type de la fréquence cardiaque en bpm
    
    Formule:
        RR (secondes) = diff(r_peaks) / fs
        HR (bpm) = 60 / RR
    """
    
    if len(r_peaks) < 2:
        return 0.0, 0.0
    
    # Calculer les intervalles RR en échantillons
    rr_samples = np.diff(r_peaks)
    
    # Convertir en secondes
    rr_seconds = rr_samples / fs
    
    # Convertir en bpm (battements par minute)
    # HR = 60 / RR
    # Exemple: si RR = 1 seconde → HR = 60 bpm
    #          si RR = 0.75 seconde → HR = 80 bpm
    hr_instantaneous = 60.0 / rr_seconds
    
    # Calculer moyenne et écart-type
    hr_mean = np.mean(hr_instantaneous)
    hr_std = np.std(hr_instantaneous)
    
    return hr_mean, hr_std
