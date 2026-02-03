"""
Module preprocess.py : filtrage du signal ECG

Ce module contient les fonctions de prétraitement (filtrage) pour nettoyer
le signal ECG avant la détection des R-peaks.

Équivalences MATLAB → Python:
- butter() → scipy.signal.butter()
- filtfilt() → scipy.signal.filtfilt()
- iirnotch() → scipy.signal.iirnotch()
"""

import numpy as np
from scipy import signal


def bandpass_filter(ecg, fs, low=0.5, high=40.0, order=4):
    """
    Applique un filtre passe-bande Butterworth sur le signal ECG.
    
    Paramètres:
        ecg : signal ECG brut (array NumPy 1D)
        fs : fréquence d'échantillonnage en Hz (ex: 250 Hz)
        low : fréquence de coupure basse en Hz (défaut: 0.5 Hz)
        high : fréquence de coupure haute en Hz (défaut: 40 Hz)
        order : ordre du filtre (défaut: 4)
    
    Retourne:
        ecg_filtered : signal ECG filtré
    
    Pourquoi ce filtre?
    - Enlève les basses fréquences (<0.5 Hz) → baseline wander (respiration)
    - Enlève les hautes fréquences (>40 Hz) → bruit musculaire, électrodes
    - Garde la bande utile de l'ECG (complexe QRS entre 5-15 Hz)
    
    Équivalent MATLAB:
        [b, a] = butter(4, [0.5 40]/(fs/2), 'bandpass');
        ecg_filtered = filtfilt(b, a, ecg);
    """
    
    # ========== ÉTAPE 1: Calculer la fréquence de Nyquist ==========
    
    # Fréquence de Nyquist = moitié de la fréquence d'échantillonnage
    # C'est la fréquence maximale qu'on peut représenter
    # Exemple: si fs=250 Hz → nyquist = 125 Hz
    nyquist = 0.5 * fs
    
    
    # ========== ÉTAPE 2: Normaliser les fréquences de coupure ==========
    
    # Les filtres numériques travaillent avec des fréquences normalisées
    # (entre 0 et 1, où 1 = fréquence de Nyquist)
    # Exemple: si low=0.5 Hz et nyquist=125 Hz → low_norm = 0.5/125 = 0.004
    low_norm = low / nyquist
    high_norm = high / nyquist
    
    
    # ========== ÉTAPE 3: Calculer les coefficients du filtre ==========
    
    # butter() crée un filtre Butterworth (réponse plate dans la bande passante)
    # order : ordre du filtre (plus élevé = coupure plus raide, mais risque d'instabilité)
    # [low_norm, high_norm] : bande passante normalisée
    # btype='band' : filtre passe-bande (vs 'low', 'high', 'stop')
    #
    # Retourne:
    #   b : coefficients du numérateur (feedforward)
    #   a : coefficients du dénominateur (feedback)
    b, a = signal.butter(order, [low_norm, high_norm], btype='band')
    
    
    # ========== ÉTAPE 4: Appliquer le filtre ==========
    
    # filtfilt() applique le filtre dans les deux sens (avant + arrière)
    # Avantages:
    #   - Pas de déphasage (phase nulle)
    #   - Pente de coupure doublée (ordre effectif = 2*order)
    # C'est l'équivalent direct de filtfilt() en MATLAB
    ecg_filtered = signal.filtfilt(b, a, ecg)
    
    return ecg_filtered


def notch_filter(ecg, fs, freq=50.0, quality=30.0):
    """
    Applique un filtre notch (réjecteur) pour enlever une fréquence spécifique.
    
    Paramètres:
        ecg : signal ECG (array NumPy 1D)
        fs : fréquence d'échantillonnage en Hz
        freq : fréquence à rejeter en Hz (défaut: 50 Hz pour Europe, 60 Hz pour USA)
        quality : facteur de qualité Q (défaut: 30)
                  Plus Q est élevé, plus la bande rejetée est étroite
    
    Retourne:
        ecg_notched : signal ECG avec la fréquence rejetée
    
    Pourquoi ce filtre?
    - Enlève l'interférence du secteur (ligne électrique)
    - En Europe: 50 Hz
    - Aux USA/Canada: 60 Hz
    
    Équivalent MATLAB:
        [b, a] = iirnotch(50/(fs/2), 50/(fs/2)/35);
        ecg_notched = filtfilt(b, a, ecg);
    """
    
    # ========== ÉTAPE 1: Normaliser la fréquence à rejeter ==========
    
    # Même principe que pour le bandpass: normaliser par rapport à Nyquist
    # Exemple: si freq=50 Hz et fs=250 Hz → freq_norm = 50/125 = 0.4
    nyquist = 0.5 * fs
    freq_norm = freq / nyquist
    
    
    # ========== ÉTAPE 2: Créer le filtre notch IIR ==========
    
    # iirnotch() crée un filtre réjecteur de bande (notch filter)
    # freq_norm : fréquence centrale normalisée
    # quality : facteur Q (largeur de bande = freq/Q)
    #
    # Exemple: freq=50 Hz, Q=30 → largeur de bande ≈ 1.67 Hz
    # (rejette de ~49.2 à ~50.8 Hz)
    b, a = signal.iirnotch(freq_norm, quality)
    
    
    # ========== ÉTAPE 3: Appliquer le filtre (bidirectionnel) ==========
    
    # Utiliser filtfilt() pour éviter le déphasage
    ecg_notched = signal.filtfilt(b, a, ecg)
    
    return ecg_notched


# ========== Fonction bonus: pipeline de prétraitement complet ==========

def preprocess_ecg(ecg, fs, apply_bandpass=True, apply_notch=True, notch_freq=50.0):
    """
    Pipeline de prétraitement complet: bandpass + notch.
    
    Paramètres:
        ecg : signal ECG brut
        fs : fréquence d'échantillonnage
        apply_bandpass : appliquer le passe-bande? (défaut: True)
        apply_notch : appliquer le notch? (défaut: True)
        notch_freq : fréquence du notch (50 ou 60 Hz)
    
    Retourne:
        ecg_clean : signal ECG prétraité
    
    Ordre recommandé:
        1. Bandpass (enlever baseline + hautes fréquences)
        2. Notch (enlever secteur)
    """
    ecg_clean = ecg.copy()
    
    if apply_bandpass:
        ecg_clean = bandpass_filter(ecg_clean, fs, low=0.5, high=40.0)
    
    if apply_notch:
        ecg_clean = notch_filter(ecg_clean, fs, freq=notch_freq)
    
    return ecg_clean
