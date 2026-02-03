"""
Module features.py : extraction de features HRV (variabilité fréquence cardiaque)

Ce module calcule la puissance spectrale (LF/HF) à partir des intervalles RR.
Deux méthodes disponibles:
1. Welch (recommandé) → moyenne de FFT sur fenêtres glissantes
2. FFT directe → transformée de Fourier classique

Équivalences MATLAB → Python:
- pwelch() → scipy.signal.welch()
- fft() → numpy.fft.fft()
"""

import numpy as np
from scipy import signal
from scipy.interpolate import interp1d


def compute_rr_intervals(r_peaks, fs):
    """
    Calcule les intervalles RR (temps entre battements) à partir des R-peaks.
    
    Paramètres:
        r_peaks : array des indices des R-peaks (résultat de detect_r_peaks)
        fs : fréquence d'échantillonnage en Hz
    
    Retourne:
        rr_seconds : array des intervalles RR en secondes
        rr_times : array des temps médians de chaque intervalle RR
    
    Exemple:
        Si r_peaks = [250, 500, 750] et fs=250 Hz
        → rr_samples = [250, 250] (différences)
        → rr_seconds = [1.0, 1.0] (250/250 = 1 seconde)
        → rr_times = [1.5, 2.5] (milieu de chaque intervalle)
    """
    
    if len(r_peaks) < 2:
        return np.array([]), np.array([])
    
    # ========== ÉTAPE 1: Calculer les intervalles en échantillons ==========
    
    # np.diff() calcule les différences successives
    # Exemple: [250, 500, 750] → [250, 250]
    rr_samples = np.diff(r_peaks)
    
    
    # ========== ÉTAPE 2: Convertir en secondes ==========
    
    # Diviser par la fréquence d'échantillonnage
    # Exemple: 250 échantillons / 250 Hz = 1.0 seconde
    rr_seconds = rr_samples / float(fs)
    
    
    # ========== ÉTAPE 3: Calculer les temps médians ==========
    
    # Pour chaque intervalle RR, on calcule le temps au milieu
    # Utile pour l'interpolation ensuite
    #
    # Temps cumulatif des R-peaks (en secondes)
    r_peak_times = r_peaks / float(fs)
    
    # Temps médian de chaque intervalle
    # Exemple: si R-peaks à [1.0s, 2.0s, 3.0s]
    # → intervalles entre [1.0-2.0] et [2.0-3.0]
    # → temps médians = [1.5s, 2.5s]
    rr_times = (r_peak_times[:-1] + r_peak_times[1:]) / 2.0
    
    return rr_seconds, rr_times


def interpolate_rr(rr_seconds, rr_times, fs_interp=4.0):
    """
    Interpole les intervalles RR pour créer un tachogramme régulier.
    
    Paramètres:
        rr_seconds : array des intervalles RR en secondes
        rr_times : array des temps médians de chaque RR
        fs_interp : fréquence d'échantillonnage du tachogramme (Hz)
                    Recommandé: 4 Hz (4 points/seconde)
    
    Retourne:
        t_interp : vecteur temps interpolé (échantillonnage régulier)
        rr_interp : tachogramme interpolé (valeurs RR à chaque instant)
    
    Pourquoi interpoler?
    - Les RR sont irréguliers dans le temps (varient selon la HR)
    - Pour calculer la FFT/Welch, on a besoin d'échantillonnage régulier
    - L'interpolation crée une série temporelle uniforme
    
    Équivalent MATLAB:
        t_interp = 0:1/fs_interp:max(rr_times);
        rr_interp = interp1(rr_times, rr_seconds, t_interp, 'spline');
    """
    
    if len(rr_seconds) < 3:
        # Pas assez de points pour interpolation fiable
        return np.array([]), np.array([])
    
    
    # ========== ÉTAPE 1: Créer le vecteur temps régulier ==========
    
    # De 0 jusqu'au dernier temps RR, avec pas = 1/fs_interp
    # Exemple: si fs_interp=4 Hz et durée=10s
    # → t_interp = [0, 0.25, 0.5, 0.75, ..., 10.0]
    t_interp = np.arange(0, rr_times[-1], 1.0 / fs_interp)
    
    
    # ========== ÉTAPE 2: Interpolation cubique ==========
    
    # interp1d crée une fonction d'interpolation
    # kind='cubic' → spline cubique (lisse, préserve les variations)
    # Alternatives: 'linear', 'quadratic', 'nearest'
    #
    # fill_value='extrapolate' → prolonge aux extrémités si besoin
    interpolator = interp1d(
        rr_times, 
        rr_seconds, 
        kind='cubic', 
        fill_value='extrapolate',
        bounds_error=False
    )
    
    # Évaluer l'interpolation sur le vecteur temps régulier
    rr_interp = interpolator(t_interp)
    
    return t_interp, rr_interp


def compute_psd_welch(rr_interp, fs_interp=4.0, nperseg=256):
    """
    Calcule la densité spectrale de puissance (PSD) avec la méthode de Welch.
    
    Paramètres:
        rr_interp : tachogramme interpolé
        fs_interp : fréquence d'échantillonnage du tachogramme
        nperseg : longueur de chaque segment (fenêtre)
                  Plus grand → meilleure résolution fréquentielle
                  Plus petit → meilleur moyennage (moins de bruit)
    
    Retourne:
        freqs : array des fréquences (Hz)
        psd : array de la puissance à chaque fréquence (s²/Hz)
    
    Méthode de Welch:
    1. Découpe le signal en segments qui se chevauchent
    2. Applique une fenêtre (Hanning) à chaque segment
    3. Calcule la FFT de chaque segment
    4. Moyenne les FFT → réduit le bruit
    
    Équivalent MATLAB:
        [psd, freqs] = pwelch(rr_interp, hanning(nperseg), [], [], fs_interp);
    """
    
    # Détrend (enlever tendance linéaire) avant PSD
    # Évite les pics parasites à basse fréquence
    rr_detrend = signal.detrend(rr_interp)
    
    
    # ========== Welch avec scipy.signal.welch() ==========
    
    # Paramètres:
    #   rr_detrend : signal d'entrée
    #   fs : fréquence d'échantillonnage
    #   nperseg : nombre de points par segment
    #   window : type de fenêtre (défaut: Hanning)
    #   noverlap : chevauchement (défaut: nperseg/2)
    #
    # Retourne:
    #   freqs : vecteur fréquences
    #   psd : puissance à chaque fréquence
    freqs, psd = signal.welch(
        rr_detrend,
        fs=fs_interp,
        nperseg=min(nperseg, len(rr_detrend)),  # éviter nperseg > longueur signal
        scaling='density'  # PSD en unités²/Hz
    )
    
    return freqs, psd


def compute_psd_fft(rr_interp, fs_interp=4.0):
    """
    Calcule la densité spectrale de puissance (PSD) avec FFT directe.
    
    Paramètres:
        rr_interp : tachogramme interpolé
        fs_interp : fréquence d'échantillonnage du tachogramme
    
    Retourne:
        freqs : array des fréquences (Hz)
        psd : array de la puissance à chaque fréquence (s²/Hz)
    
    FFT directe (moins robuste que Welch, mais plus simple):
    1. Détrend le signal
    2. Applique une fenêtre (Hanning)
    3. Calcule la FFT
    4. Prend le module au carré (puissance)
    5. Normalise
    
    Équivalent MATLAB:
        Y = fft(detrend(rr_interp) .* hanning(length(rr_interp)));
        psd = abs(Y).^2 / (fs_interp * length(rr_interp));
        freqs = (0:length(Y)-1) * fs_interp / length(Y);
    """
    
    # Détrend
    rr_detrend = signal.detrend(rr_interp)
    
    
    # ========== ÉTAPE 1: Appliquer fenêtre de Hanning ==========
    
    # Fenêtre de Hanning réduit les fuites spectrales (leakage)
    # C'est une courbe en cloche qui atténue les bords du signal
    window = np.hanning(len(rr_detrend))
    rr_windowed = rr_detrend * window
    
    
    # ========== ÉTAPE 2: Calculer la FFT ==========
    
    # np.fft.fft() → transformée de Fourier rapide
    # Convertit signal temporel → spectre fréquentiel
    fft_result = np.fft.fft(rr_windowed)
    
    
    # ========== ÉTAPE 3: Calculer la puissance ==========
    
    # Puissance = |FFT|² (module au carré)
    # np.abs() donne le module d'un nombre complexe
    power = np.abs(fft_result) ** 2
    
    # Normalisation pour obtenir une PSD (densité spectrale)
    # Diviser par (fs * N) où N = longueur du signal
    N = len(rr_detrend)
    psd = power / (fs_interp * N)
    
    
    # ========== ÉTAPE 4: Créer le vecteur fréquences ==========
    
    # Les fréquences vont de 0 à fs_interp avec pas = fs_interp/N
    # Exemple: si fs=4 Hz et N=1000
    # → freqs = [0, 0.004, 0.008, ..., 3.996] Hz
    freqs = np.fft.fftfreq(N, d=1.0/fs_interp)
    
    
    # ========== ÉTAPE 5: Garder seulement les fréquences positives ==========
    
    # La FFT retourne fréquences négatives + positives (symétrique)
    # On garde seulement la moitié positive
    positive_freqs = freqs >= 0
    freqs = freqs[positive_freqs]
    psd = psd[positive_freqs]
    
    return freqs, psd


def compute_lf_hf(freqs, psd, lf_band=(0.04, 0.15), hf_band=(0.15, 0.4)):
    """
    Calcule les puissances LF et HF à partir d'une PSD.
    
    Paramètres:
        freqs : vecteur fréquences (Hz)
        psd : densité spectrale de puissance (s²/Hz)
        lf_band : bande basses fréquences (défaut: 0.04-0.15 Hz)
        hf_band : bande hautes fréquences (défaut: 0.15-0.4 Hz)
    
    Retourne:
        results : dict avec:
            - lf : puissance LF absolue (ms²)
            - hf : puissance HF absolue (ms²)
            - lf_hf : ratio LF/HF
            - lf_nu : LF normalisée (%)
            - hf_nu : HF normalisée (%)
            - total : puissance totale (ms²)
    
    Interprétation physiologique:
    - LF (0.04-0.15 Hz) : activité sympathique + parasympathique
    - HF (0.15-0.4 Hz) : activité parasympathique (respiration)
    - LF/HF : balance sympatho-vagale (stress vs relax)
    """
    
    # ========== ÉTAPE 1: Intégrer la puissance dans chaque bande ==========
    
    # Masque pour sélectionner les fréquences dans la bande LF
    lf_mask = (freqs >= lf_band[0]) & (freqs <= lf_band[1])
    
    # Masque pour sélectionner les fréquences dans la bande HF
    hf_mask = (freqs >= hf_band[0]) & (freqs <= hf_band[1])
    
    # Intégration par la méthode des trapèzes
    # np.trapz() calcule l'aire sous la courbe
    # C'est l'équivalent de l'intégrale: ∫ PSD(f) df
    lf_power = np.trapz(psd[lf_mask], freqs[lf_mask]) if np.any(lf_mask) else 0.0
    hf_power = np.trapz(psd[hf_mask], freqs[hf_mask]) if np.any(hf_mask) else 0.0
    
    # Puissance totale (toutes les fréquences)
    total_power = np.trapz(psd, freqs)
    
    
    # ========== ÉTAPE 2: Convertir en ms² (convention HRV) ==========
    
    # Les intervalles RR sont en secondes, donc puissance en s²
    # Convention: exprimer en ms² (1 s² = 1,000,000 ms²)
    lf_power_ms2 = lf_power * 1000 * 1000
    hf_power_ms2 = hf_power * 1000 * 1000
    total_power_ms2 = total_power * 1000 * 1000
    
    
    # ========== ÉTAPE 3: Calculer le ratio LF/HF ==========
    
    lf_hf_ratio = lf_power / hf_power if hf_power > 0 else np.inf
    
    
    # ========== ÉTAPE 4: Calculer les puissances normalisées ==========
    
    # Puissance normalisée (n.u. = normalized units)
    # LF_nu = LF / (LF + HF) × 100
    # HF_nu = HF / (LF + HF) × 100
    #
    # Avantage: indépendant de la puissance totale
    # Permet comparaison entre sujets/conditions
    total_lf_hf = lf_power + hf_power
    lf_nu = 100.0 * lf_power / total_lf_hf if total_lf_hf > 0 else 0.0
    hf_nu = 100.0 * hf_power / total_lf_hf if total_lf_hf > 0 else 0.0
    
    
    # ========== ÉTAPE 5: Retourner les résultats ==========
    
    results = {
        'lf': float(lf_power_ms2),
        'hf': float(hf_power_ms2),
        'total': float(total_power_ms2),
        'lf_hf': float(lf_hf_ratio),
        'lf_nu': float(lf_nu),
        'hf_nu': float(hf_nu)
    }
    
    return results


# ========== Pipeline complet: RR → LF/HF ==========

def extract_hrv_features(r_peaks, fs, method='welch', fs_interp=4.0):
    """
    Pipeline complet: R-peaks → intervalles RR → PSD → LF/HF.
    
    Paramètres:
        r_peaks : array des indices des R-peaks
        fs : fréquence d'échantillonnage du signal ECG
        method : 'welch' (recommandé) ou 'fft'
        fs_interp : fréquence d'interpolation du tachogramme (Hz)
    
    Retourne:
        results : dict avec lf, hf, lf_hf, lf_nu, hf_nu, total
    
    Exemple d'utilisation:
        peaks = detect_r_peaks(ecg_clean, fs)
        hrv = extract_hrv_features(peaks, fs, method='welch')
        print(f"LF/HF ratio: {hrv['lf_hf']:.2f}")
    """
    
    # Étape 1: Calculer RR
    rr_sec, rr_times = compute_rr_intervals(r_peaks, fs)
    
    if len(rr_sec) < 10:
        # Pas assez de battements pour HRV fiable
        return {'lf': 0, 'hf': 0, 'total': 0, 'lf_hf': np.nan, 'lf_nu': 0, 'hf_nu': 0}
    
    # Étape 2: Interpoler
    t_interp, rr_interp = interpolate_rr(rr_sec, rr_times, fs_interp)
    
    if len(rr_interp) < 50:
        return {'lf': 0, 'hf': 0, 'total': 0, 'lf_hf': np.nan, 'lf_nu': 0, 'hf_nu': 0}
    
    # Étape 3: Calculer PSD
    if method == 'welch':
        freqs, psd = compute_psd_welch(rr_interp, fs_interp)
    elif method == 'fft':
        freqs, psd = compute_psd_fft(rr_interp, fs_interp)
    else:
        raise ValueError(f"Méthode inconnue: {method}. Utilisez 'welch' ou 'fft'.")
    
    # Étape 4: Calculer LF/HF
    results = compute_lf_hf(freqs, psd)
    
    return results
