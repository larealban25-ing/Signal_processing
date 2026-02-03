"""
Script pour télécharger et analyser des enregistrements ECG réels depuis PhysioNet.

Bases de données recommandées:
- MIT-BIH Arrhythmia Database (mitdb) : ECG avec arythmies
- MIT-BIH Normal Sinus Rhythm Database (nsrdb) : ECG normaux
"""

import sys
sys.path.insert(0, 'c:\\Users\\larea\\Documents\\Projet_Signal')

import os


def download_physionet_record(database='mitdb', record='100', destination='data'):
    """
    Télécharge un enregistrement depuis PhysioNet.
    
    Paramètres:
        database : nom de la base (ex: 'mitdb', 'nsrdb')
        record : numéro d'enregistrement (ex: '100', '16265')
        destination : dossier de destination
    
    Bases de données populaires:
    - mitdb : MIT-BIH Arrhythmia (enregistrements 100-234)
    - nsrdb : Normal Sinus Rhythm (enregistrements 16265, 16272, 16273, etc.)
    - apnea-ecg : Sleep Apnea (a01-a20, b01-b05, c01-c10)
    """
    try:
        import wfdb
    except ImportError:
        print(" Erreur: package 'wfdb' non installé.")
        print("Installation: pip install wfdb")
        return None
    
    # Créer le dossier de destination si inexistant
    db_path = os.path.join(destination, database)
    os.makedirs(db_path, exist_ok=True)
    
    print(f" Téléchargement de {database}/{record} depuis PhysioNet...")
    
    try:
        # Télécharger l'enregistrement avec wfdb.rdrecord en spécifiant pn_dir
        # Cela télécharge automatiquement les fichiers nécessaires
        record_name = f"{record}"
        
        # Télécharger dans le dossier local
        wfdb.rdrecord(record_name, pn_dir=database, sampfrom=0, sampto=1000)
        
        # Utiliser dl_database pour télécharger les fichiers localement
        print(f"   Téléchargement des fichiers...")
        wfdb.dl_database(database, db_path, records=[record])
        
        print(f" Téléchargement réussi dans {db_path}/")
        return os.path.join(db_path, record)
        
    except Exception as e:
        print(f" Téléchargement direct échoué: {e}")
        print(f" Lecture directe depuis PhysioNet (sans téléchargement local)...")
        # Retourner un chemin spécial qui indique de lire depuis PhysioNet
        return f"STREAM:{database}/{record}"


def load_physionet_ecg(record_path, channel=0, sampfrom=0, sampto=None):
    """
    Charge un enregistrement ECG PhysioNet.
    
    Paramètres:
        record_path : chemin vers l'enregistrement (sans extension) 
                      ou "STREAM:database/record" pour lecture directe
        channel : canal ECG à charger (0 ou 1 généralement)
        sampfrom : échantillon de départ
        sampto : échantillon de fin (None = tout)
    
    Retourne:
        ecg : signal ECG (array 1D)
        fs : fréquence d'échantillonnage
        info : dict avec métadonnées
    """
    try:
        import wfdb
    except ImportError:
        print(" Package 'wfdb' requis. Installation: pip install wfdb")
        return None, None, None
    
    try:
        # Vérifier si c'est un stream depuis PhysioNet
        if record_path.startswith("STREAM:"):
            # Format: STREAM:database/record
            parts = record_path.replace("STREAM:", "").split("/")
            database = parts[0]
            record_name = parts[1]
            
            print(f"   Lecture directe depuis PhysioNet (streaming)...")
            # Charger directement depuis PhysioNet sans téléchargement
            record = wfdb.rdrecord(record_name, pn_dir=database, 
                                  sampfrom=sampfrom, sampto=sampto)
        else:
            # Charger depuis fichier local
            record = wfdb.rdrecord(record_path, sampfrom=sampfrom, sampto=sampto)
        
        # Extraire le signal et la fréquence
        ecg = record.p_signal[:, channel]
        fs = record.fs
        
        # Informations utiles
        info = {
            'record_name': record.record_name,
            'fs': fs,
            'n_samples': len(ecg),
            'duration_s': len(ecg) / fs,
            'n_channels': record.n_sig,
            'channel_names': record.sig_name,
            'units': record.units[channel] if channel < len(record.units) else 'unknown'
        }
        
        return ecg, fs, info
        
    except Exception as e:
        print(f" Erreur lors du chargement: {e}")
        return None, None, None


# ========== Script principal ==========

if __name__ == "__main__":
    
    print("="*60)
    print("TÉLÉCHARGEMENT ET ANALYSE ECG PHYSIONET")
    print("="*60)
    
    # ========== Configuration ==========
    
    # Option 1: MIT-BIH Arrhythmia Database (enregistrements courts)
    DATABASE = 'mitdb'
    RECORD = '100'  # Enregistrement classique avec battements normaux
    
    # Option 2: Normal Sinus Rhythm Database (enregistrements longs)
    # DATABASE = 'nsrdb'
    # RECORD = '16265'
    
    DURATION_MINUTES = 5  # Combien de minutes analyser
    
    # ========== Téléchargement ==========
    
    record_path = download_physionet_record(DATABASE, RECORD, destination='data')
    
    if record_path is None:
        print("\n Téléchargement échoué. Vérifiez votre connexion internet.")
        sys.exit(1)
    
    # ========== Chargement ==========
    
    print(f"\n Chargement du signal ECG...")
    ecg, fs, info = load_physionet_ecg(record_path, channel=0, sampto=int(DURATION_MINUTES * 60 * 360))
    
    if ecg is None:
        print(" Chargement échoué.")
        sys.exit(1)
    
    print(f"\n Signal chargé:")
    print(f"   - Enregistrement: {info['record_name']}")
    print(f"   - Fréquence: {info['fs']} Hz")
    print(f"   - Durée: {info['duration_s']:.1f} secondes ({info['duration_s']/60:.1f} min)")
    print(f"   - Canaux disponibles: {info['channel_names']}")
    print(f"   - Unités: {info['units']}")
    
    # ========== Analyse HRV complète ==========
    
    print(f"\n Prétraitement et détection des R-peaks...")
    
    from src.ecg_processing.preprocess import preprocess_ecg
    from src.ecg_processing.peak_detection import detect_r_peaks, compute_heart_rate
    from src.ecg_processing.features import extract_hrv_features
    
    # Prétraitement
    ecg_clean = preprocess_ecg(ecg, fs, apply_bandpass=True, apply_notch=True, notch_freq=60.0)
    
    # Détection R-peaks (ajuster prominence_factor selon la qualité du signal)
    peaks = detect_r_peaks(ecg_clean, fs, prominence_factor=1.5)
    
    print(f" {len(peaks)} R-peaks détectés")
    
    # Fréquence cardiaque
    hr_mean, hr_std = compute_heart_rate(peaks, fs)
    print(f" Fréquence cardiaque: {hr_mean:.1f} ± {hr_std:.1f} bpm")
    
    # HRV (LF/HF)
    if len(peaks) > 20:
        print(f"\n Calcul HRV (LF/HF)...")
        hrv = extract_hrv_features(peaks, fs, method='welch')
        
        print(f"\n{'='*60}")
        print("RÉSULTATS HRV")
        print(f"{'='*60}")
        print(f"LF (ms²):       {hrv['lf']:.1f}")
        print(f"HF (ms²):       {hrv['hf']:.1f}")
        print(f"LF/HF ratio:    {hrv['lf_hf']:.2f}")
        print(f"LF_nu (%):      {hrv['lf_nu']:.1f}")
        print(f"HF_nu (%):      {hrv['hf_nu']:.1f}")
        print(f"Total (ms²):    {hrv['total']:.1f}")
        print(f"{'='*60}")
    else:
        print(" Pas assez de R-peaks pour calcul HRV fiable.")
    
    # ========== Visualisation ==========
    
    print(f"\n Génération des graphiques...")
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Calculer la PSD pour le graphique
    if len(peaks) > 20:
        from src.ecg_processing.features import (
            compute_rr_intervals, 
            interpolate_rr, 
            compute_psd_welch
        )
        rr_sec, rr_times_hrv = compute_rr_intervals(peaks, fs)
        t_interp, rr_interp = interpolate_rr(rr_sec, rr_times_hrv, fs_interp=4.0)
        freqs_psd, psd = compute_psd_welch(rr_interp, fs_interp=4.0)
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # Plot 1: Signal brut
    ax1 = axes[0, 0]
    t = np.arange(len(ecg)) / fs
    ax1.plot(t, ecg, linewidth=0.5, color='gray', alpha=0.7)
    ax1.set_title(f'Signal ECG brut - {info["record_name"]} (PhysioNet {DATABASE})', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Temps (s)')
    ax1.set_ylabel(f'Amplitude ({info["units"]})')
    ax1.grid(alpha=0.3)
    
    # Plot 2: Signal filtré + R-peaks
    ax2 = axes[0, 1]
    ax2.plot(t, ecg_clean, linewidth=0.7, color='blue', alpha=0.8)
    peaks_times = peaks / fs
    ax2.scatter(peaks_times, ecg_clean[peaks], color='red', s=50, zorder=5, label=f'{len(peaks)} R-peaks')
    ax2.set_title(f'Signal filtré + R-peaks détectés | HR = {hr_mean:.1f} bpm', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('Temps (s)')
    ax2.set_ylabel(f'Amplitude ({info["units"]})')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Plot 3: Tachogramme (intervalles RR)
    ax3 = axes[1, 0]
    if len(peaks) > 1:
        rr_samples = np.diff(peaks)
        rr_ms = (rr_samples / fs) * 1000
        rr_times = peaks_times[1:]
        ax3.plot(rr_times, rr_ms, marker='o', linestyle='-', markersize=3, linewidth=1, color='green')
        ax3.set_title('Tachogramme (intervalles RR)', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Temps (s)')
        ax3.set_ylabel('RR (ms)')
        ax3.grid(alpha=0.3)
    
    # Plot 4: PSD avec bandes LF/HF
    ax4 = axes[1, 1]
    if len(peaks) > 20:
        # Définir les bandes
        lf_band = (0.04, 0.15)
        hf_band = (0.15, 0.4)
        
        # Plot PSD
        ax4.plot(freqs_psd, psd * 1e6, linewidth=2, color='black', label='PSD (Welch)')
        
        # Colorier les bandes LF et HF
        lf_mask = (freqs_psd >= lf_band[0]) & (freqs_psd <= lf_band[1])
        hf_mask = (freqs_psd >= hf_band[0]) & (freqs_psd <= hf_band[1])
        
        ax4.fill_between(freqs_psd[lf_mask], 0, psd[lf_mask] * 1e6, 
                          alpha=0.4, color='orange', label=f'LF ({lf_band[0]}-{lf_band[1]} Hz)')
        ax4.fill_between(freqs_psd[hf_mask], 0, psd[hf_mask] * 1e6, 
                          alpha=0.4, color='cyan', label=f'HF ({hf_band[0]}-{hf_band[1]} Hz)')
        
        ax4.set_title(f'Densité Spectrale de Puissance | LF/HF = {hrv["lf_hf"]:.2f}', 
                      fontsize=14, fontweight='bold')
        ax4.set_xlabel('Fréquence (Hz)')
        ax4.set_ylabel('PSD (ms²/Hz)')
        ax4.set_xlim(0, 0.5)
        ax4.legend(loc='upper right')
        ax4.grid(alpha=0.3)
        
        # Annotations
        max_psd_value = max(psd * 1e6) if len(psd) > 0 else 1
        ax4.text(0.35, max_psd_value * 0.85, 
                 f'LF: {hrv["lf"]:.1f} ms²\nHF: {hrv["hf"]:.1f} ms²\nLF_nu: {hrv["lf_nu"]:.1f}%\nHF_nu: {hrv["hf_nu"]:.1f}%',
                 fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax4.text(0.5, 0.5, 'Pas assez de R-peaks\npour calculer la PSD', 
                 ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Densité Spectrale de Puissance', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print("\n Analyse terminée!")
