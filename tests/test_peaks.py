import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ecg_processing.io import synthetic_ecg
from src.ecg_processing.preprocess import preprocess_ecg
from src.ecg_processing.peak_detection import detect_r_peaks, compute_heart_rate
import matplotlib.pyplot as plt

# Générer ECG (30 secondes, HR=75 bpm)
ecg, fs = synthetic_ecg(duration_s=30, fs=250, heart_rate=75)

# Prétraiter
ecg_clean = preprocess_ecg(ecg, fs)

# Détecter R-peaks

peaks = detect_r_peaks(ecg_clean, fs)

# Calculer HR
hr_mean, hr_std = compute_heart_rate(peaks, fs)

print(f" {len(peaks)} R-peaks détectés")
print(f" Fréquence cardiaque: {hr_mean:.1f} ± {hr_std:.1f} bpm")

# Visualiser (premiers 5 secondes)
plt.figure(figsize=(14, 4))
n_show = 5 * fs  # 5 secondes
plt.plot(ecg_clean[:n_show], label='ECG filtré', alpha=0.7)
peaks_show = peaks[peaks < n_show]
plt.scatter(peaks_show, ecg_clean[peaks_show], color='red', s=100, 
            label=f'R-peaks ({len(peaks_show)} détectés)', zorder=5)
plt.title('Détection des R-peaks')
plt.xlabel('échantillons')
plt.ylabel('µV')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()