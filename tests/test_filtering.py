import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ecg_processing.io import synthetic_ecg
from src.ecg_processing.preprocess import preprocess_ecg
import matplotlib.pyplot as plt
import numpy as np

# Générer un ECG synthétique (10 secondes)
ecg, fs = synthetic_ecg(duration_s=10, fs=250, heart_rate=60)

# Ajouter du bruit secteur (50 Hz) pour le test
t = np.arange(len(ecg)) / fs
ecg_noisy = ecg + 0.3 * np.sin(2 * np.pi * 50 * t)

# Prétraiter
ecg_clean = preprocess_ecg(ecg_noisy, fs, apply_bandpass=True, apply_notch=True)

# Comparer avant/après
plt.figure(figsize=(14, 6))

plt.subplot(2, 1, 1)
plt.plot(ecg_noisy[:750], label='Brut (avec bruit 50 Hz)', alpha=0.7)
plt.title('Avant filtrage')
plt.ylabel('µV')
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(ecg_clean[:750], label='Filtré', color='green')
plt.title('Après bandpass + notch')
plt.xlabel('échantillons')
plt.ylabel('µV')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()