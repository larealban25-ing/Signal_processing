import sys
sys.path.insert(0, 'c:\\Users\\larea\\Documents\\Projet_Signal')

from src.ecg_processing.io import synthetic_ecg
import matplotlib.pyplot as plt

# Générer 10 secondes d'ECG à 250 Hz
ecg, fs = synthetic_ecg(duration_s=10, fs=250, heart_rate=60)

# Afficher les 3 premières secondes
plt.figure(figsize=(12, 3))
plt.plot(ecg[:3*fs])
plt.title('ECG synthétique')
plt.xlabel('échantillons')
plt.ylabel('µV')
plt.grid()
plt.show()