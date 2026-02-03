import sys
sys.path.insert(0, 'c:\\Users\\larea\\Documents\\Projet_Signal')

from src.ecg_processing.io import synthetic_ecg
from src.ecg_processing.preprocess import preprocess_ecg
from src.ecg_processing.peak_detection import detect_r_peaks
from src.ecg_processing.features import extract_hrv_features

# Générer ECG (2 minutes pour avoir assez de données HRV)
ecg, fs = synthetic_ecg(duration_s=120, fs=250, heart_rate=70)
ecg_clean = preprocess_ecg(ecg, fs)
peaks = detect_r_peaks(ecg_clean, fs, prominence_factor=2.0)

# Comparer Welch vs FFT
hrv_welch = extract_hrv_features(peaks, fs, method='welch')
hrv_fft = extract_hrv_features(peaks, fs, method='fft')

print("========== Méthode Welch ==========")
print(f"LF: {hrv_welch['lf']:.1f} ms²")
print(f"HF: {hrv_welch['hf']:.1f} ms²")
print(f"LF/HF: {hrv_welch['lf_hf']:.2f}")
print(f"LF_nu: {hrv_welch['lf_nu']:.1f} %")
print(f"HF_nu: {hrv_welch['hf_nu']:.1f} %")

print("\n========== Méthode FFT ==========")
print(f"LF: {hrv_fft['lf']:.1f} ms²")
print(f"HF: {hrv_fft['hf']:.1f} ms²")
print(f"LF/HF: {hrv_fft['lf_hf']:.2f}")
print(f"LF_nu: {hrv_fft['lf_nu']:.1f} %")
print(f"HF_nu: {hrv_fft['hf_nu']:.1f} %")