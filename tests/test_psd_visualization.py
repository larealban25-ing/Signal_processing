import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ecg_processing.io import synthetic_ecg
from src.ecg_processing.preprocess import preprocess_ecg
from src.ecg_processing.peak_detection import detect_r_peaks
from src.ecg_processing.features import (
    compute_rr_intervals, 
    interpolate_rr, 
    compute_psd_welch,
    compute_psd_fft,
    compute_lf_hf
)
import matplotlib.pyplot as plt
import numpy as np

# ========== Générer et prétraiter ECG ==========
print("Génération du signal ECG...")
ecg, fs = synthetic_ecg(duration_s=120, fs=250, heart_rate=70)
ecg_clean = preprocess_ecg(ecg, fs)
peaks = detect_r_peaks(ecg_clean, fs, prominence_factor=2.0)

print(f" {len(peaks)} R-peaks détectés")

# ========== Calculer RR et interpoler ==========
rr_sec, rr_times = compute_rr_intervals(peaks, fs)
t_interp, rr_interp = interpolate_rr(rr_sec, rr_times, fs_interp=4.0)

# ========== Calculer PSD avec les deux méthodes ==========
freqs_welch, psd_welch = compute_psd_welch(rr_interp, fs_interp=4.0)
freqs_fft, psd_fft = compute_psd_fft(rr_interp, fs_interp=4.0)

# ========== Calculer LF/HF ==========
hrv_welch = compute_lf_hf(freqs_welch, psd_welch)
hrv_fft = compute_lf_hf(freqs_fft, psd_fft)

# ========== Visualisation ==========
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# ========== Subplot 1: Tachogramme (RR interpolé) ==========
ax1 = axes[0, 0]
ax1.plot(t_interp, rr_interp * 1000, linewidth=1, color='darkblue')
ax1.set_title('Tachogramme interpolé', fontsize=14, fontweight='bold')
ax1.set_xlabel('Temps (s)')
ax1.set_ylabel('RR (ms)')
ax1.grid(alpha=0.3)

# ========== Subplot 2: RR bruts (scatter) ==========
ax2 = axes[0, 1]
ax2.scatter(rr_times, rr_sec * 1000, s=30, color='red', alpha=0.7, edgecolors='black')
ax2.plot(t_interp, rr_interp * 1000, linewidth=0.8, alpha=0.5, color='blue', label='Interpolation cubique')
ax2.set_title('Intervalles RR (bruts + interpolation)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Temps (s)')
ax2.set_ylabel('RR (ms)')
ax2.legend()
ax2.grid(alpha=0.3)

# ========== Subplot 3: PSD - Méthode Welch ==========
ax3 = axes[1, 0]

# Définir les bandes LF et HF
lf_band = (0.04, 0.15)
hf_band = (0.15, 0.4)

# Plot PSD
ax3.plot(freqs_welch, psd_welch * 1e6, linewidth=2, color='black', label='PSD (Welch)')

# Colorier les bandes LF et HF
lf_mask = (freqs_welch >= lf_band[0]) & (freqs_welch <= lf_band[1])
hf_mask = (freqs_welch >= hf_band[0]) & (freqs_welch <= hf_band[1])

ax3.fill_between(freqs_welch[lf_mask], 0, psd_welch[lf_mask] * 1e6, 
                  alpha=0.4, color='orange', label=f'LF ({lf_band[0]}-{lf_band[1]} Hz)')
ax3.fill_between(freqs_welch[hf_mask], 0, psd_welch[hf_mask] * 1e6, 
                  alpha=0.4, color='cyan', label=f'HF ({hf_band[0]}-{hf_band[1]} Hz)')

ax3.set_title(f'PSD - Méthode Welch | LF/HF = {hrv_welch["lf_hf"]:.2f}', 
              fontsize=14, fontweight='bold')
ax3.set_xlabel('Fréquence (Hz)')
ax3.set_ylabel('PSD (ms²/Hz)')
ax3.set_xlim(0, 0.5)
ax3.legend(loc='upper right')
ax3.grid(alpha=0.3)

# Annotations
ax3.text(0.35, max(psd_welch * 1e6) * 0.9, 
         f'LF: {hrv_welch["lf"]:.1f} ms²\nHF: {hrv_welch["hf"]:.1f} ms²\nLF_nu: {hrv_welch["lf_nu"]:.1f}%\nHF_nu: {hrv_welch["hf_nu"]:.1f}%',
         fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# ========== Subplot 4: PSD - Méthode FFT ==========
ax4 = axes[1, 1]

# Plot PSD
ax4.plot(freqs_fft, psd_fft * 1e6, linewidth=2, color='black', label='PSD (FFT)')

# Colorier les bandes LF et HF
lf_mask_fft = (freqs_fft >= lf_band[0]) & (freqs_fft <= lf_band[1])
hf_mask_fft = (freqs_fft >= hf_band[0]) & (freqs_fft <= hf_band[1])

ax4.fill_between(freqs_fft[lf_mask_fft], 0, psd_fft[lf_mask_fft] * 1e6, 
                  alpha=0.4, color='orange', label=f'LF ({lf_band[0]}-{lf_band[1]} Hz)')
ax4.fill_between(freqs_fft[hf_mask_fft], 0, psd_fft[hf_mask_fft] * 1e6, 
                  alpha=0.4, color='cyan', label=f'HF ({hf_band[0]}-{hf_band[1]} Hz)')

ax4.set_title(f'PSD - Méthode FFT | LF/HF = {hrv_fft["lf_hf"]:.2f}', 
              fontsize=14, fontweight='bold')
ax4.set_xlabel('Fréquence (Hz)')
ax4.set_ylabel('PSD (ms²/Hz)')
ax4.set_xlim(0, 0.5)
ax4.legend(loc='upper right')
ax4.grid(alpha=0.3)

# Annotations
ax4.text(0.35, max(psd_fft * 1e6) * 0.9, 
         f'LF: {hrv_fft["lf"]:.1f} ms²\nHF: {hrv_fft["hf"]:.1f} ms²\nLF_nu: {hrv_fft["lf_nu"]:.1f}%\nHF_nu: {hrv_fft["hf_nu"]:.1f}%',
         fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()

# ========== Résumé dans le terminal ==========
print("\n" + "="*50)
print("RÉSUMÉ HRV - Comparaison Welch vs FFT")
print("="*50)
print(f"\n{'Métrique':<15} {'Welch':<15} {'FFT':<15}")
print("-"*50)
print(f"{'LF (ms²)':<15} {hrv_welch['lf']:<15.1f} {hrv_fft['lf']:<15.1f}")
print(f"{'HF (ms²)':<15} {hrv_welch['hf']:<15.1f} {hrv_fft['hf']:<15.1f}")
print(f"{'Total (ms²)':<15} {hrv_welch['total']:<15.1f} {hrv_fft['total']:<15.1f}")
print(f"{'LF/HF':<15} {hrv_welch['lf_hf']:<15.2f} {hrv_fft['lf_hf']:<15.2f}")
print(f"{'LF_nu (%)':<15} {hrv_welch['lf_nu']:<15.1f} {hrv_fft['lf_nu']:<15.1f}")
print(f"{'HF_nu (%)':<15} {hrv_welch['hf_nu']:<15.1f} {hrv_fft['hf_nu']:<15.1f}")
print("="*50)
