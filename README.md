# Projet_Signal — Traitement ECG et HRV (Python)

Projet pédagogique complet pour le traitement de signaux ECG en Python : filtrage, détection R-peaks, extraction HRV (LF/HF) et analyse physiologique. Conçu pour faciliter la transition de MATLAB vers Python dans VS Code (Windows).

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)

---

## ✨ Fonctionnalités

-  **Générateur ECG synthétique** — créer des signaux ECG réalistes
-  **Prétraitement** — filtres Butterworth (bandpass) + notch (50/60 Hz)
-  **Détection R-peaks** — algorithme optimisé
-  **Extraction HRV** — calcul LF/HF via Welch et FFT
-  **Analyse PhysioNet** — données ECG réelles
-  Visualisations interactives
-  Tests unitaires (pytest)
-  Notebooks Jupyter pédagogiques

---

##  Structure

```
Projet_Signal/
├── src/ecg_processing/        # Modules Python
│   ├── io.py                  # Générateur + chargement
│   ├── preprocess.py          # Filtres
│   ├── peak_detection.py      # R-peaks
│   └── features.py            # HRV (LF/HF)
├── notebooks/                 # Jupyter interactifs
│   └── tutorial_complet.ipynb # Pipeline complet
├── tests/                     # Tests unitaires
├── scripts/
│   └── analyze_physionet.py   # Analyse PhysioNet
└── requirements.txt
```

---

##  Installation (Windows)

```powershell
# Créer environnement
python -m venv .venv
.\.venv\Scripts\activate.bat

# Installer dépendances
pip install -r requirements.txt

# Dans VS Code: Ctrl+Shift+P → Python: Select Interpreter → .venv
```

---

##  Démarrage rapide

### Notebook interactif (recommandé)
```powershell
code notebooks/tutorial_complet.ipynb
# Exécuter les cellules (Shift+Enter)
```

### Script Python
```python
from src.ecg_processing.io import synthetic_ecg
from src.ecg_processing.preprocess import preprocess_ecg
from src.ecg_processing.peak_detection import detect_r_peaks
from src.ecg_processing.features import extract_hrv_features

# Générer ECG
ecg, fs = synthetic_ecg(duration_s=120, fs=250, heart_rate=70)

# Pipeline complet
ecg_clean = preprocess_ecg(ecg, fs)
peaks = detect_r_peaks(ecg_clean, fs, prominence_factor=2.0)
hrv = extract_hrv_features(peaks, fs, method='welch')

print(f"LF/HF ratio: {hrv['lf_hf']:.2f}")
```

### Données PhysioNet
```powershell
python scripts/analyze_physionet.py
```

---

##  Pipeline ECG→HRV

```
ECG → Prétraitement → R-peaks → RR → Interpolation → PSD → LF/HF
```

---

##  Tests

```powershell
pytest -v
```

---

##  MATLAB → Python

| MATLAB | Python |
|--------|--------|
| `butter(4, [f1 f2]/(fs/2))` | `signal.butter(4, [f1/nyq, f2/nyq])` |
| `filtfilt(b, a, x)` | `signal.filtfilt(b, a, x)` |
| `findpeaks(x, 'MinPeakDistance', d)` | `find_peaks(x, distance=d)` |
| `pwelch(x, ...)` | `signal.welch(x, fs=fs)` |

---

##  Interprétation HRV

- **LF/HF < 1** → Relaxation (parasympathique)
- **LF/HF > 2** → Stress (sympathique)

---

##  Notebooks

- `tutorial_complet.ipynb` — Pipeline complet expliqué
- `matlab_to_python_examples.ipynb` — Équivalences MATLAB↔Python

---

**Bon traitement de signaux ! **
