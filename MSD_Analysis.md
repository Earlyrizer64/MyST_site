---
title: MSD Analysis
kernelspec:
  name: python3
  display_name: Python 3 (Pyodide)
---

# MSD Analysis

Hopefully you obtained your data files from the simulation
If not, use the linked files below

| <a href="https://github.com/Earlyrizer64/MyST_site/raw/main/Reference_Files/MSD_Analysis_Ex/msd_300.dat"> <img src="https://img.shields.io/badge/msd_300-blue" width="100" height="50"> </a> | <a href="https://github.com/Earlyrizer64/MyST_site/raw/main/Reference_Files/MSD_Analysis_Ex/msd_400.dat"> <img src="https://img.shields.io/badge/msd_400-blue" width="100" height="50"> </a> |
| -- | -- |
| msd_300.dat | msd_400.dat |

```{code-cell} python
# Cell 1: Run Cell and Upload msd_300.dat and msd_400.dat Obtained from the Simulation or from the download above
import micropip
await micropip.install("ipywidgets")
await micropip.install("matplotlib")

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import io

upload_1 = widgets.FileUpload(accept='.dat', multiple=False)
display(upload_1)
print("Upload your msd_300.dat file above.")

upload_2 = widgets.FileUpload(accept='.dat', multiple=False)
dsplay(upload_1)
print("Upload your msd_400.dat file above.")
```




```{code-cell} python
# Read / Collect Data from user input files

content_1 = upload1.value[0]['content']
text_1 = io.BytesIO(bytes(content_1)).read().decode('utf-8')

# Parse — skip comment lines starting with #
lines = [l for l in text_1.strip().split('\n') if not l.startswith('#')]
data_300 = np.array([list(map(float, l.split())) for l in lines if l.strip()])

content_2 = upload2.value[0]['content']
text_2 = io.BytesIO(bytes(content_2)).read().decode('utf-8')

# Parse — skip comment lines starting with #
lines = [l for l in text_1.strip().split('\n') if not l.startswith('#')]
data_400 = np.array([list(map(float, l.split())) for l in lines if l.strip()])

timestep_300 = data_300[:, 0] # femptosecond (Default time for Real Units)
msd_300 = data_300[:, 1]

timestep_400 = data_400[:, 0] # femptosecond (Default time for Real Units)
msd_400 = data_400[:, 1]

# Convert to picoseconds

time_300 = timestep_300 / 1000
time_400 = timestep_400 / 1000

# Define the minimum time for fitting (diffusive regime starts here)
t_min = 10.0  # ps

# Create masks to select data where time > t_min
mask_300 = time_300 > t_min
mask_400 = time_400 > t_min

# Extract the diffusive region data
time_fit_300 = time_300[mask_300]
msd_fit_300  = msd_300[mask_300]

time_fit_400 = time_400[mask_400]
msd_fit_400  = msd_400[mask_400]

print(f"Using {len(time_fit_300)} points for 300K fit (t > {t_min} ps)")
print(f"Using {len(time_fit_400)} points for 400K fit (t > {t_min} ps)")
```

```{code-cell} python
# Convert time back to femtoseconds for fitting
time_fit_300_fs = time_fit_300 * 1000.0  # Convert ps to fs
time_fit_400_fs = time_fit_400 * 1000.0  # Convert ps to fs

# Perform linear fit: MSD = slope * t + intercept
coeffs_300 = np.polyfit(time_fit_300_fs, msd_fit_300, 1)
slope_300  = coeffs_300[0]
intercept_300 = coeffs_300[1]

coeffs_400 = np.polyfit(time_fit_400_fs, msd_fit_400, 1)
slope_400  = coeffs_400[0]
intercept_400 = coeffs_400[1]

print("="*60)
print("LINEAR FIT RESULTS")
print("="*60)
print(f"\n300 K:")
print(f"  Slope     = {slope_300:.6e} Ų/fs")
print(f"  Intercept = {intercept_300:.6f} Ų")

print(f"\n400 K:")
print(f"  Slope     = {slope_400:.6e} Ų/fs")
print(f"  Intercept = {intercept_400:.6f} Ų")
```

```{code-cell} python
# Calculate diffusion coefficient in Ų/fs
D_300_A2_fs = slope_300 / 6.0
D_400_A2_fs = slope_400 / 6.0

# Convert to cm²/s (standard units)
# 1 Ų/fs = 1e-16 m²/fs = 1e-16 m² / 1e-15 s = 1e-1 m²/s = 1e3 cm²/s
# Actually: 1 Ų = 1e-20 m², 1 fs = 1e-15 s
# So 1 Ų/fs = 1e-20/1e-15 = 1e-5 m²/s = 1e-1 cm²/s
conversion = 1e-1  # Ų/fs to cm²/s

D_300_cm2_s = D_300_A2_fs * conversion
D_400_cm2_s = D_400_A2_fs * conversion

print("="*60)
print("DIFFUSION COEFFICIENTS")
print("="*60)
print(f"\n300 K:")
print(f"  D = {D_300_A2_fs:.6e} Ų/fs")
print(f"  D = {D_300_cm2_s:.6e} cm²/s")

print(f"\n400 K:")
print(f"  D = {D_400_A2_fs:.6e} Ų/fs")
print(f"  D = {D_400_cm2_s:.6e} cm²/s")

print(f"\nRatio: D(400K)/D(300K) = {D_400_cm2_s/D_300_cm2_s:.3f}")
print(f"\nDiffusion is {D_400_cm2_s/D_300_cm2_s:.2f}x faster at 400K compared to 300K.")
print("="*60)
```

```{code-cell} python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# --- Plot 300K ---
ax1.plot(time_300, msd_300, 'o', markersize=3, alpha=0.6, 
         color='steelblue', label='LAMMPS Data')
ax1.plot(time_fit_300, slope_300*time_fit_300_fs + intercept_300, 
         'r-', linewidth=2.5, label=f'Linear Fit (D={D_300_cm2_s:.3e} cm²/s)')
ax1.axvline(x=t_min, color='gray', linestyle='--', linewidth=1, 
           label=f'Fit region starts (t={t_min} ps)')
ax1.set_xlabel('Time (ps)', fontsize=12)
ax1.set_ylabel('MSD (Ų)', fontsize=12)
ax1.set_title('300 K: Mean-Squared Displacement', fontsize=13, fontweight='bold')
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3)

# --- Plot 400K ---
ax2.plot(time_400, msd_400, 'o', markersize=3, alpha=0.6, 
         color='darkorange', label='LAMMPS Data')
ax2.plot(time_fit_400, slope_400*time_fit_400_fs + intercept_400, 
         'r-', linewidth=2.5, label=f'Linear Fit (D={D_400_cm2_s:.3e} cm²/s)')
ax2.axvline(x=t_min, color='gray', linestyle='--', linewidth=1, 
           label=f'Fit region starts (t={t_min} ps)')
ax2.set_xlabel('Time (ps)', fontsize=12)
ax2.set_ylabel('MSD (Ų)', fontsize=12)
ax2.set_title('400 K: Mean-Squared Displacement', fontsize=13, fontweight='bold')
ax2.legend(loc='upper left', fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('msd_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nPlot saved as 'msd_analysis.png'")
```

```{code-cell} python
# Calculate R-squared for both fits
def calculate_r2(y_data, y_fit):
    """Calculate coefficient of determination R^2"""
    ss_res = np.sum((y_data - y_fit)**2)  # Residual sum of squares
    ss_tot = np.sum((y_data - np.mean(y_data))**2)  # Total sum of squares
    return 1 - (ss_res / ss_tot)

# Predicted values from linear fit
msd_pred_300 = slope_300 * time_fit_300_fs + intercept_300
msd_pred_400 = slope_400 * time_fit_400_fs + intercept_400

r2_300 = calculate_r2(msd_fit_300, msd_pred_300)
r2_400 = calculate_r2(msd_fit_400, msd_pred_400)

print("="*60)
print("FIT QUALITY (R² values)")
print("="*60)
print(f"300 K: R² = {r2_300:.6f}")
print(f"400 K: R² = {r2_400:.6f}")
print(f"\nInterpretation:")
print(f"  R² > 0.99  → Excellent fit")
print(f"  R² > 0.95  → Good fit")
print(f"  R² < 0.95  → Poor fit (consider longer simulation or different fit range)")
print("="*60)
```

```{code-cell} python
print("\n" + "="*60)
print("FINAL RESULTS FOR SUBMISSION")
print("="*60)
print(f"\nDiffusion Coefficients:")
print(f"  D(300K) = {D_300_cm2_s:.6e} cm²/s")
print(f"  D(400K) = {D_400_cm2_s:.6e} cm²/s")
print(f"  Ratio   = {D_400_cm2_s/D_300_cm2_s:.3f}")
print(f"\nFit Quality:")
print(f"  R²(300K) = {r2_300:.4f}")
print(f"  R²(400K) = {r2_400:.4f}")
print(f"\nPhysical Interpretation:")
print(f"  Increasing temperature by {(400-300)/300*100:.0f}% increased diffusion by {(D_400_cm2_s/D_300_cm2_s-1)*100:.0f}%")
print("="*60)
```

