"""
Supplementary Material for Journal Submission
------------------------------------------------
This script reproduces Figure 6

This script plots the quantum defect values for Rb 87 
from Mack et al.(Phys. Rev. A 83, 052515), along with the Rydberg-Ritz fit to the data.
The quantum defect from ARC (both isotopes) is overlayed for reference.

Dependencies:
  numpy, matplotlib, scipy
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pathlib
from pathlib import Path


try:
    PROJECT_DIR = Path(__file__).parent.resolve()  # works in .py scripts
except NameError:
    PROJECT_DIR = Path.cwd().resolve()             # works in Jupyter

print(PROJECT_DIR)
# ============================================================
# Rydberg–Ritz Quantum Defect Fit for nD3/2 and nD5/2
# Based on Mack et al. (Phys. Rev. A 83, 052515)
# ============================================================

# ---------------------------
# Load Mack data from .npz files
# ---------------------------
w = np.load(PROJECT_DIR / '32_xy_vals.npz')
v = np.load(PROJECT_DIR / '52_xy_vals.npz')

n_data_32, defect_data_32 = w['d32x'], w['d32y']
n_data_52, defect_data_52 = v['d52x'], v['d52y']

# ---------------------------
# Define Rydberg–Ritz formula
# ---------------------------
def rydberg_ritz(n, a, b, c):
    return a + b / ((n - a)**2) + c / ((n - a)**4)

# ---------------------------
# Fit nD3/2 data
# ---------------------------
initial_guess = [1.34, -0.59, -1.51]  # Rydberg Atoms by T. Gallagher Table 16.2 p.353
popt_D32, pcov_D32 = curve_fit(rydberg_ritz, n_data_32, defect_data_32, p0=initial_guess)

# Generate fit curve for D3/2
n_fit_32 = np.linspace(min(n_data_32), max(n_data_32) + 5, 500)
defect_fit_32 = rydberg_ritz(n_fit_32, *popt_D32)

# Load parameters for Rubidium from ARC
data1 = np.load(PROJECT_DIR / 'd32_ARC_data.npz')
data2 = np.load(PROJECT_DIR / 'd52_ARC_data.npz')

# Plot D3/2
plt.figure(figsize=(10, 6))
plt.rcParams.update({'font.family': 'serif', 'font.size': 14})
plt.plot(n_data_32, defect_data_32, 'r.', markersize=20, alpha=0.5,
         label=r'$nD_{3/2}$ defect data from Mack et al. (Phys. Rev. A 83, 052515)')
plt.plot(n_fit_32, defect_fit_32, 'b-', label='Rydberg–Ritz fit to data from Mack et al.', lw=1.5)
plt.plot(data1['x0'],data1['y0'], 'y^', markersize=15, alpha=0.5,
         label=r'$nD_{3/2}$ defect data from ARC')
plt.xlabel('Principal Quantum Number $n$')
plt.ylabel('Quantum Defect $\\delta(n)$')
plt.legend(loc='lower right', frameon=False)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.text(0.95, 0.25, '(a)', transform=plt.gca().transAxes,
         fontsize=16, fontweight='bold', ha='right', va='bottom')

# Print fitted equation for D3/2
equation_D32 = (
    f"δ(n) = {popt_D32[0]:.6f} + {popt_D32[1]:.6f}/(n - {popt_D32[0]:.6f})² "
    f"+ {popt_D32[2]:.6f}/(n - {popt_D32[0]:.6f})⁴"
)
print("\nFitted Rydberg–Ritz Equation for nD_{3/2}:")
print(equation_D32)
print("\nFit Parameters:")
print(f"a = {popt_D32[0]:.6f}")
print(f"b = {popt_D32[1]:.6f}")
print(f"c = {popt_D32[2]:.6f}")

# ---------------------------
# Fit nD5/2 data
# ---------------------------
popt_D52, pcov_D52 = curve_fit(rydberg_ritz, n_data_52, defect_data_52, p0=initial_guess)

# Generate fit curve for D5/2
n_fit_52 = np.linspace(min(n_data_52), max(n_data_52) + 5, 500)
defect_fit_52 = rydberg_ritz(n_fit_52, *popt_D52)

# Plot D5/2
plt.figure(figsize=(10, 6))
plt.rcParams.update({'font.family': 'serif', 'font.size': 14})
plt.plot(n_data_52, defect_data_52, 'r.', markersize=20, alpha=0.5,
         label=r'$nD_{5/2}$ defect data from Mack et al. (Phys. Rev. A 83, 052515)')
plt.plot(n_fit_52, defect_fit_52, 'b-', label='Rydberg–Ritz fit to data from Mack et al.', lw=1.5)
plt.plot(data2['x0'],data2['y0'], 'y^', markersize=15, alpha=0.5,
         label=r'$nD_{5/2}$ defect data from ARC')
plt.xlabel('Principal Quantum Number $n$')
plt.ylabel('Quantum Defect $\\delta(n)$')
plt.legend(loc='lower right', frameon=False)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.text(0.95, 0.25, '(b)', transform=plt.gca().transAxes,
         fontsize=16, fontweight='bold', ha='right', va='bottom')
plt.show()
# Print fitted equation for D5/2
equation_D52 = (
    f"δ(n) = {popt_D52[0]:.6f} + {popt_D52[1]:.6f}/(n - {popt_D52[0]:.6f})² "
    f"+ {popt_D52[2]:.6f}/(n - {popt_D52[0]:.6f})⁴"
)
print("\nFitted Rydberg–Ritz Equation for nD_{5/2}:")
print(equation_D52)
print("\nFit Parameters:")
print(f"a = {popt_D52[0]:.6f}")
print(f"b = {popt_D52[1]:.6f}")
print(f"c = {popt_D52[2]:.6f}")
