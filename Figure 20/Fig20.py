"""
Supplementary Material for Journal Submission
------------------------------------------------
This script reproduces Figure 20

Dependencies:
  numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import pathlib
from pathlib import Path

try:
    PROJECT_DIR = Path(__file__).parent.resolve()  # works in .py scripts
except NameError:
    PROJECT_DIR = Path.cwd().resolve()             # works in Jupyter

print(PROJECT_DIR)

################ n=55 ################
# Load the file
data1 = np.load(PROJECT_DIR / "n55_new_rabi_plot_data.npz")

rf_field = data1['rf_field']
rabi_freqs_mhz = data1['rabi_freqs_mhz']
rabi_errors_mhz = data1['rabi_errors_mhz']
rf_field_fit = data1['rf_field_fit']
rabi_fit_line = data1['rabi_fit_line']
E_peak = data1['E_peak']
rabi_freqs_mhzS = data1['rabi_freqs_mhzS']
rabi_errorsS = data1['rabi_errorsS']
E_fit = data1['E_fit']
rabi_fit_lineS = data1['rabi_fit_lineS']
mu_fit = data1['mu_fit']
mu_err = data1['mu_err']
mu_fitS =data1['mu_fitS']
mu_errS = data1['mu_errS']
r_squared= data1['r_squared']
r_squaredS=data1['r_squaredS']

plt.figure(figsize=(10, 6))
plt.rcParams.update({'font.family': 'serif', 'font.size': 14})
plt.subplots_adjust(left=0.15)
plt.errorbar(rf_field, rabi_freqs_mhz, yerr=rabi_errors_mhz, fmt='o', label='Data', capsize=5)
plt.plot(rf_field_fit, rabi_fit_line, label=f'Fit: μ = {mu_fit:.2f} ± {mu_err:.2f} MHz/(V/m)\nR² = {r_squared:.4f}', color='red', lw=1.5)
plt.errorbar(E_peak, rabi_freqs_mhzS, yerr=rabi_errorsS, fmt='o', label='ARC Simulated Data', capsize=5)
plt.plot(E_fit, rabi_fit_lineS, label=f'Simullation Fit: μ = {mu_fitS:.2f} ± {mu_errS:.2f} MHz/(V/m)\nR² = {r_squaredS:.4f}', color='cyan', lw=1.5)
plt.xlabel('Electric Field (V/m)')
plt.ylabel('Rabi Frequency (MHz)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(frameon=False)
plt.text(0.95, 0.05, '$55D_{5/2} \\rightarrow 56P_{3/2}$', transform=plt.gca().transAxes,
         fontsize=16, fontweight='bold', ha='right', va='bottom')
plt.text(0.95, 0.15, '(a)', transform=plt.gca().transAxes,
         fontsize=16, fontweight='bold', ha='right', va='bottom')
plt.tight_layout()
# plt.savefig('conf2_AT_n55.eps')
plt.show()


################ n=59 ################
# Load the file
data1 = np.load(PROJECT_DIR / "n59_new_rabi_plot_data.npz")

rf_field = data1['rf_field']
rabi_freqs_mhz = data1['rabi_freqs_mhz']
rabi_errors_mhz = data1['rabi_errors_mhz']
rf_field_fit = data1['rf_field_fit']
rabi_fit_line = data1['rabi_fit_line']
E_peak = data1['E_peak']
rabi_freqs_mhzS = data1['rabi_freqs_mhzS']
rabi_errorsS = data1['rabi_errorsS']
E_fit = data1['E_fit']
rabi_fit_lineS = data1['rabi_fit_lineS']
mu_fit = data1['mu_fit']
mu_err = data1['mu_err']
mu_fitS =data1['mu_fitS']
mu_errS = data1['mu_errS']
r_squared= data1['r_squared']
r_squaredS=data1['r_squaredS']

plt.figure(figsize=(10, 6))
plt.rcParams.update({'font.family': 'serif', 'font.size': 14})
plt.subplots_adjust(left=0.15)
plt.errorbar(rf_field, rabi_freqs_mhz, yerr=rabi_errors_mhz, fmt='o', label='Data', capsize=5)
plt.plot(rf_field_fit, rabi_fit_line, label=f'Fit: μ = {mu_fit:.2f} ± {mu_err:.2f} MHz/(V/m)\nR² = {r_squared:.4f}', color='red', lw=1.5)
plt.errorbar(E_peak, rabi_freqs_mhzS, yerr=rabi_errorsS, fmt='o', label='ARC Simulated Data', capsize=5)
plt.plot(E_fit, rabi_fit_lineS, label=f'Simullation Fit: μ = {mu_fitS:.2f} ± {mu_errS:.2f} MHz/(V/m)\nR² = {r_squaredS:.4f}', color='cyan', lw=1.5)
plt.xlabel('Electric Field (V/m)')
plt.ylabel('Rabi Frequency (MHz)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(frameon=False)
plt.text(0.95, 0.05, '$59D_{5/2} \\rightarrow 60P_{3/2}$', transform=plt.gca().transAxes,
         fontsize=16, fontweight='bold', ha='right', va='bottom')
plt.text(0.95, 0.15, '(b)', transform=plt.gca().transAxes,
         fontsize=16, fontweight='bold', ha='right', va='bottom')
plt.tight_layout()
# plt.savefig('conf2_AT_n59.eps')
plt.show()


################ n=69 ################
# Load the file
data1 = np.load(PROJECT_DIR / "n68_new_rabi_plot_data.npz")

rf_field = data1['rf_field']
rabi_freqs_mhz = data1['rabi_freqs_mhz']
rabi_errors_mhz = data1['rabi_errors_mhz']
rf_field_fit = data1['rf_field_fit']
rabi_fit_line = data1['rabi_fit_line']
E_peak = data1['E_peak']
rabi_freqs_mhzS = data1['rabi_freqs_mhzS']
rabi_errorsS = data1['rabi_errorsS']
E_fit = data1['E_fit']
rabi_fit_lineS = data1['rabi_fit_lineS']
mu_fit = data1['mu_fit']
mu_err = data1['mu_err']
mu_fitS =data1['mu_fitS']
mu_errS = data1['mu_errS']
r_squared= data1['r_squared']
r_squaredS=data1['r_squaredS']

plt.figure(figsize=(10, 6))
plt.rcParams.update({'font.family': 'serif', 'font.size': 14})
plt.subplots_adjust(left=0.15)
plt.errorbar(rf_field, rabi_freqs_mhz, yerr=rabi_errors_mhz, fmt='o', label='Data', capsize=5)
plt.plot(rf_field_fit, rabi_fit_line, label=f'Fit: μ = {mu_fit:.2f} ± {mu_err:.2f} MHz/(V/m)\nR² = {r_squared:.4f}', color='red', lw=1.5)
plt.errorbar(E_peak, rabi_freqs_mhzS, yerr=rabi_errorsS, fmt='o', label='ARC Simulated Data', capsize=5)
plt.plot(E_fit, rabi_fit_lineS, label=f'Simullation Fit: μ = {mu_fitS:.2f} ± {mu_errS:.2f} MHz/(V/m)\nR² = {r_squaredS:.4f}', color='cyan', lw=1.5)
plt.xlabel('Electric Field (V/m)')
plt.ylabel('Rabi Frequency (MHz)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(frameon=False)
plt.text(0.95, 0.05, '$68D_{5/2} \\rightarrow 69P_{3/2}$', transform=plt.gca().transAxes,
         fontsize=16, fontweight='bold', ha='right', va='bottom')
plt.text(0.95, 0.15, '(c)', transform=plt.gca().transAxes,
         fontsize=16, fontweight='bold', ha='right', va='bottom')
plt.tight_layout()
# plt.savefig('conf2_AT_n68.eps')
plt.show()


################ n=74 ################
# Load the file
data1 = np.load(PROJECT_DIR / "n74_new_rabi_plot_data.npz")

rf_field = data1['rf_field']
rabi_freqs_mhz = data1['rabi_freqs_mhz']
rabi_errors_mhz = data1['rabi_errors_mhz']
rf_field_fit = data1['rf_field_fit']
rabi_fit_line = data1['rabi_fit_line']
E_peak = data1['E_peak']
rabi_freqs_mhzS = data1['rabi_freqs_mhzS']
rabi_errorsS = data1['rabi_errorsS']
E_fit = data1['E_fit']
rabi_fit_lineS = data1['rabi_fit_lineS']
mu_fit = data1['mu_fit']
mu_err = data1['mu_err']
mu_fitS =data1['mu_fitS']
mu_errS = data1['mu_errS']
r_squared= data1['r_squared']
r_squaredS=data1['r_squaredS']

plt.figure(figsize=(10, 6))
plt.rcParams.update({'font.family': 'serif', 'font.size': 14})
plt.subplots_adjust(left=0.15)
plt.errorbar(rf_field, rabi_freqs_mhz, yerr=rabi_errors_mhz, fmt='o', label='Data', capsize=5)
plt.plot(rf_field_fit, rabi_fit_line, label=f'Fit: μ = {mu_fit:.2f} ± {mu_err:.2f} MHz/(V/m)\nR² = {r_squared:.4f}', color='red', lw=1.5)
plt.errorbar(E_peak, rabi_freqs_mhzS, yerr=rabi_errorsS, fmt='o', label='ARC Simulated Data', capsize=5)
plt.plot(E_fit, rabi_fit_lineS, label=f'Simullation Fit: μ = {mu_fitS:.2f} ± {mu_errS:.2f} MHz/(V/m)\nR² = {r_squaredS:.4f}', color='cyan', lw=1.5)
plt.xlabel('Electric Field (V/m)')
plt.ylabel('Rabi Frequency (MHz)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(frameon=False)
plt.text(0.95, 0.05, '$74D_{5/2} \\rightarrow 75P_{3/2}$', transform=plt.gca().transAxes,
         fontsize=16, fontweight='bold', ha='right', va='bottom')
plt.text(0.95, 0.15, '(d)', transform=plt.gca().transAxes,
         fontsize=16, fontweight='bold', ha='right', va='bottom')
plt.tight_layout()
# plt.savefig('conf2_AT_n74.eps')
plt.show()


################ n=82 ################
# Load the file
data1 = np.load(PROJECT_DIR / "n82_new_rabi_plot_data.npz")

rf_field = data1['rf_field']
rabi_freqs_mhz = data1['rabi_freqs_mhz']
rabi_errors_mhz = data1['rabi_errors_mhz']
rf_field_fit = data1['rf_field_fit']
rabi_fit_line = data1['rabi_fit_line']
E_peak = data1['E_peak']
rabi_freqs_mhzS = data1['rabi_freqs_mhzS']
rabi_errorsS = data1['rabi_errorsS']
E_fit = data1['E_fit']
rabi_fit_lineS = data1['rabi_fit_lineS']
mu_fit = data1['mu_fit']
mu_err = data1['mu_err']
mu_fitS =data1['mu_fitS']
mu_errS = data1['mu_errS']
r_squared= data1['r_squared']
r_squaredS=data1['r_squaredS']

plt.figure(figsize=(10, 6))
plt.rcParams.update({'font.family': 'serif', 'font.size': 14})
plt.subplots_adjust(left=0.15)
plt.errorbar(rf_field, rabi_freqs_mhz, yerr=rabi_errors_mhz, fmt='o', label='Data', capsize=5)
plt.plot(rf_field_fit, rabi_fit_line, label=f'Fit: μ = {mu_fit:.2f} ± {mu_err:.2f} MHz/(V/m)\nR² = {r_squared:.4f}', color='red', lw=1.5)
plt.errorbar(E_peak, rabi_freqs_mhzS, yerr=rabi_errorsS, fmt='o', label='ARC Simulated Data', capsize=5)
plt.plot(E_fit, rabi_fit_lineS, label=f'Simullation Fit: μ = {mu_fitS:.2f} ± {mu_errS:.2f} MHz/(V/m)\nR² = {r_squaredS:.4f}', color='cyan', lw=1.5)
plt.xlabel('Electric Field (V/m)')
plt.ylabel('Rabi Frequency (MHz)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(frameon=False)
plt.text(0.95, 0.05, '$82D_{5/2} \\rightarrow 83P_{3/2}$', transform=plt.gca().transAxes,
         fontsize=16, fontweight='bold', ha='right', va='bottom')
plt.text(0.95, 0.15, '(e)', transform=plt.gca().transAxes,
         fontsize=16, fontweight='bold', ha='right', va='bottom')
plt.tight_layout()
# plt.savefig('conf2_AT_n82.eps')
plt.show()
