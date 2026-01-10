import numpy as np
import matplotlib.pyplot as plt

################ n=53 ################

# Load the file
data = np.load("n53_AT_data.npz")

# Access individual arrays
X0 = data['x0']
Y0 = data['y0']
X1 = data['x1']
Y1 = data['y1']
X2 = data['x2']
Y2 = data['y2']
X3 = data['x3']
Y3 = data['y3']

plt.figure(figsize=(10, 6))
plt.rcParams.update({'font.family': 'serif', 'font.size': 14})
plt.subplots_adjust(left=0.15)
plt.plot(X0, Y0, label='RF Off')
plt.plot(X1, Y1, label='-2 dBm')
plt.plot(X2, Y2, label='0 dBm')
plt.plot(X3, Y3, label='5 dBm')
plt.xlabel('Coupling Laser Detuning [MHz]')
plt.ylabel('Probe Transmission (arb.)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(frameon=False)
plt.tight_layout()
plt.text(0.25, 0.85, '$53D_{5/2} \\rightarrow 54P_{3/2}$', transform=plt.gca().transAxes,
         fontsize=16, fontweight='bold', ha='right', va='bottom')
plt.show()

# Load the file
data1 = np.load("n53rabi_plot_data.npz")

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
plt.text(0.95, 0.05, '$53D_{5/2} \\rightarrow 54P_{3/2}$', transform=plt.gca().transAxes,
         fontsize=16, fontweight='bold', ha='right', va='bottom')
plt.tight_layout()
plt.show()


################ n=75 ################

# Load the file
data2 = np.load("n75_AT_data.npz")

# Access individual arrays
X0 = data2['x0']
Y0 = data2['y0']
X1 = data2['x1']
Y1 = data2['y1']
X2 = data2['x2']
Y2 = data2['y2']
X3 = data2['x3']
Y3 = data2['y3']

plt.figure(figsize=(10, 6))
plt.rcParams.update({'font.family': 'serif', 'font.size': 14})
plt.subplots_adjust(left=0.15)
plt.plot(X0, Y0, label='RF Off')
plt.plot(X1, Y1, label='-8 dBm')
plt.plot(X2, Y2, label='-6 dBm')
plt.plot(X3, Y3, label='0 dBm')
plt.xlabel('Coupling Laser Detuning [MHz]')
plt.ylabel('Probe Transmission (arb.)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(frameon=False)
plt.tight_layout()
plt.text(0.25, 0.85, '$75D_{5/2} \\rightarrow 76P_{3/2}$', transform=plt.gca().transAxes,
         fontsize=16, fontweight='bold', ha='right', va='bottom')
plt.show()

# Load the file
data3 = np.load("n75rabi_plot_data.npz")

rf_field = data3['rf_field']
rabi_freqs_mhz = data3['rabi_freqs_mhz']
rabi_errors_mhz = data3['rabi_errors_mhz']
rf_field_fit = data3['rf_field_fit']
rabi_fit_line = data3['rabi_fit_line']
E_peak = data3['E_peak']
rabi_freqs_mhzS = data3['rabi_freqs_mhzS']
rabi_errorsS = data3['rabi_errorsS']
E_fit = data3['E_fit']
rabi_fit_lineS = data3['rabi_fit_lineS']
mu_fit = data3['mu_fit']
mu_err = data3['mu_err']
mu_fitS =data3['mu_fitS']
mu_errS = data3['mu_errS']
r_squared= data3['r_squared']
r_squaredS=data3['r_squaredS']

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
plt.text(0.95, 0.05, '$75D_{5/2} \\rightarrow 76P_{3/2}$', transform=plt.gca().transAxes,
         fontsize=16, fontweight='bold', ha='right', va='bottom')
plt.tight_layout()
plt.show()

################ n=85 ################

# Load the file
data4 = np.load("n85_AT_data.npz")

# Access individual arrays
X0 = data4['x0']
Y0 = data4['y0']
X1 = data4['x1']
Y1 = data4['y1']
X2 = data4['x2']
Y2 = data4['y2']
X3 = data4['x3']
Y3 = data4['y3']

plt.figure(figsize=(10, 6))
plt.rcParams.update({'font.family': 'serif', 'font.size': 14})
plt.subplots_adjust(left=0.15)
plt.plot(X0, Y0, label='RF Off')
plt.plot(X1, Y1, label='-10 dBm')
plt.plot(X2, Y2, label='-6 dBm')
plt.plot(X3, Y3, label='-2 dBm')
plt.xlabel('Coupling Laser Detuning [MHz]')
plt.ylabel('Probe Transmission (arb.)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(frameon=False)
plt.tight_layout()
plt.text(0.25, 0.85, '$85D_{5/2} \\rightarrow 86P_{3/2}$', transform=plt.gca().transAxes,
         fontsize=16, fontweight='bold', ha='right', va='bottom')
plt.show()

# Load the file
data5 = np.load("n85rabi_plot_data.npz")

rf_field = data5['rf_field']
rabi_freqs_mhz = data5['rabi_freqs_mhz']
rabi_errors_mhz = data5['rabi_errors_mhz']
rf_field_fit = data5['rf_field_fit']
rabi_fit_line = data5['rabi_fit_line']
E_peak = data5['E_peak']
rabi_freqs_mhzS = data5['rabi_freqs_mhzS']
rabi_errorsS = data5['rabi_errorsS']
E_fit = data5['E_fit']
rabi_fit_lineS = data5['rabi_fit_lineS']
mu_fit = data5['mu_fit']
mu_err = data5['mu_err']
mu_fitS =data5['mu_fitS']
mu_errS = data5['mu_errS']
r_squared= data5['r_squared']
r_squaredS=data5['r_squaredS']

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
plt.text(0.95, 0.05, '$85D_{5/2} \\rightarrow 86P_{3/2}$', transform=plt.gca().transAxes,
         fontsize=16, fontweight='bold', ha='right', va='bottom')
plt.tight_layout()
plt.show()