import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import voigt_profile

# load data
w = np.load('n53_xy_values.npz')

X_fit = w['x_vals']
Y_fit_norm = w['y_vals']

# Define double Gaussian function
def double_gaussian(x, a1, x01, sigma1, a2, x02, sigma2, offset):
    return (a1 * np.exp(-(x - x01)**2 / (2 * sigma1**2)) +
            a2 * np.exp(-(x - x02)**2 / (2 * sigma2**2)) + offset)

# Initial guess for the parameters in the double Gaussian fit:
# a1: amplitude of the first (main) peak
# mu1: center position of the first peak (in seconds)
# sigma1: standard deviation (width) of the first peak
# a2: amplitude of the second (shoulder) peak
# mu2: center position of the second peak (in seconds)
# sigma2: standard deviation (width) of the second peak
initial_guess = [1, 0.00, 0.002, 0.25, -0.10, 0.002, 0.1]
# Fit the model to the data with increased maxfev
params_G, _ = curve_fit(double_gaussian, X_fit, Y_fit_norm, p0=initial_guess, maxfev=10000)

# Extract peak centers
mu1, mu2 = params_G[1], params_G[4]
temporal_spacing_G = abs(mu2 - mu1)


# Output the temporal spacing
print(f"The temporal spacing between the two Gaussian peaks is approximately {temporal_spacing_G:.6f} seconds.")


# Define a double Lorentzian model
def double_lorentzian(x, a1, mu1, gamma1, a2, mu2, gamma2):
    return (a1 * gamma1**2 / ((x - mu1)**2 + gamma1**2) +
            a2 * gamma2**2 / ((x - mu2)**2 + gamma2**2))

# Initial guess for the parameters in the double Lorentzian fit:
# a1: amplitude of the first (main) peak
# mu1: center position of the first peak (in seconds)
# gamma1: half-width at half-maximum of the first peak
# a2: amplitude of the second (shoulder) peak
# mu2: center position of the second peak (in seconds)
# gamma2: half-width at half-maximum of the second peak
initial_guess = [1, 0.00, 0.002, 0.25, -0.10, 0.002]

# Fit the model to the data
params_L, _ = curve_fit(double_lorentzian, X_fit, Y_fit_norm, p0=initial_guess)

# Extract peak centers
mu1, mu2 = params_L[1], params_L[4]
temporal_spacing_L = abs(mu2 - mu1)


# Output the temporal spacing
print(f"The temporal spacing between the two Lorentzian peaks is approximately {temporal_spacing_L:.6f} seconds.")

# Define a double Voigt model
def double_voigt(x, a1, mu1, sigma1, gamma1, a2, mu2, sigma2, gamma2):
    return (a1 * voigt_profile(x - mu1, sigma1, gamma1) +
            a2 * voigt_profile(x - mu2, sigma2, gamma2))

# Initial guess for the parameters in the double Voigt fit:
# a1: amplitude of the first (main) peak
# mu1: center position of the first peak
# sigma1: Gaussian width of the first peak
# gamma1: Lorentzian width of the first peak
# a2: amplitude of the second (shoulder) peak
# mu2: center position of the second peak
# sigma2: Gaussian width of the second peak
# gamma2: Lorentzian width of the second peak
initial_guess = [1, -0.005, 0.002, 0.002, 0.25, -0.11, 0.0005, 0.002]

# Fit the model to the data
params_V, _ = curve_fit(double_voigt, X_fit, Y_fit_norm, p0=initial_guess)

# Extract peak centers
mu1, mu2 = params_V[1], params_V[5]
temporal_spacing_V = abs(mu2 - mu1)


# Output the temporal spacing
print(f"The temporal spacing between the two Voigt peaks is approximately {temporal_spacing_V:.6f} seconds.")


# Frequency axis conversion
dt = temporal_spacing_V
freq_spacing = 78.05  # MHz from Rydberg-Ritz formula (manuscript equation 17)
X_freq = (X_fit * freq_spacing) / dt

# Plot the results
plt.figure(figsize=(10, 6))
plt.rcParams.update({'font.family': 'serif', 'font.size': 14})
plt.subplots_adjust(left=0.15)
plt.plot(X_freq, Y_fit_norm, '-.',label='Normalized Data', color='blue', lw=1.5)
plt.plot(X_freq, double_lorentzian(X_fit, *params_L), label='Double Lorentzian Fit', color='red', lw=1.5)
plt.plot(X_freq, double_gaussian(X_fit, *params_G), label='Double Gaussian Fit', color='green', lw=1.5)
plt.plot(X_freq, double_voigt(X_fit, *params_V), label='Double Voigt Fit', color='purple', lw=1.5)
plt.xlabel('Coupling laser detuning [MHz]')
plt.ylabel('Probe Transmission (arb.)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(frameon=False)
plt.tight_layout()
plt.show()