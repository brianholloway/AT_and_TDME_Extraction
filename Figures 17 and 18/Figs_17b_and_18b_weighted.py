"""
Supplementary Material for Journal Submission
------------------------------------------------
This script reproduces Figures 17 (b) and 18 (b)

This script analyzes RF field delivery efficiency in a Rydberg atom experiment
as a function of principal quantum number (n). It compares two configurations:
1. Configuration 1: RF horn oriented off-axis from the vapor cell.
2. Configuration 2: RF horn oriented perpendicular to the vapor cell.

Updates in this version:
- Per-point uncertainty on eta using Ratio arrays R and R_new (derived from slope uncertainties).
- Weighted quartic fits with 95% confidence interval bands for the quartic model (replacing the prior ±5% uniform band).
- Error bars on empirical points use sigma_i = R_i * eta_i.

Dependencies:
  numpy, matplotlib, scipy, sklearn
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jn
import scipy.constants as const
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import norm
import pathlib
from pathlib import Path

try:
    PROJECT_DIR = Path(__file__).parent.resolve()  # works in .py scripts
except NameError:
    PROJECT_DIR = Path.cwd().resolve()             # works in Jupyter

print(PROJECT_DIR)

# -----------------------------
# Physical constants and setup
# -----------------------------
c = const.c  # Speed of light [m/s]
cell_width = 0.0175  # Vapor cell width [m]
cell_length = 0.075  # Vapor cell length [m]
k = 1  # wave-number multiplier for phase term

# -----------------------------
# Helper: Weighted quartic fit
# -----------------------------
def weighted_quartic_fit(x, y, sigma, x_min=None, x_max=None, label_prefix='Quartic'):
    """Weighted least squares quartic fit with 95% CI band.
    x, y: data arrays; sigma: per-point absolute uncertainties.
    Returns dict with fit values and CI band; also generates elements for plotting.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    sigma = np.asarray(sigma)

    # Design matrix: columns [x^4, x^3, x^2, x, 1]
    X = np.vstack([x**4, x**3, x**2, x, np.ones_like(x)]).T
    W = np.diag(1.0 / (sigma**2))

    # Weighted normal equations
    XTWX = X.T @ W @ X
    XTWy = X.T @ W @ y
    beta = np.linalg.solve(XTWX, XTWy)  # shape (5,)

    # Grid for plotting
    if x_min is None: x_min = float(x.min())
    if x_max is None: x_max = float(x.max())
    x_fit = np.linspace(x_min, x_max, 500)
    X_fit = np.vstack([x_fit**4, x_fit**3, x_fit**2, x_fit, np.ones_like(x_fit)]).T

    # Fitted curve
    y_fit = X_fit @ beta

    # 95% CI band using propagation with known heteroscedastic variances
    cov_beta = np.linalg.inv(XTWX)
    var_pred = np.sum(X_fit @ cov_beta * X_fit, axis=1)  # diag(X_fit Cov X_fit^T)
    z = norm.ppf(0.975)
    ci_half = z * np.sqrt(var_pred)
    y_lo = y_fit - ci_half
    y_hi = y_fit + ci_half

    poly_text = (
        f"E(n) = {beta[0]:+.4e}·n^4 {beta[1]:+.4e}·n^3 {beta[2]:+.4e}·n^2 "
        f"{beta[3]:+.4e}·n {beta[4]:+.4e}"
    ).replace('+', '+ ').replace('-', '- ')

    return {
        'beta': beta,
        'x_fit': x_fit,
        'y_fit': y_fit,
        'y_lo': y_lo,
        'y_hi': y_hi,
        'poly_text': poly_text
    }

# ============================================================
# CONFIGURATION 1: RF horn oriented off-axis
# ============================================================
print("########## Configuration 1 ##########")
# Empirical data: Principal quantum numbers and RF efficiency factors
D_old = np.array([53, 55, 68, 75, 85])
F_old = np.array([0.489, 0.73, 0.894, 0.883, 0.400])

# Ratio from slope uncertainties (provided externally)
slope_old = np.array([11.02, 17.68, 33.56, 40.01, 23.90])
slope_uncert_old = np.array([0.19, 0.72, 0.63, 1.25, 0.61])
R_old = slope_uncert_old / slope_old  # relative uncertainty per point
sigma_old = R_old * F_old             # absolute uncertainty on eta for error bars and WLS

# Weighted quartic
wls_old = weighted_quartic_fit(D_old, F_old, sigma_old, label_prefix='Config 1 Quartic')

# Bessel + cosine physical model (same as your original script)
f = np.array([14.232, 12.694, 10.186, 9.673, 6.585, 5.0987, 4.893, 3.712, 3.350]) * 1e9
l = c / f
n_old = D_old
eta_old_baseline = wls_old['y_fit'][np.argmin(np.abs(wls_old['x_fit'] - n_old[0]))]  # not used; keep quartic for baseline only if needed

# New data point to test against the fit
new_D = 60
new_F = 0.95

# Parameters for physical model
s = 1.64  # Off-axis antinode factor
z_ax = 0.60  # Axial overlap factor

# Compute scaled radii for Bessel argument
R1 = cell_width / (s * l)
R2 = (cell_length * z_ax) / l
indices = [0, 1, 4, 6, 8]  # Select indices corresponding to n_old
RR1 = 2 * np.pi * np.array([R1[i] for i in indices])
RR2 = 2 * np.pi * np.array([R2[i] for i in indices])
# Interpolators for RR1 and RR2 as functions of n
RR1_interp_old = interp1d(n_old, RR1, kind='cubic', fill_value='extrapolate')
RR2_interp_old = interp1d(n_old, RR2, kind='cubic', fill_value='extrapolate')

# Physical model: combination of Bessel functions and cosine term

def bessel_sine_Amp_old(n, w1, w2, w3, w4):
    rr1 = RR1_interp_old(n)
    rr2 = RR2_interp_old(n)
    return w1 * jn(0, rr1) + w2 * jn(1, rr1) + w3 * np.cos(k * rr2 + w4)

# Fit physical model to the quartic's values at the original points
eta_old_at_points = np.poly1d(np.polyfit(D_old, F_old, deg=4))(n_old)  # unchanged baseline for model fitting
initial_guess = [0.3, 0.3, 1.0, 1.0]
popt_old, pcov_old = curve_fit(bessel_sine_Amp_old, n_old, eta_old_at_points, p0=initial_guess)
J_fit_old = bessel_sine_Amp_old(n_old, *popt_old)
r2_old = r2_score(eta_old_at_points, J_fit_old)
rmse_old = np.sqrt(mean_squared_error(eta_old_at_points, J_fit_old))
print("Configuration 1 Optimized Weights:")
for i, (w, u) in enumerate(zip(popt_old, np.sqrt(np.diag(pcov_old)))):
    print(f"w{i+1} = {w:.4f} ± {u:.4f}")

# Visualization with confidence bands
n_smooth_old = np.linspace(D_old.min(), D_old.max(), 500)
J_smooth_old = bessel_sine_Amp_old(n_smooth_old, *popt_old)

# Propagate parameter covariance for physical model band
J_rows = []
w3_old, w4_old = popt_old[2], popt_old[3]
for n in n_smooth_old:
    rr1 = RR1_interp_old(n)
    rr2 = RR2_interp_old(n)
    J_rows.append([
        jn(0, rr1),
        jn(1, rr1),
        np.cos(k * rr2 + w4_old),
        -w3_old * np.sin(k * rr2 + w4_old)
    ])
J_rows = np.array(J_rows)
pred_var_old = np.sum(J_rows @ pcov_old * J_rows, axis=1)
pred_std_old = np.sqrt(pred_var_old)
z_score = 1.96
lower_band_old = J_smooth_old - z_score * pred_std_old
upper_band_old = J_smooth_old + z_score * pred_std_old

# Plot Config 1: quartic weighted band + physical model band + per-point error bars
fig1, ax1 = plt.subplots(figsize=(10, 6))
plt.rcParams.update({'font.family': 'serif', 'font.size': 14})
ax1.errorbar(D_old, F_old, yerr=sigma_old, fmt='o', markersize=9, color='black',
             label='Empirical Data (±σ = R·η)', capsize=5)
ax1.plot(wls_old['x_fit'], wls_old['y_fit'], color='blue', label='$\\eta_q$ (weighted quartic)')
ax1.fill_between(wls_old['x_fit'], wls_old['y_lo'], wls_old['y_hi'], color='blue', alpha=0.20,
                 label='Quartic 95% CI band')
ax1.plot(n_smooth_old, J_smooth_old, label=f'$\\eta_p$ (s={s:.2f}, z={z_ax:.2f})', color='orange', linestyle='--')
ax1.fill_between(n_smooth_old, lower_band_old, upper_band_old, color='orange', alpha=0.30,
                 label='Physical model 95% band')
ax1.errorbar(new_D, new_F, yerr=0.0163680, fmt='ro', markersize=10, label='Validation Point', capsize=5) # Configuration 1 validation point
ax1.set_xlabel('Principal Quantum Number (n)')
ax1.set_ylabel('$\\eta$')
ax1.legend(loc='upper right', frameon=False)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.text(0.95, 0.05, '(b)', transform=plt.gca().transAxes,
         fontsize=16, fontweight='bold', ha='right', va='bottom')
fig1.tight_layout()
# fig1.savefig('Fig_17_Config1_weighted.png', dpi=300)
# fig1.savefig('Fig_17_Config1_weighted.eps')
# fig1.savefig('Fig_17_Config1_weighted.pdf')

# ============================================================
# CONFIGURATION 2: RF horn oriented perpendicular to the vapor cell
# ============================================================
print("########## Configuration 2 ##########")
D_new = np.array([55, 59, 68, 74, 82])  # principal quantum number
F_new = np.array([0.93, 1.05, 1.10, 1.40, 1.66])  # efficiency factor

# New data point tested against the fit
new_D = np.array([60, 66])
new_F = np.array([1.025, 1.069])

# Ratio from slope uncertainties for new config
slope_new = np.array([22.83, 30.90, 41.11, 63.17, 92.41])
slope_uncert_new = np.array([0.57, 0.57, 0.58, 1.24, 1.77])
R_new = slope_uncert_new / slope_new
sigma_new = R_new * F_new

# Weighted quartic for Config 2
wls_new = weighted_quartic_fit(D_new, F_new, sigma_new, label_prefix='Config 2 Quartic')

# Bessel + cosine physical model for Config 2
f = np.array([14.232, 12.694, 10.186, 9.673, 6.585, 5.0987, 4.893, 3.712, 3.350]) * 1e9
l = c / f
n_new = D_new
# Use indices tailored for this configuration
s2 = 2.00
z2 = 0.40
R1 = cell_width / (s2 * l)
R2 = (cell_length * z2) / l
indices = [1, 2, 4, 5, 7]
RR1 = 2 * np.pi * np.array([R1[i] for i in indices])
RR2 = 2 * np.pi * np.array([R2[i] for i in indices])
RR1_interp_new = interp1d(n_new, RR1, kind='cubic', fill_value='extrapolate')
RR2_interp_new = interp1d(n_new, RR2, kind='cubic', fill_value='extrapolate')

def bessel_sine_Amp_new(n, w1, w2, w3, w4):
    rr1 = RR1_interp_new(n)
    rr2 = RR2_interp_new(n)
    return w1 * jn(0, rr1) + w2 * jn(1, rr1) + w3 * np.cos(k * rr2 + w4)

eta_new_at_points = np.poly1d(np.polyfit(D_new, F_new, deg=4))(n_new)
initial_guess = [0.1, 0.1, 0.0, 0.0]
popt_new, pcov_new = curve_fit(bessel_sine_Amp_new, n_new, eta_new_at_points, p0=initial_guess)
J_fit_new = bessel_sine_Amp_new(n_new, *popt_new)
r2_new = r2_score(eta_new_at_points, J_fit_new)
rmse_new = np.sqrt(mean_squared_error(eta_new_at_points, J_fit_new))
print("Configuration 2 Optimized Weights:")
for i, (w, u) in enumerate(zip(popt_new, np.sqrt(np.diag(pcov_new)))):
    print(f"w{i+1} = {w:.4f} ± {u:.4f}")

# Visualization with confidence bands for Config 2
n_smooth_new = np.linspace(D_new.min(), D_new.max(), 500)
J_smooth_new = bessel_sine_Amp_new(n_smooth_new, *popt_new)

J_rows = []
w3_new, w4_new = popt_new[2], popt_new[3]
for n in n_smooth_new:
    rr1 = RR1_interp_new(n)
    rr2 = RR2_interp_new(n)
    J_rows.append([
        jn(0, rr1),
        jn(1, rr1),
        np.cos(k * rr2 + w4_new),
        -w3_new * np.sin(k * rr2 + w4_new)
    ])
J_rows = np.array(J_rows)
pred_var_new = np.sum(J_rows @ pcov_new * J_rows, axis=1)
pred_std_new = np.sqrt(pred_var_new)
lower_band_new = J_smooth_new - z_score * pred_std_new
upper_band_new = J_smooth_new + z_score * pred_std_new

fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.errorbar(D_new, F_new, yerr=sigma_new, fmt='o', markersize=9, color='black',
             label='Empirical Data (±σ = R·η)', capsize=5)
ax2.plot(wls_new['x_fit'], wls_new['y_fit'], color='blue', label='$\\eta_q$ (weighted quartic)')
ax2.fill_between(wls_new['x_fit'], wls_new['y_lo'], wls_new['y_hi'], color='blue', alpha=0.20,
                 label='Quartic 95% CI band')
ax2.plot(n_smooth_new, J_smooth_new, label=f'$\\eta_p$ (s={s2:.2f}, z={z2:.2f})', color='orange', linestyle='--')
ax2.fill_between(n_smooth_new, lower_band_new, upper_band_new, color='orange', alpha=0.30,
                 label='Physical model 95% band')
ax2.errorbar(new_D[0], new_F[0], yerr=0.0169421, fmt='ro', markersize=10, label='Validation Point', capsize=5) # Configuration 2 validation point n60
ax2.set_xlabel('Principal Quantum Number (n)')
ax2.set_ylabel('$\\eta$')
ax2.legend(loc='upper left', frameon=False)
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.text(0.95, 0.05, '(b)', transform=plt.gca().transAxes,
         fontsize=16, fontweight='bold', ha='right', va='bottom')
fig2.tight_layout()
plt.show()
