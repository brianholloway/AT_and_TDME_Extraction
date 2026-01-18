
"""
RF Field + Gaussian Beam Interaction Regions (Figure 16)
--------------------------------------------------------
This script reproduces the two-panel figure (Configuration 1 & 2) used in the manuscript.
It computes an RF field map in a cylindrical Rb vapor cell, overlays a Gaussian
laser-beam profile, and highlights the region where BOTH the RF field and the
beam exceed a specified fraction of their respective maxima ("interaction region").

"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy.special import jn  # Bessel functions J_n used for radial structure

# -----------------------------------------------------------------------------
# Global settings and physical inputs
# -----------------------------------------------------------------------------

# Threshold that defines "interaction region" in a relative sense:
# - RF field (normalized to its *max* in the panel) must be > interaction_threshold
# - Gaussian beam intensity (normalized to its *max* in the panel) must be > interaction_threshold
interaction_threshold = 0.25

# Autler–Townes transition frequencies (Hz)
#   n-values:    [53,   55,     59,    60,   68,     74,    75,    82,    85]
f = np.array([14.232, 12.694, 10.186, 9.673, 6.585, 5.0987, 4.893, 3.712, 3.350]) * 1e9
n_values = [53, 55, 59, 60, 68, 74, 75, 82, 85]  # labels for reference/debug

# Convenience list to select the index N of the principal quantum number
W = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])

######### Select Principal Quantum Number and Transition Frequency #########
N = W[3]  # <- CHANGE THIS INDEX to switch n (currently n = 60)
############################################################################

# -----------------------------------------------------------------------------
# Vapor cell geometry (mm)
# -----------------------------------------------------------------------------
cell_diameter = 17.5
cell_length = 75.0

# -----------------------------------------------------------------------------
# RF configuration parameters
#   Each configuration uses:
#   - s : radial scale factor for Bessel arguments
#   - z : axial scaling factor for standing-wave phase
#   - w1, w2 : Bessel weights (J0, J1)
#   - w3, w4 : axial standing-wave weights (cos, sin)
# -----------------------------------------------------------------------------
configs = {
    "Configuration 1": {"s": 1.64, "z": 1.0, "w1": -0.1232, "w2": 1.7235, "w3": 0.0335, "w4": -0.1068},
    "Configuration 2": {"s": 2.00, "z": 1.0, "w1": 1.2560, "w2": 1.0895, "w3": 0.0137, "w4": 0.3117}
}

# -----------------------------------------------------------------------------
# Computational grids
#   r: radial coordinate from axis to cell wall (0 → R)
#   z: axial coordinate along the cell (0 → L)
# -----------------------------------------------------------------------------
r = np.linspace(0, cell_diameter/2, 100)
z = np.linspace(0, cell_length, 200)
r_grid, z_grid = np.meshgrid(r, z)

# -----------------------------------------------------------------------------
# Gaussian coupling beam parameters (mm)
#   - beam_waist: 1/e^2 radius
#   - offset: for Configuration 1 the beam is offset off-axis (positive r)
# -----------------------------------------------------------------------------
beam_waist = 2.76  # mm, 1/e^2 waist
offset = (17.5/1.64) - (17.5/2)  # ≈ 1.92 mm off-axis for Config 1 from working geometry

# -----------------------------------------------------------------------------
# RF wave number (1/mm) for selected transition frequency f[N]
# -----------------------------------------------------------------------------
k = (2 * np.pi * f[N] / 299792458.0) / 1000.0  # convert 1/m → 1/mm

# -----------------------------------------------------------------------------
# RF field model
#   Combines:
#   - radial structure: J0(kr/s), J1(kr/s)
#   - axial standing waves: cos(k L z_s), sin(k L z_s) with z_s = (z/L)*z_param
#   We take |sum| to represent field magnitude up to a global scaling (arb. units).
# -----------------------------------------------------------------------------
def rf_field(r, z, config):
    s = config["s"]
    z_param = config["z"]
    w1, w2, w3, w4 = config["w1"], config["w2"], config["w3"], config["w4"]

    # Scale axial coordinate so that one full cell length corresponds to z_param
    z_scaled = (z / cell_length) * z_param

    # Weighted sum of radial and axial basis functions
    j_sum = (
        w1 * jn(0, k * r / s) +
        w2 * jn(1, k * r / s) +
        w3 * np.cos(k * cell_length * z_scaled) +
        w4 * np.sin(k * cell_length * z_scaled)
    )
    return np.abs(j_sum)

# -----------------------------------------------------------------------------
# Compute RF fields for both configurations
# -----------------------------------------------------------------------------
field1 = rf_field(r_grid, z_grid, configs["Configuration 1"])
field2 = rf_field(r_grid, z_grid, configs["Configuration 2"])

# Common normalization bounds for both panels
vmin = min(field1.min(), field2.min())
vmax = max(field1.max(), field2.max())

# Normalize fields to their own maxima for thresholding
field1_norm = field1 / field1.max()
field2_norm = field2 / field2.max()

# -----------------------------------------------------------------------------
# Build Gaussian beam profiles across radius (symmetric coordinate for plotting)
# -----------------------------------------------------------------------------
radial_positions = np.linspace(-cell_diameter/2, cell_diameter/2, 400)
gaussian_profile1 = np.exp(-2 * (radial_positions - offset)**2 / (beam_waist**2))  # offset beam (Config 1)
gaussian_profile2 = np.exp(-2 * (radial_positions)**2          / (beam_waist**2))  # centered beam (Config 2)

# Resize/replicate 1D Gaussians along z to match the 2D (z × r) fields
# (interpolated only for r ≥ 0, using |r| symmetry from the positive half)
beam_profile1_resized = np.zeros_like(field1)
beam_profile2_resized = np.zeros_like(field2)
for i in range(len(z)):
    beam_profile1_resized[i, :] = np.interp(r, np.abs(radial_positions[200:]), gaussian_profile1[200:])
    beam_profile2_resized[i, :] = np.interp(r, np.abs(radial_positions[200:]), gaussian_profile2[200:])

# Normalize the beam profiles for thresholding (relative intensity)
beam_profile1_norm = beam_profile1_resized / np.max(beam_profile1_resized)
beam_profile2_norm = beam_profile2_resized / np.max(beam_profile2_resized)

# -----------------------------------------------------------------------------
# Interaction regions (boolean masks)
#   Points where BOTH:
#     RF normalized field > threshold  AND
#     Gaussian normalized intensity > threshold
# -----------------------------------------------------------------------------
interaction_region1 = (field1_norm > interaction_threshold) & (beam_profile1_norm > interaction_threshold)
interaction_region2 = (field2_norm > interaction_threshold) & (beam_profile2_norm > interaction_threshold)

# -----------------------------------------------------------------------------
# Figure and axes (two rows: Config 1 on top, Config 2 on bottom)
# -----------------------------------------------------------------------------
plt.rcParams.update({'font.family': 'serif', 'font.size': 14})
fig = plt.figure(figsize=(8, 9))
gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0.0001)

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
levels = 20  # number of contour levels (visual only)

# Shared normalization so both panels share the same color scale
norm = plt.Normalize(vmin=vmin, vmax=vmax)

# -----------------------------------------------------------------------------
# Panel 1: Configuration 1
# -----------------------------------------------------------------------------
# Draw field on +r and mirrored -r to show full diameter
ax1.contourf(z_grid,  r_grid, field1, levels=levels, cmap='YlOrRd', alpha=0.8, norm=norm)
ax1.contourf(z_grid, -r_grid, field1, levels=levels, cmap='YlOrRd', alpha=0.8, norm=norm)

# Debug prints (useful when switching n)
print(f"Selected frequency: {f[N]/1e9:.4f} GHz (n = {n_values[N]})")
print(f"RF wave number k: {k:.6f} 1/mm")
print(f"Interaction region 1 has points: {np.any(interaction_region1)}")

# Highlight interaction region (only positive radial side by design choice)
interaction_mask1 = np.ma.masked_where(~interaction_region1, np.ones_like(field1))
ax1.contour (z_grid,  r_grid, interaction_mask1, levels=[0.5], colors=['lime'], linewidths=3)
ax1.contourf(z_grid,  r_grid, interaction_mask1, levels=[0.5, 1.5], colors=['lime'], alpha=0.3)

# Cell outline
ax1.plot([0, cell_length], [ cell_diameter/2,  cell_diameter/2], 'k-', lw=2)
ax1.plot([0, cell_length], [-cell_diameter/2, -cell_diameter/2], 'k-', lw=2)
ax1.plot([0, 0], [-cell_diameter/2,  cell_diameter/2], 'k-', lw=2)
ax1.plot([cell_length, cell_length], [-cell_diameter/2,  cell_diameter/2], 'k-', lw=2)

ax1.set_title("Configuration 1")
ax1.set_ylabel('Radial Position (mm)')
ax1.set_xticks([])      # no x ticks in top panel
ax1.set_xlabel('')      # and no x label
ax1.set_xlim(0, cell_length)
ax1.set_ylim(-cell_diameter/2, cell_diameter/2)

# Gaussian beam shading (visual aid)
radial_positions = np.linspace(-cell_diameter/2, cell_diameter/2, 400)
gaussian_profile1 = np.exp(-2 * (radial_positions - offset)**2 / (beam_waist**2))
beam_shading = np.tile(gaussian_profile1, (len(z), 1))
ax1.imshow(
    beam_shading.T,
    extent=[0, cell_length, -cell_diameter/2, cell_diameter/2],
    origin='lower', cmap='Blues', alpha=0.25, aspect='auto', zorder=1
)

# Dashed beam “stripe” and label (offset beam)
ax1.add_patch(plt.Rectangle((0, offset - beam_waist/2), cell_length, beam_waist,
                            fill=False, edgecolor='blue', linestyle='--', linewidth=2))
ax1.text(cell_length/2, offset + beam_waist/2 + 1.5, "Coupling Beam (2.76 mm waist)",
         ha='center', color='black', fontsize=12)

# -----------------------------------------------------------------------------
# Panel 2: Configuration 2
# -----------------------------------------------------------------------------
ax2.contourf(z_grid,  r_grid, field2, levels=levels, cmap='YlOrRd', alpha=0.8, norm=norm)
ax2.contourf(z_grid, -r_grid, field2, levels=levels, cmap='YlOrRd', alpha=0.8, norm=norm)

print(f"Interaction region 2 has points: {np.any(interaction_region2)}")

# Highlight interaction region (both ±r sides)
interaction_mask2 = np.ma.masked_where(~interaction_region2, np.ones_like(field2))
ax2.contour (z_grid,  r_grid, interaction_mask2, levels=[0.5], colors=['lime'], linewidths=3)
ax2.contour (z_grid, -r_grid, interaction_mask2, levels=[0.5], colors=['lime'], linewidths=3)
ax2.contourf(z_grid,  r_grid, interaction_mask2, levels=[0.5, 1.5], colors=['lime'], alpha=0.3)
ax2.contourf(z_grid, -r_grid, interaction_mask2, levels=[0.5, 1.5], colors=['lime'], alpha=0.3)

# Cell outline
ax2.plot([0, cell_length], [ cell_diameter/2,  cell_diameter/2], 'k-', lw=2)
ax2.plot([0, cell_length], [-cell_diameter/2, -cell_diameter/2], 'k-', lw=2)
ax2.plot([0, 0], [-cell_diameter/2,  cell_diameter/2], 'k-', lw=2)
ax2.plot([cell_length, cell_length], [-cell_diameter/2,  cell_diameter/2], 'k-', lw=2)

ax2.set_title("Configuration 2")
ax2.set_xlabel('Axial Position (mm)')
ax2.set_ylabel('Radial Position (mm)')
ax2.set_xlim(0, cell_length)
ax2.set_ylim(-cell_diameter/2, cell_diameter/2)

# Gaussian beam shading (centered beam)
gaussian_profile2 = np.exp(-2 * (radial_positions**2) / (beam_waist**2))
beam_shading2 = np.tile(gaussian_profile2, (len(z), 1))
ax2.imshow(
    beam_shading2.T,
    extent=[0, cell_length, -cell_diameter/2, cell_diameter/2],
    origin='lower', cmap='Blues', alpha=0.25, aspect='auto', zorder=1
)

# Dashed beam “stripe” and label (centered beam)
ax2.add_patch(plt.Rectangle((0, -beam_waist/2), cell_length, beam_waist,
                            fill=False, edgecolor='blue', linestyle='--', linewidth=2))
ax2.text(cell_length/2, beam_waist/2 + 1.5, "Coupling Beam (2.76 mm waist)",
         ha='center', color='black', fontsize=12)

# -----------------------------------------------------------------------------
# Legend and shared colorbar
# -----------------------------------------------------------------------------
legend_elements = [Line2D(
    [0], [0], color='green', lw=2,
    label=rf'Interaction Region ($\mathrm{{RF}} > {interaction_threshold:.2f}\times\mathrm{{max}}$)'
)]
ax2.legend(handles=legend_elements, loc='lower right', frameon=False)

# Shared colorbar using the same normalization as the filled contours
sm = plt.cm.ScalarMappable(norm=norm, cmap='YlOrRd')
sm.set_array([])  # required by Matplotlib when providing a ScalarMappable
cbar = fig.colorbar(sm, ax=[ax1, ax2], orientation='vertical', fraction=0.05, pad=0.02)
cbar.set_label('RF Field Intensity (arb. units)')

# -----------------------------------------------------------------------------
# Render (optional: enable saving instead)
# -----------------------------------------------------------------------------

# Optional saving (uncomment to write to disk)
# filename = f'figure_n{n_values[N]}_thresh{interaction_threshold}.pdf'
# plt.savefig(filename, bbox_inches='tight')
