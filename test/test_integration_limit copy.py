import numpy as np
from scipy.integrate import quad, fixed_quad
import time
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# --- PATH SETUP ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.physics import s_func

# --- CORE FUNCTION (Substituted) ---
def s_func_substituted(u, x, y, z, v, alpha, a):
    """ Transformed Integrand: s = u^2 (Removes singularity) """
    u = np.atleast_1d(u)
    s = u**2
    denom = 4 * alpha * s + a**2
    term_z = z**2 / (4 * alpha * s) if z != 0 else np.zeros_like(s)
    term_lat = (y**2 + (x + v * s)**2) / denom
    return 2 * np.exp(-term_z - term_lat) / denom

# --- VISUALIZATION ---
def run_visualization():
    print(">>> GENERATING 4-PANEL ANALYSIS (With Heatmap)...")
    
    # 1. Parameters
    P, v, a = 200.0, 1.0, 50e-6
    alpha = 8e-6
    x, y, z = 0, 0, 0 
    t_dwell = a / v
    
    # --- DATA GENERATION ---
    
    # A. Curves (Row 1)
    s_limit = 2.0 * t_dwell
    s_vals = np.linspace(0, s_limit, 1000)
    s_vals_safe = np.maximum(s_vals, 1e-12) 
    y_orig = s_func(s_vals_safe, x, y, z, v, alpha, a)
    
    u_vals = np.linspace(0, np.sqrt(s_limit), 1000)
    y_sub = s_func_substituted(u_vals, x, y, z, v, alpha, a)
    
    # B. Ground Truth (High Precision)
    print("   -> Calculating Ground Truth...")
    limit_truth = 100 * t_dwell
    val_truth, _ = quad(s_func, 1e-12, limit_truth, args=(x, y, z, v, alpha, a), epsabs=1e-12)

    # C. Line Plot Data (Row 2, Left)
    lim_s_fixed = 10 * t_dwell
    lim_u_fixed = np.sqrt(lim_s_fixed)
    n_steps_line = [5, 10, 15, 20, 30, 40, 50]
    err_orig_list = []
    err_sub_list = []
    
    for n in n_steps_line:
        v1, _ = fixed_quad(s_func, 1e-10, lim_s_fixed, args=(x, y, z, v, alpha, a), n=n)
        err_orig_list.append(abs(v1 - val_truth)/val_truth * 100)
        
        v2, _ = fixed_quad(s_func_substituted, 0, lim_u_fixed, args=(x, y, z, v, alpha, a), n=n)
        err_sub_list.append(abs(v2 - val_truth)/val_truth * 100)

    # D. Heatmap Data (Row 2, Right)
    print("   -> Generating Heatmap Data...")
    multipliers = [2, 5, 10, 15, 20, 30, 50] # Y-axis
    steps_grid = [5, 10, 20, 30, 40, 50, 60] # X-axis
    
    error_matrix = np.zeros((len(multipliers), len(steps_grid)))
    
    for i, mult in enumerate(multipliers):
        lim_u = np.sqrt(mult * t_dwell)
        for j, n in enumerate(steps_grid):
            val, _ = fixed_quad(s_func_substituted, 0, lim_u, args=(x, y, z, v, alpha, a), n=n)
            err_pct = abs(val - val_truth) / val_truth * 100
            # Clip extremely small errors to avoiding log(0) issues in plot
            error_matrix[i, j] = max(err_pct, 1e-8)

    # --- PLOTTING ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plt.subplots_adjust(hspace=0.3, wspace=0.25)
    
    # 1. Original (Top Left)
    spike_cutoff = s_func(t_dwell * 0.01, x, y, z, v, alpha, a)
    axes[0, 0].plot(s_vals * 1e6, y_orig, 'r-', linewidth=2)
    axes[0, 0].axvline(t_dwell * 1e6, color='k', linestyle='--', alpha=0.5, label='Dwell Time')
    axes[0, 0].set_title("1. Original (Singularity)", fontweight='bold')
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].set_xlabel("Time (µs)")
    axes[0, 0].set_ylim(0, spike_cutoff)
    axes[0, 0].text(s_limit*1e6*0.05, spike_cutoff*0.9, "<- Spike clipped", color='red')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # 2. Substituted (Top Right)
    axes[0, 1].plot(u_vals, y_sub, 'b-', linewidth=2)
    axes[0, 1].axvline(np.sqrt(t_dwell), color='k', linestyle='--', alpha=0.5, label='Sqrt(Dwell)')
    axes[0, 1].set_title("2. Substituted (Smooth)", fontweight='bold')
    axes[0, 1].set_xlabel("Transformed u")
    axes[0, 1].set_ylim(0, np.max(y_sub)*1.1)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # 3. Convergence (Bottom Left)
    axes[1, 0].plot(n_steps_line, err_orig_list, 'r-o', label='Original')
    axes[1, 0].plot(n_steps_line, err_sub_list, 'b-o', label='Substituted')
    axes[1, 0].axhline(0.1, color='gray', linestyle=':', label='0.1% Target')
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_title("3. Error Convergence", fontweight='bold')
    axes[1, 0].set_xlabel("Steps (N)")
    axes[1, 0].set_ylabel("Error % (Log Scale)")
    axes[1, 0].grid(True, which="both", alpha=0.3)
    axes[1, 0].legend()

    # 4. Heatmap (Bottom Right)
    # Using LogNorm to visualize small errors clearly
    im = axes[1, 1].imshow(error_matrix, interpolation='nearest', origin='lower',
                           norm=LogNorm(vmin=1e-6, vmax=1.0), cmap='inferno_r')
    
    # Ticks
    axes[1, 1].set_xticks(np.arange(len(steps_grid)))
    axes[1, 1].set_xticklabels(steps_grid)
    axes[1, 1].set_yticks(np.arange(len(multipliers)))
    axes[1, 1].set_yticklabels(multipliers)
    
    axes[1, 1].set_title("4. Error Heatmap (Substituted)", fontweight='bold')
    axes[1, 1].set_xlabel("Steps (N)")
    axes[1, 1].set_ylabel("Limit Multiplier (x Dwell)")
    
    # Colorbar
    cbar = plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
    cbar.set_label('Error %')

    # Annotate Sweet Spot
    axes[1, 1].text(len(steps_grid)-1, len(multipliers)-1, "Expensive", ha='right', va='top', color='white', fontsize=8)
    axes[1, 1].text(0, 0, "Inaccurate", ha='left', va='bottom', color='black', fontsize=8)

    plt.tight_layout()
    
    out_path = os.path.join(os.path.dirname(__file__), 'output', "s_func_4panel_analysis.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    print(f"Plot saved to: {out_path}")

if __name__ == "__main__":
    run_visualization()