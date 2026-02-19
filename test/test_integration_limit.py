import numpy as np
from scipy.integrate import quad, fixed_quad
import time
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

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
    print(">>> GENERATING 3-PANEL STORY PLOT...")
    
    # 1. Parameters
    P, v, a = 200.0, 1.0, 50e-6
    alpha = 8e-6
    x, y, z = 0, 0, 0 
    t_dwell = a / v
    
    # 2. Generate Curve Data
    s_limit = 2.0 * t_dwell
    
    # Original Data (s)
    s_vals = np.linspace(0, s_limit, 1000)
    s_vals_safe = np.maximum(s_vals, 1e-12) # Avoid div/0
    y_orig = s_func(s_vals_safe, x, y, z, v, alpha, a)
    
    # Substituted Data (u)
    u_vals = np.linspace(0, np.sqrt(s_limit), 1000)
    y_sub = s_func_substituted(u_vals, x, y, z, v, alpha, a)
    
    # 3. Generate Error Data
    print("   -> Calculating Error Convergence...")
    limit_truth = 100 * t_dwell
    val_truth, _ = quad(s_func, 1e-12, limit_truth, args=(x, y, z, v, alpha, a))
    
    # Sweep settings
    lim_s = 10 * t_dwell
    lim_u = np.sqrt(lim_s)
    n_steps = [5, 10, 15, 20, 30, 40, 50, 60]
    
    err_orig_list = []
    err_sub_list = []
    
    for n in n_steps:
        # Original Error
        val, _ = fixed_quad(s_func, 1e-10, lim_s, args=(x, y, z, v, alpha, a), n=n)
        err_orig_list.append(abs(val - val_truth)/val_truth * 100)
        
        # Substituted Error
        val, _ = fixed_quad(s_func_substituted, 0, lim_u, args=(x, y, z, v, alpha, a), n=n)
        err_sub_list.append(abs(val - val_truth)/val_truth * 100)

    # 4. Create Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # --- PLOT 1: Original (The Problem) ---
    # Manually cut the spike at 1% dwell time to see the tail
    spike_cutoff_val = s_func(t_dwell * 0.01, x, y, z, v, alpha, a)
    
    axes[0].plot(s_vals * 1e6, y_orig, 'r-', linewidth=2)
    axes[0].axvline(t_dwell * 1e6, color='k', linestyle='--', alpha=0.5, label='Dwell Time')
    
    axes[0].set_title("1. Original (Singularity)", fontweight='bold')
    axes[0].set_xlabel("Time (µs)")
    axes[0].set_ylabel("Amplitude (per sec)")
    axes[0].set_ylim(0, spike_cutoff_val) # Crop the infinity
    axes[0].text(s_limit*1e6*0.05, spike_cutoff_val*0.9, "<- Spike clipped", color='red')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # --- PLOT 2: Substituted (The Fix) ---
    # Auto-scale to its OWN maximum (Fixes the flatness)
    axes[1].plot(u_vals, y_sub, 'b-', linewidth=2)
    axes[1].axvline(np.sqrt(t_dwell), color='k', linestyle='--', alpha=0.5, label='Sqrt(Dwell)')
    
    axes[1].set_title("2. Substituted (Smooth)", fontweight='bold')
    axes[1].set_xlabel("Transformed u")
    axes[1].set_ylabel("Amplitude (per u)") # Different units!
    
    # FIX: Let it scale naturally, just add a tiny margin
    axes[1].set_ylim(0, np.max(y_sub) * 1.1) 
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # --- PLOT 3: Convergence (The Proof) ---
    axes[2].plot(n_steps, err_orig_list, 'r-o', label='Original')
    axes[2].plot(n_steps, err_sub_list, 'b-o', label='Substituted')
    
    axes[2].axhline(0.1, color='gray', linestyle=':', label='Target (0.1%)')
    axes[2].axhline(0.01, color='green', linestyle=':', label='Target (0.01%)')
    
    axes[2].set_yscale('log')
    axes[2].set_title("3. Error vs Steps (Log Scale)", fontweight='bold')
    axes[2].set_xlabel("Steps (N)")
    axes[2].set_ylabel("Error %")
    axes[2].grid(True, which="both", alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    
    out_path = os.path.join(os.path.dirname(__file__), 'output', "s_func_story_3panel.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    print(f"Plot saved to: {out_path}")

# Run
if __name__ == "__main__":
    run_visualization()