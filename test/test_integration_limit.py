import numpy as np
from scipy.integrate import fixed_quad, quad
import logging
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt


# Add src to path to import your original physics function
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.physics import s_func

# --- 1. LOGGING SETUP (Keep existing file) ---
def setup_logger():
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    # Using the same log file name as before
    log_file = os.path.join(log_dir, 'compare_funcs.log')
    
    logger = logging.getLogger("CompareFuncs")
    logger.setLevel(logging.INFO)
    
    # Clear handlers to prevent duplicate prints if run multiple times
    if logger.hasHandlers(): 
        logger.handlers.clear()
    
    # File Handler (Append mode 'a' is safer if you want history, but 'w' cleans it up)
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(logging.Formatter('%(message)s'))
    
    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter('%(message)s'))
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger, log_file

# --- 2. SUBSTITUTED FUNCTION (s = u^2) ---
def s_func_substituted(u, x, y, z, v, alpha, a):
    """
    Optimized version: s = u^2.
    Removes the 1/sqrt(s) singularity at 0.
    """
    u = np.atleast_1d(u)
    s = u**2
    
    denom_lateral = 4 * alpha * s + a**2
    
    # Safety for z term
    term_z = np.zeros_like(s)
    safe_mask = s > 1e-20
    if z != 0:
        term_z[safe_mask] = z**2 / (4 * alpha * s[safe_mask])
        term_z[~safe_mask] = np.inf 
    
    term_lateral = (y**2 + (x + v * s)**2) / denom_lateral
    exp_val = np.exp(-term_z - term_lateral)
    
    # Jacobian (2u) cancels 1/sqrt(s). Result: 2 * exp / denom
    return 2 * exp_val / denom_lateral

# --- 3. COMPARISON TEST ---
def run_comparison():
    logger, log_path = setup_logger()
    logger.info(">>> COMPARING ORIGINAL vs. SUBSTITUTED (Values & Errors)")
    logger.info("Objective: Show actual integral values to understand why methods fail.\n")

    # Parameters
    P, v, a = 200.0, 1.0, 50e-6
    alpha = 8e-6
    x, y, z = 0, 0, -20e-6
    
    # A. Ground Truth
    smart_limit = (a / v) * 100.0 # 0.005s
    truth_val, _ = quad(s_func, 1e-12, smart_limit, args=(x, y, z, v, alpha, a), limit=200)
    
    logger.info(f"Ground Truth (s_func + Smart Limit): {truth_val:.6e}\n")

    # B. Test Conditions
    limits_to_test = [
        smart_limit,        # 0.005s (Ideal)
        0.05,               # 0.05s
        1.0,                # 1.0s
        100.0,              # 100s
        1e10                # Infinity
    ]
    
    N_points = 20  # Fast simulation
    
    # C. Run Comparison
    data = []
    
    for lim in limits_to_test:
        # 1. Original (s)
        try:
            val_orig, _ = fixed_quad(s_func, 1e-9, lim, args=(x,y,z,v,alpha,a), n=N_points)
            err_orig = abs(val_orig - truth_val) / truth_val * 100 if truth_val else 0
        except Exception:
            val_orig = 0.0
            err_orig = 100.0

        # 2. Substituted (u^2)
        try:
            val_sub, _ = fixed_quad(s_func_substituted, 0, np.sqrt(lim), args=(x,y,z,v,alpha,a), n=N_points)
            err_sub = abs(val_sub - truth_val) / truth_val * 100 if truth_val else 0
        except Exception:
            val_sub = 0.0
            err_sub = 100.0
            
        data.append({
            "Limit": lim,
            "Orig Val": val_orig,
            "Orig Err%": err_orig,
            "Sub Val": val_sub,
            "Sub Err%": err_sub
        })

    # D. Display Results with VALUES
    df = pd.DataFrame(data)
    
    # Wider formatting to fit values
    header = f"{'Limit (s)':<10} | {'Orig Val':<12} | {'Orig Err%':<10} | {'Sub Val':<12} | {'Sub Err%':<10} | {'Notes':<15}"
    logger.info(header)
    logger.info("-" * 95)
    
    for _, row in df.iterrows():
        orig_val = row['Orig Val']
        orig_err = row['Orig Err%']
        sub_val = row['Sub Val']
        sub_err = row['Sub Err%']
        lim = row['Limit']
        
        # Note logic
        if sub_err < 1.0: note = "Sub Perfect"
        elif sub_err < 10.0: note = "Sub OK"
        elif sub_val == 0.0: note = "Sub Failed (0)"
        else: note = "Sub Inacc."
        
        if orig_val == 0.0:
            note += " / Orig 0"
        elif orig_err > 50:
            note += " / Orig Bad"
            
        logger.info(f"{lim:<10.1e} | {orig_val:<12.4e} | {orig_err:9.4f}% | {sub_val:<12.4e} | {sub_err:9.4f}% | {note}")

    logger.info("-" * 95)
    logger.info("Log saved to: " + log_path)

def s_func_original(s, x, y, z, v, alpha, a):
    """ 
    Original Eagar-Tsai Integrand.
    Singular at s=0 due to 1/sqrt(s).
    """
    # Avoid true 0 to prevent div/0 in plot, start at tiny epsilon
    s = np.maximum(s, 1e-12) 
    
    denom_lateral = 4 * alpha * s + a**2
    
    # z term (z is negative depth)
    term_z = z**2 / (4 * alpha * s)
    
    # lateral term (moving source)
    term_lateral = (y**2 + (x + v * s)**2) / denom_lateral
    
    exp_val = np.exp(-term_z - term_lateral)
    
    # The trouble maker: 1 / sqrt(s)
    return exp_val / (denom_lateral * np.sqrt(s))

def s_func_substituted(u, x, y, z, v, alpha, a):
    """ 
    Optimized Integrand (s = u^2).
    Smooth at u=0.
    """
    u = np.atleast_1d(u)
    s = u**2
    
    denom_lateral = 4 * alpha * s + a**2
    
    # Handle z term safely
    term_z = np.zeros_like(s)
    safe_mask = s > 1e-20
    if z != 0:
        term_z[safe_mask] = z**2 / (4 * alpha * s[safe_mask])
        term_z[~safe_mask] = np.inf 
    
    term_lateral = (y**2 + (x + v * s)**2) / denom_lateral
    
    exp_val = np.exp(-term_z - term_lateral)
    
    # Singularity removed: Just 2 * exp / denom
    return 2 * exp_val / denom_lateral

# --- 2. GENERATE PLOTS ---

def plot_perspective():
    print(">>> Generating S-Function Perspective Plots...")
    
    # Standard NiTi Parameters
    P, v, a = 200.0, 1.0, 50e-6
    alpha = 8e-6
    x, y, z = 0, 0, -20e-6 # 20um depth
    
    args = (x, y, z, v, alpha, a)
    
    # Define "Smart Limit" (0.005s) vs "Long Limit" (0.1s)
    smart_limit = (a/v) * 100 
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # --- PLOT 1: The Singularity (Original, Short Time) ---
    s_short = np.linspace(0, smart_limit, 1000)
    y_short = s_func_original(s_short, *args)
    
    axes[0].plot(s_short, y_short, 'r-', linewidth=2)
    axes[0].set_title("1. Original Function (Zoomed In)\nNote the Spike at s=0")
    axes[0].set_xlabel("Time Lag s (seconds)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)
    
    # --- PLOT 2: The "Empty Space" (Original, Long Time) ---
    s_long = np.linspace(0, 0.1, 1000) # 0.1s is still small, but huge for melt pool
    y_long = s_func_original(s_long, *args)
    
    axes[1].plot(s_long, y_long, 'r-', linewidth=2)
    axes[1].axvline(smart_limit, color='k', linestyle='--', label='Smart Limit')
    axes[1].fill_between(s_long, y_long, where=(s_long > smart_limit), color='gray', alpha=0.1)
    
    axes[1].set_title("2. The 'Needle' in the Haystack\nEverything right of dashed line is ~0")
    axes[1].set_xlabel("Time Lag s (seconds)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # --- PLOT 3: The Fix (Substituted) ---
    # Plotting against u (where s = u^2)
    u_vals = np.linspace(0, np.sqrt(smart_limit), 1000)
    y_sub = s_func_substituted(u_vals, *args)
    
    axes[2].plot(u_vals, y_sub, 'b-', linewidth=2)
    axes[2].set_title("3. Optimized Function (u = sqrt(s))\nSmooth! No Singularity.")
    axes[2].set_xlabel("Transformed Variable u")
    axes[2].grid(True, alpha=0.3)
    
    # Save
    out_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, "s_func_visualization.png")
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")

if __name__ == "__main__":
    run_comparison()
    plot_perspective()