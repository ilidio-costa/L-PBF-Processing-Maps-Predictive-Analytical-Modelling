import sys
import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, fixed_quad
import logging

# --- Setup Paths ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

from src.physics import s_func_substituted

# --- Setup Logging & Output ---
log_dir = os.path.join(BASE_DIR, 'test', 'logs')
out_dir = os.path.join(BASE_DIR, 'test', 'output')
os.makedirs(log_dir, exist_ok=True)
os.makedirs(out_dir, exist_ok=True)

log_file = os.path.join(log_dir, 'directional_benchmark.log')
with open(log_file, 'w'): pass  # Clear old log

logging.basicConfig(filename=log_file, level=logging.INFO, format='%(message)s')

# ==============================================================================
# 1. Integration Methods
# ==============================================================================

def temp_redundant_stats(x, y, z, P, v, a, material, T_ambient=293.15):
    A_val, k, alpha = material['A'], material['k'], material['alpha']
    pre_factor = (A_val * P / (np.pi * k)) * np.sqrt(alpha / np.pi)
    def f(u): return float(np.squeeze(s_func_substituted(u, x, y, z, v, alpha, a)))

    u_peak_guess = np.sqrt(max(0, -x / v) + (a / v))
    if f(u_peak_guess) < 1e-20: return T_ambient
        
    upper_calc_limit = np.sqrt((abs(x)/v) * 3 + (x**2+y**2+z**2)/(4*alpha) + 100*(a/v))
    area, _ = quad(f, 0, upper_calc_limit, limit=100)
    if area < 1e-20: return T_ambient
        
    def u_times_f(u): return u * f(u)
    mean_u, _ = quad(u_times_f, 0, upper_calc_limit, limit=100)
    mean_u /= area
    
    def var_integrand(u): return ((u - mean_u)**2) * f(u)
    variance_u, _ = quad(var_integrand, 0, upper_calc_limit, limit=100)
    std_u = np.sqrt(abs(variance_u / area))
    
    limit_lower = max(0.0, mean_u - (4.0 * std_u))
    limit_upper = mean_u + (4.0 * std_u)

    integral_val, _ = fixed_quad(s_func_substituted, limit_lower, limit_upper, args=(x, y, z, v, alpha, a), n=60)
    return (pre_factor * integral_val) + T_ambient

def temp_adaptive_quad(x, y, z, P, v, a, material, T_ambient=293.15):
    A_val, k, alpha = material['A'], material['k'], material['alpha']
    pre_factor = (A_val * P / (np.pi * k)) * np.sqrt(alpha / np.pi)
    def f(u): return float(np.squeeze(s_func_substituted(u, x, y, z, v, alpha, a)))

    if f(np.sqrt(max(0, -x / v) + (a / v))) < 1e-20: return T_ambient
        
    upper_calc_limit = np.sqrt((abs(x)/v) * 3 + (x**2+y**2+z**2)/(4*alpha) + 100*(a/v))
    area, _ = quad(f, 0, upper_calc_limit, limit=100) 
    return (pre_factor * area) + T_ambient

def temp_fast_fixed_quad(x, y, z, P, v, a, material, T_ambient=293.15):
    A_val, k, alpha = material['A'], material['k'], material['alpha']
    pre_factor = (A_val * P / (np.pi * k)) * np.sqrt(alpha / np.pi)

    u_peak_guess = np.sqrt(max(0, -x / v) + (a / v))
    if float(np.squeeze(s_func_substituted(u_peak_guess, x, y, z, v, alpha, a))) < 1e-20:
        return T_ambient
        
    upper_calc_limit = np.sqrt((abs(x)/v) * 3 + (x**2+y**2+z**2)/(4*alpha) + 100*(a/v))
    integral_val, _ = fixed_quad(s_func_substituted, 0.0, upper_calc_limit, args=(x, y, z, v, alpha, a), n=250)
    return (pre_factor * integral_val) + T_ambient

# ==============================================================================
# 2. Benchmark Runner
# ==============================================================================

def load_ti64():
    """Loads just Ti64 for the nominal test, calculates alpha if needed."""
    mat_path = os.path.join(BASE_DIR, 'materials', 'Ti64.json')
    if not os.path.exists(mat_path):
        # Fallback dictionary if file not found
        return {"name": "Ti-6Al-4V", "rho": 4430, "C_p": 526, "k": 6.7, "alpha": 2.87e-06, "A": 0.40}
    
    with open(mat_path, 'r') as f:
        mat = json.load(f)
        if 'alpha' not in mat:
            mat['alpha'] = mat['k'] / (mat['rho'] * mat['C_p'])
        return mat

def run_directional_benchmark():
    mat_props = load_ti64()
    P, v, a = 250.0, 0.8, 40e-6
    
    # Define distance vector (0 to 1000 µm)
    distances = np.linspace(0, 1000e-6, 150)
    inv_sqrt3 = 1.0 / np.sqrt(3)

    # Dictionary of directions defining (x, y, z) for a given distance d
    # Note: Front, Side, and Depth have shorter relevant ranges, so we scale d down for them
    directions = {
        "1. Wake (-x)":                 lambda d: (-d, 0, 0),
        "2. Front (+x)":                lambda d: (d/5, 0, 0),
        "3. Side (+y)":                 lambda d: (0, d/5, 0),
        "4. Depth (-z)":                lambda d: (0, 0, -d/5),
        "5. Diagonal (-x, +y, -z)":     lambda d: (-d*inv_sqrt3, d*inv_sqrt3, -d*inv_sqrt3),
    }

    logging.info("===================================================================")
    logging.info(f"       3D DIRECTIONAL BENCHMARK | {mat_props.get('name', 'Ti64')} | P={P}W, v={v}m/s")
    logging.info("===================================================================\n")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    fig.suptitle(f"Directional Thermal Profiles (Ti-6Al-4V, P={P}W, v={v}m/s)", fontsize=16)

    for i, (title, coord_func) in enumerate(directions.items()):
        logging.info(f"--- ORIENTATION: {title} ---")
        logging.info(f"{'Method':<20} | {'Time (s)':<10} | {'Max Err (K)':<12} | {'RMSE (K)':<10} | {'Speedup'}")
        logging.info("-" * 80)
        
        ax = axes[i]
        
        # Calculate coordinates for this direction
        coords = [coord_func(d) for d in distances]
        x_vals, y_vals, z_vals = zip(*coords)
        
        # 1. Baseline
        t0 = time.time()
        T_base = np.array([temp_redundant_stats(x, y, z, P, v, a, mat_props) for x, y, z in coords])
        t_base = time.time() - t0
        
        # 2. Adaptive Quad
        t0 = time.time()
        T_quad = np.array([temp_adaptive_quad(x, y, z, P, v, a, mat_props) for x, y, z in coords])
        t_quad = time.time() - t0
        
        # 3. Fast Fixed Quad
        t0 = time.time()
        T_fixed = np.array([temp_fast_fixed_quad(x, y, z, P, v, a, mat_props) for x, y, z in coords])
        t_fixed = time.time() - t0

        # Error Metrics
        err_quad = np.abs(T_base - T_quad)
        err_fixed = np.abs(T_base - T_fixed)

        rmse_quad = np.sqrt(np.mean(err_quad**2))
        rmse_fixed = np.sqrt(np.mean(err_fixed**2))

        # Logging
        logging.info(f"{'1. Baseline (Stats)':<20} | {t_base:<10.4f} | {'0.00':<12} | {'0.00':<10} | 1.00x")
        logging.info(f"{'2. Adaptive Quad':<20} | {t_quad:<10.4f} | {np.max(err_quad):<12.2e} | {rmse_quad:<10.2e} | {t_base/t_quad:.1f}x")
        logging.info(f"{'3. Fast Fixed Quad':<20} | {t_fixed:<10.4f} | {np.max(err_fixed):<12.2e} | {rmse_fixed:<10.2e} | {t_base/t_fixed:.1f}x")
        logging.info("")
        
        # Plotting (x-axis is the distance vector magnitude)
        plot_dist = distances * 1e6
        if "+x" in title or "+y" in title or "-z" in title:
            plot_dist = plot_dist / 5  # Match the scaled distances for tighter gradients
            
        ax.plot(plot_dist, T_base, 'k-', linewidth=4, label="Baseline", alpha=0.5)
        ax.plot(plot_dist, T_quad, 'r--', linewidth=2, label="Adaptive Quad")
        ax.plot(plot_dist, T_fixed, 'b:', linewidth=2, label="Fast Fixed Quad")
        
        ax.set_title(title)
        ax.set_xlabel("Distance from Laser Center (µm)")
        ax.set_ylabel("Temperature (K)")
        ax.grid(True, alpha=0.3)
        if i == 0: ax.legend()

    # Hide the unused 6th subplot
    axes[5].axis('off')

    plt.tight_layout()
    plot_path = os.path.join(out_dir, 'directional_benchmark.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"\n>>> Directional Benchmark Complete!")
    print(f">>> Log saved to: {log_file}")
    print(f">>> Plot saved to: {plot_path}")

if __name__ == "__main__":
    run_directional_benchmark()