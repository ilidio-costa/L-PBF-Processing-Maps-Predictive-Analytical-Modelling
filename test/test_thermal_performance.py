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

log_file = os.path.join(log_dir, 'integration_benchmark.log')
with open(log_file, 'w'): pass  # Clear old log

logging.basicConfig(
    filename=log_file, level=logging.INFO,
    format='%(message)s'
)

# ==============================================================================
# 1. The Three Competitors
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

    # Using 250 nodes to completely crush the error gap
    integral_val, _ = fixed_quad(s_func_substituted, 0.0, upper_calc_limit, args=(x, y, z, v, alpha, a), n=250)
    return (pre_factor * integral_val) + T_ambient

# ==============================================================================
# 2. Benchmark Runner
# ==============================================================================

def load_materials():
    """Loads all JSON materials and calculates thermal diffusivity (alpha) if missing."""
    materials = {}
    mat_dir = os.path.join(BASE_DIR, 'materials')
    
    # Check if directory exists before trying to list it
    if not os.path.exists(mat_dir):
        print(f"Warning: Materials directory not found at {mat_dir}")
        return materials

    for file in os.listdir(mat_dir):
        if file.endswith('.json'):
            with open(os.path.join(mat_dir, file), 'r') as f:
                name = file.replace('.json', '')
                mat_props = json.load(f)
                
                # --- The Alpha Fix ---
                # If 'alpha' is not in the JSON, calculate it using k / (rho * Cp)
                if 'alpha' not in mat_props:
                    mat_props['alpha'] = mat_props['k'] / (mat_props['rho'] * mat_props['C_p'])
                    
                materials[name] = mat_props
    return materials

def run_comprehensive_benchmark():
    materials = load_materials()
    if not materials:
        print("Error: No materials loaded. Cannot run benchmark.")
        return

    # Sweep Parameters
    test_matrix = [
        {"P": 100, "v": 0.4, "desc": "Low P, Slow v"},
        {"P": 250, "v": 0.8, "desc": "Nominal LPBF"},
        {"P": 400, "v": 1.5, "desc": "High P, Fast v"}
    ]
    a = 40e-6
    x_vals = np.linspace(-2000e-6, 200e-6, 200)  # 200 spatial points

    logging.info("===================================================================")
    logging.info("          COMPREHENSIVE INTEGRATION ALGORITHM BENCHMARK            ")
    logging.info("===================================================================\n")

    # Plot Setup
    fig, axes = plt.subplots(len(materials), len(test_matrix), figsize=(18, 12))
    fig.suptitle("Eagar-Tsai Temperature Integration Methods", fontsize=16)

    # Ensure axes is always a 2D array for consistent indexing
    if len(materials) == 1:
        axes = np.expand_dims(axes, axis=0)
    if len(test_matrix) == 1:
        axes = np.expand_dims(axes, axis=1)

    for i, (mat_name, mat_props) in enumerate(materials.items()):
        logging.info(f"--- TESTING MATERIAL: {mat_name} ---")
        logging.info(f"{'Condition':<18} | {'Method':<20} | {'Time (s)':<10} | {'Max Err (K)':<12} | {'RMSE (K)':<10} | {'Speedup'}")
        logging.info("-" * 85)

        for j, params in enumerate(test_matrix):
            P, v = params['P'], params['v']
            desc = params['desc']
            
            # 1. Baseline
            t0 = time.time()
            T_base = np.array([temp_redundant_stats(x, 0, 0, P, v, a, mat_props) for x in x_vals])
            t_base = time.time() - t0
            
            # 2. Adaptive Quad
            t0 = time.time()
            T_quad = np.array([temp_adaptive_quad(x, 0, 0, P, v, a, mat_props) for x in x_vals])
            t_quad = time.time() - t0
            
            # 3. Fast Fixed Quad
            t0 = time.time()
            T_fixed = np.array([temp_fast_fixed_quad(x, 0, 0, P, v, a, mat_props) for x in x_vals])
            t_fixed = time.time() - t0

            # Error Metrics
            err_quad = np.abs(T_base - T_quad)
            err_fixed = np.abs(T_base - T_fixed)

            rmse_quad = np.sqrt(np.mean(err_quad**2))
            rmse_fixed = np.sqrt(np.mean(err_fixed**2))

            # Logging
            logging.info(f"{desc:<18} | {'1. Baseline (Stats)':<20} | {t_base:<10.4f} | {'0.00':<12} | {'0.00':<10} | 1.00x")
            logging.info(f"{'':<18} | {'2. Adaptive Quad':<20} | {t_quad:<10.4f} | {np.max(err_quad):<12.2e} | {rmse_quad:<10.2e} | {t_base/t_quad:.1f}x")
            logging.info(f"{'':<18} | {'3. Fast Fixed Quad':<20} | {t_fixed:<10.4f} | {np.max(err_fixed):<12.2e} | {rmse_fixed:<10.2e} | {t_base/t_fixed:.1f}x")
            
            # Plotting
            ax = axes[i, j]
            ax.plot(x_vals*1e6, T_base, 'k-', linewidth=3, label="Baseline")
            ax.plot(x_vals*1e6, T_quad, 'r--', linewidth=1.5, label="Adaptive Quad")
            ax.plot(x_vals*1e6, T_fixed, 'b:', linewidth=2, label="Fast Fixed Quad")
            
            ax.set_title(f"{mat_name} - {desc}\nP={P}W, v={v}m/s")
            ax.set_xlabel("x (µm)")
            ax.set_ylabel("Temperature (K)")
            ax.grid(True, alpha=0.3)
            if i == 0 and j == 0: ax.legend()

        logging.info("") # Spacing

    plt.tight_layout()
    plot_path = os.path.join(out_dir, 'integration_benchmark.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"\n>>> Benchmark Complete!")
    print(f">>> Log saved to: {log_file}")
    print(f">>> Plot saved to: {plot_path}")

if __name__ == "__main__":
    run_comprehensive_benchmark()