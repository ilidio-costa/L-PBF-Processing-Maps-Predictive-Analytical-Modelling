import sys
import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import logging

# --- Setup Paths ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

# Import the updated models from your physics engine
from src.physics import (
    eagar_tsai_temp, get_eagar_tsai_dimensions,
    rubenchik_variables, rubenchik_temp, get_rubenchik_dimensions
)

# --- Setup Logging & Output ---
log_dir = os.path.join(BASE_DIR, 'test', 'logs')
out_dir = os.path.join(BASE_DIR, 'test', 'output')
os.makedirs(log_dir, exist_ok=True)
os.makedirs(out_dir, exist_ok=True)

log_file = os.path.join(log_dir, 'analytical_comparison.log')
with open(log_file, 'w'): pass  # Clear old log

logging.basicConfig(filename=log_file, level=logging.INFO, format='%(message)s')

def load_ti64():
    """Loads Ti64 from the materials folder and ensures alpha is calculated."""
    mat_path = os.path.join(BASE_DIR, 'materials', 'Ti64.json')
    if not os.path.exists(mat_path):
        print(f"Error: Could not find material file at {mat_path}")
        sys.exit(1)
        
    with open(mat_path, 'r') as f:
        mat = json.load(f)
        if 'alpha' not in mat:
            mat['alpha'] = mat['k'] / (mat['rho'] * mat['C_p'])
        return mat

def run_comparison():
    mat = load_ti64()
    P, v, a = 250.0, 0.8, 40e-6
    T_ambient = 293.15

    logging.info("===================================================================")
    logging.info(f"   EAGAR-TSAI vs RUBENCHIK COMPARISON | {mat['name']} | P={P}W, v={v}m/s")
    logging.info("===================================================================\n")

    # -----------------------------------------------------------------------
    # TEST 1: Spatial Temperature Profile
    # -----------------------------------------------------------------------
    logging.info("--- TEST 1: SPATIAL TEMPERATURE PROFILE (Wake to Front) ---")
    x_vals = np.linspace(-1500e-6, 200e-6, 150)
    
    # Run Eagar-Tsai
    t0 = time.time()
    T_et = np.array([eagar_tsai_temp(x, 0, 0, P, v, a, mat, T_ambient) for x in x_vals])
    time_et_prof = time.time() - t0

    # Run Rubenchik
    t0 = time.time()
    T_rub = []
    for x in x_vals:
        coords, B, p_val = rubenchik_variables(x, 0, 0, mat, P, v, a)
        T_rub.append(rubenchik_temp(coords, B, p_val, mat, T_ambient))
    T_rub = np.array(T_rub)
    time_rub_prof = time.time() - t0

    err_prof = np.abs(T_et - T_rub)
    max_err_prof = np.max(err_prof)

    logging.info(f"{'Model':<15} | {'Time (s)':<10} | {'Max Difference (K)':<20}")
    logging.info("-" * 50)
    logging.info(f"{'Eagar-Tsai':<15} | {time_et_prof:<10.4f} | {'-':<20}")
    logging.info(f"{'Rubenchik':<15} | {time_rub_prof:<10.4f} | {max_err_prof:<20.4e}\n")

    # -----------------------------------------------------------------------
    # TEST 2: Melt Pool Dimensions Solvers
    # -----------------------------------------------------------------------
    logging.info("--- TEST 2: MELT POOL DIMENSION EXTRACTION ---")
    
    t0 = time.time()
    L_et, W_et, D_et, tail_et, front_et = get_eagar_tsai_dimensions(P, v, a, mat, resolution=100)
    time_et_dim = time.time() - t0

    t0 = time.time()
    L_rub, W_rub, D_rub, tail_rub, front_rub = get_rubenchik_dimensions(P, v, a, mat, T_ambient, resolution=100)
    time_rub_dim = time.time() - t0

    logging.info(f"{'Model':<12} | {'Time (s)':<10} | {'Length (µm)':<12} | {'Width (µm)':<12} | {'Depth (µm)':<12}")
    logging.info("-" * 65)
    logging.info(f"{'Eagar-Tsai':<12} | {time_et_dim:<10.4f} | {L_et*1e6:<12.2f} | {W_et*1e6:<12.2f} | {D_et*1e6:<12.2f}")
    logging.info(f"{'Rubenchik':<12} | {time_rub_dim:<10.4f} | {L_rub*1e6:<12.2f} | {W_rub*1e6:<12.2f} | {D_rub*1e6:<12.2f}")
    
    err_L = abs(L_et - L_rub) * 1e6
    err_W = abs(W_et - W_rub) * 1e6
    err_D = abs(D_et - D_rub) * 1e6
    logging.info(f"{'Difference':<12} | {'-':<10} | {err_L:<12.2e} | {err_W:<12.2e} | {err_D:<12.2e}\n")

    # -----------------------------------------------------------------------
    # PLOTTING
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_vals * 1e6, T_et, 'k-', linewidth=4, label="Eagar-Tsai (Physical)", alpha=0.6)
    ax.plot(x_vals * 1e6, T_rub, 'r--', linewidth=2, label="Rubenchik (Dimensionless)")
    
    # Highlight the boundaries
    ax.axhline(mat['T_m'], color='b', linestyle=':', label='Melting Temp ($T_m$)')
    ax.axvspan(tail_et * 1e6, front_et * 1e6, color='grey', alpha=0.2, label='Melt Pool Length')

    ax.set_title(f"Model Comparison: Eagar-Tsai vs Rubenchik\n{mat['name']}, P={P}W, v={v}m/s")
    ax.set_xlabel("Distance from Laser Center, X (µm)")
    ax.set_ylabel("Surface Temperature (K)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plot_path = os.path.join(out_dir, 'analytical_comparison.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"\n>>> Comparison Complete!")
    print(f">>> Log saved to: {log_file}")
    print(f">>> Plot saved to: {plot_path}")

if __name__ == "__main__":
    run_comparison()