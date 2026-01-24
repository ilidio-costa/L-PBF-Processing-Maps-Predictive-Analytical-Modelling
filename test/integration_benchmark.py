import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, fixed_quad
import sys
import os
import logging

# Add parent directory to path to import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 1. IMPORT THE ORIGINAL FUNCTION FROM PHYSICS.PY
from src.physics import s_func

# --- SETUP LOGGING ---
def setup_logger():
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'integration_benchmark.log')
    
    logger = logging.getLogger("Benchmark")
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()
        
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(logging.Formatter('%(message)s'))
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter('%(message)s'))
    
    logger.addHandler(fh)
    logger.addHandler(console)
    return logger, log_file

# --- OPTIMIZED FUNCTION (Local Definition) ---
def s_func_substituted(u, x, y, z, v, alpha, a):
    """
    Optimized version of 's_func' using s = u^2 substitution.
    Removes the singularity at s=0.
    """
    u = np.atleast_1d(u)
    s = u**2
    
    denom_lateral = 4 * alpha * s + a**2
    
    # Safety for z term when s -> 0
    term_z = np.zeros_like(s)
    safe_mask = s > 1e-20
    
    if z != 0:
        term_z[safe_mask] = z**2 / (4 * alpha * s[safe_mask])
        term_z[~safe_mask] = np.inf 
    
    term_lateral = (y**2 + (x + v * s)**2) / denom_lateral
    exp_val = np.exp(-term_z - term_lateral)
    
    # Jacobian (2u) cancels the 1/sqrt(s). Result is: 2 * exp / denom
    return 2 * exp_val / denom_lateral

def run_benchmark():
    logger, log_path = setup_logger()
    logger.info(">>> Running Integration Benchmark (Original vs. Optimized)...")
    
    # Parameters (NiTi approx)
    P, v, a = 200.0, 1.0, 50e-6
    alpha = 8e-6
    x, y, z = 0, 0, -20e-6 
    
    # LIMITS: 
    # 0.05s is "infinity" for a melt pool (events last ~0.0005s)
    upper_limit_s = 0.05 
    upper_limit_u = np.sqrt(upper_limit_s)
    
    # --- GROUND TRUTH (Using ORIGINAL s_func) ---
    logger.info(f"Computing Ground Truth using ORIGINAL s_func (limit={upper_limit_s}s)...")
    
    # We use quad with the original function (carefully avoiding 0)
    # This proves the Optimized version matches the Physics file.
    truth_val, _ = quad(s_func, 1e-12, upper_limit_s, args=(x, y, z, v, alpha, a), limit=200)
    logger.info(f"Ground Truth Value: {truth_val:.6e}")

    # --- BENCHMARK LOOP ---
    n_points_list = [5, 10, 20, 50, 100, 200, 500, 1000]
    iterations = 1000 # Loop for timing accuracy
    
    header = f"\n{'N':<5} | {'Method':<20} | {'Result':<12} | {'Error %':<12} | {'Time (µs)':<10}"
    logger.info(header)
    logger.info("-" * 75)

    # Store for plotting
    results = {'orig': {'err': [], 'time': []}, 'sub': {'err': [], 'time': []}}

    for n in n_points_list:
        # --- 1. ORIGINAL METHOD (s_func) ---
        # Fixed Quad on s (must avoid 0)
        t_start = time.perf_counter()
        for _ in range(iterations):
            val_orig, _ = fixed_quad(s_func, 1e-9, upper_limit_s, args=(x,y,z,v,alpha,a), n=n)
        t_avg = ((time.perf_counter() - t_start) / iterations) * 1e6
        
        err_orig = abs(val_orig - truth_val) / truth_val * 100 if truth_val else 0
        results['orig']['err'].append(err_orig)
        results['orig']['time'].append(t_avg)
        
        logger.info(f"{n:<5} | {'Original (s)':<20} | {val_orig:.4e}   | {err_orig:9.4f}%   | {t_avg:.2f}")

        # --- 2. SUBSTITUTED METHOD (u^2) ---
        # Fixed Quad on u (0 to sqrt(limit))
        t_start = time.perf_counter()
        for _ in range(iterations):
            val_sub, _ = fixed_quad(s_func_substituted, 0, upper_limit_u, args=(x,y,z,v,alpha,a), n=n)
        t_avg = ((time.perf_counter() - t_start) / iterations) * 1e6
        
        err_sub = abs(val_sub - truth_val) / truth_val * 100 if truth_val else 0
        results['sub']['err'].append(err_sub)
        results['sub']['time'].append(t_avg)

        logger.info(f"{n:<5} | {'Optimized (u^2)':<20} | {val_sub:.4e}   | {err_sub:9.4f}%   | {t_avg:.2f}")
        logger.info("-" * 75)

    # --- PLOTTING ---
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Accuracy (Error)
        ax1.plot(n_points_list, results['orig']['err'], 'r--o', label='Original (s)')
        ax1.plot(n_points_list, results['sub']['err'], 'b-s', linewidth=2, label='Optimized (u^2)')
        ax1.set_title('Accuracy Convergence')
        ax1.set_xlabel('Number of Points (N)')
        ax1.set_ylabel('Relative Error (%)')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Speed (Time)
        ax2.plot(n_points_list, results['orig']['time'], 'r--o', label='Original (s)')
        ax2.plot(n_points_list, results['sub']['time'], 'b-s', label='Optimized (u^2)')
        ax2.set_title('Computational Cost')
        ax2.set_xlabel('Number of Points (N)')
        ax2.set_ylabel('Time per call (µs)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.suptitle(f"Benchmark: Original s_func vs Optimized u^2\n(Ground Truth: {truth_val:.4e})")
        
        out_file = os.path.join(os.path.dirname(__file__), 'output', 'integration_benchmark.png')
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        plt.savefig(out_file)
        logger.info(f"\nBenchmark Plot saved to {out_file}")
        
    except Exception as e:
        logger.error(f"Plotting failed: {e}")

if __name__ == "__main__":
    run_benchmark()