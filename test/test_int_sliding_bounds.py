import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import fixed_quad
import logging

# --- Setup Logging ---
log_dir = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'boundary_diagnostics_fixed.log')

# Clear old log and setup new one
with open(log_file, 'w'): pass 
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(message)s'
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.physics import s_func_substituted

def calculate_statistical_sliding_window_fixed(x, y, z, v, alpha, a, n_points=60):
    """
    Calculates boundaries using fixed_quad with dynamic, tightly wrapped integration bounds.
    """
    def f_vec(u):
        return s_func_substituted(u, x, y, z, v, alpha, a)
    
    # 1. Locate the Theoretical Peak
    t_peak = max(0.0, -x/v) + a/v
    peak_guess = np.sqrt(t_peak)
    max_f = float(np.squeeze(s_func_substituted(peak_guess, x, y, z, v, alpha, a)))
    
    # If the curve is practically zero everywhere, underflow
    if max_f < 1e-20:
        return 0.0, np.sqrt(a/v), 0.0, 0.0, 0.0, max_f, "UNDERFLOW: Curve is too small"

    # 2. Calculate Smart Integration Bounds
    # The physical spread of the heat kernel is roughly proportional to sqrt(4 * alpha * t)
    # We convert this spatial spread to a time spread, and take a generous multiple (8x)
    time_spread = np.sqrt(4.0 * alpha * t_peak) / v
    time_spread = max(time_spread, a/v) # Fallback to avoid extremely narrow bounds for small t
    
    t_lower = max(0.0, t_peak - 8.0 * time_spread)
    t_upper = t_peak + 8.0 * time_spread
    
    u_lower = np.sqrt(t_lower)
    u_upper = np.sqrt(t_upper)

    # 3. Integrate Area
    area, _ = fixed_quad(f_vec, u_lower, u_upper, n=n_points)
    
    if area < 1e-20:
        return 0.0, np.sqrt(a/v), area, 0.0, 0.0, max_f, "FAIL: Area essentially zero"
        
    # 4. Mean (mu)
    def u_times_f(u): return u * f_vec(u)
    mean_u, _ = fixed_quad(u_times_f, u_lower, u_upper, n=n_points)
    mean_u /= area
    
    # 5. Standard Deviation (sigma)
    def var_integrand(u): return ((u - mean_u)**2) * f_vec(u)
    variance_u, _ = fixed_quad(var_integrand, u_lower, u_upper, n=n_points)
    std_u = np.sqrt(abs(variance_u / area))
    
    # 6. Limits
    limit_upper = mean_u + (4.0 * std_u)
    limit_lower = max(0.0, mean_u - (4.0 * std_u))
        
    return limit_lower, limit_upper, area, mean_u, std_u, max_f, "SUCCESS"

def run_logged_study_fixed():
    v = 0.8          
    a = 40e-6        
    alpha = 5e-6     
    n_points = 60 # Number of quadrature points. Adjust between 50-150 based on accuracy needs.
    
    distances_micron = np.array([40, 200, 500, 1000, 2000])
    distances = distances_micron * 1e-6
    
    cmap = plt.get_cmap('inferno')
    colors = cmap(np.linspace(0.1, 0.8, len(distances)))

    inv_sqrt3 = 1.0 / np.sqrt(3)
    directions = {
        "1. Wake (-x)":                 lambda d: (-d, 0, 0),
        "2. Front (+x)":                lambda d: (d, 0, 0),
        "3. Side (+y)":                 lambda d: (0, d, 0),
        "4. Depth (-z)":                lambda d: (0, 0, -d),
        "5. Diagonal Front (+x,+y,-z)": lambda d: (d*inv_sqrt3, d*inv_sqrt3, -d*inv_sqrt3),
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    fig.suptitle(f"Sliding Window Boundary Study (Optimized with fixed_quad, n={n_points})", fontsize=16)

    logging.info("==================================================================================")
    logging.info(f"BOUNDARY DIAGNOSTICS LOG | v={v} m/s | a={a*1e6} µm | alpha={alpha} | n_quad={n_points}")
    logging.info("==================================================================================\n")

    start_time = time.time()

    for i, (title, coord_func) in enumerate(directions.items()):
        ax = axes[i]
        ax.set_title(title)
        ax.set_xlabel("u = √s")
        ax.set_ylabel("Integrand")
        ax.grid(True, linestyle='--', alpha=0.5)
        
        logging.info(f"--- DIRECTION: {title} ---")
        logging.info(f"{'Dist (µm)':<10} | {'x (µm)':<8} | {'y (µm)':<8} | {'z (µm)':<8} | {'Max f(u)':<12} | {'Area':<12} | {'Mean (u)':<10} | {'Std (u)':<10} | {'L_Lower':<10} | {'L_Upper':<10} | {'Status'}")
        
        max_dist = np.max(distances)
        max_t_plot = (max_dist / v) * 2.5 if "-x" in title else (max_dist / v) * 0.5 + 20*(a/v)
        u_vals = np.linspace(0, np.sqrt(max_t_plot), 500)

        for j, (d, color) in enumerate(zip(distances, colors)):
            x, y, z = coord_func(d)
            
            # Use the new fixed_quad function
            l_low, l_up, area, mu, std, max_f, status = calculate_statistical_sliding_window_fixed(x, y, z, v, alpha, a, n_points)
            
            logging.info(f"{d*1e6:<10.1f} | {x*1e6:<8.1f} | {y*1e6:<8.1f} | {z*1e6:<8.1f} | {max_f:<12.2e} | {area:<12.2e} | {mu:<10.5f} | {std:<10.5f} | {l_low:<10.5f} | {l_up:<10.5f} | {status}")
            
            integrand_vals = s_func_substituted(u_vals, x, y, z, v, alpha, a)
            if max_f > 1e-20:
                norm_vals = integrand_vals / np.max(integrand_vals)
                ax.plot(u_vals, norm_vals, color=color, linewidth=2, label=f"d={d*1e6:.0f}µm")
                ax.axvspan(l_low, l_up, color=color, alpha=0.1)
                ax.axvline(x=l_low, color=color, linestyle=':', alpha=0.8)
                ax.axvline(x=l_up, color=color, linestyle='--', alpha=0.8)

        if i == 0:
            ax.legend(loc='upper right')
        logging.info("") 

    end_time = time.time()
    elapsed = end_time - start_time

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    out_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, 'test_sliding_window_fixed_logged.png')
    plt.savefig(out_file, dpi=300)
    plt.close(fig)
    
    print(f">>> Integration completed in {elapsed:.4f} seconds.")
    print(f">>> Log saved to: {log_file}")
    print(f">>> Plot saved to: {out_file}")

if __name__ == "__main__":
    run_logged_study_fixed()