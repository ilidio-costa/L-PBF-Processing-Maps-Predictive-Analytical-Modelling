import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.integrate import quad
import logging

# --- Setup Logging ---
log_dir = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'boundary_diagnostics.log')

# Clear old log and setup new one
with open(log_file, 'w'): pass 
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(message)s'
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.physics import s_func_substituted

def calculate_statistical_sliding_window(x, y, z, v, alpha, a):
    """
    Calculates boundaries and returns all diagnostic info.
    """
    def f(u):
        # s_func_substituted returns an array due to np.atleast_1d. 
        # We must extract the first element to get a pure float for quad() and logging.
        val = s_func_substituted(u, x, y, z, v, alpha, a)
        return float(np.squeeze(val))
    
    # 1. Base Normalization Area
    upper_calc_limit = np.sqrt((abs(x)/v) * 3 + (x**2+y**2+z**2)/(4*alpha) + 100*(a/v))
    
    # Find theoretical max to see if we are dealing with underflow
    peak_guess = np.sqrt(max(0, -x/v) + a/v)
    max_f = f(peak_guess)
    
    # If the curve is practically zero everywhere, don't even try to integrate
    if max_f < 1e-20:
        return 0.0, np.sqrt(a/v), 0.0, 0.0, 0.0, max_f, "UNDERFLOW: Curve is too small"

    area, _ = quad(f, 0, upper_calc_limit, limit=100)
    
    if area < 1e-20:
        return 0.0, np.sqrt(a/v), area, 0.0, 0.0, max_f, "FAIL: Area essentially zero"
        
    # 2. Mean (mu)
    def u_times_f(u): return u * f(u)
    mean_u, _ = quad(u_times_f, 0, upper_calc_limit, limit=100)
    mean_u /= area
    
    # 3. Standard Deviation (sigma)
    def var_integrand(u): return ((u - mean_u)**2) * f(u)
    variance_u, _ = quad(var_integrand, 0, upper_calc_limit, limit=100)
    std_u = np.sqrt(abs(variance_u / area))
    
    # 4. Limits
    limit_upper = mean_u + (4.0 * std_u)
    limit_lower = max(0.0, mean_u - (4.0 * std_u))
        
    return limit_lower, limit_upper, area, mean_u, std_u, max_f, "SUCCESS"

def run_logged_study():
    v = 0.8          
    a = 40e-6        
    alpha = 5e-6     
    
    distances_micron = np.array([40, 200, 500, 1000, 2000])
    distances = distances_micron * 1e-6
    
    # FIX 1: Use plt.get_cmap instead of cm.get_cmap to avoid Matplotlib deprecation warning
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
    fig.suptitle(f"Sliding Window Boundary Study", fontsize=16)

    logging.info("==================================================================================")
    logging.info(f"BOUNDARY DIAGNOSTICS LOG | v={v} m/s | a={a*1e6} µm | alpha={alpha}")
    logging.info("==================================================================================\n")

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
            
            # Get boundaries and diagnostic info
            l_low, l_up, area, mu, std, max_f, status = calculate_statistical_sliding_window(x, y, z, v, alpha, a)
            
            # Log it! (FIX 2: max_f is now guaranteed to be a pure float)
            logging.info(f"{d*1e6:<10.1f} | {x*1e6:<8.1f} | {y*1e6:<8.1f} | {z*1e6:<8.1f} | {max_f:<12.2e} | {area:<12.2e} | {mu:<10.5f} | {std:<10.5f} | {l_low:<10.5f} | {l_up:<10.5f} | {status}")
            
            # Plotting (Normalize for visualization if curve actually exists)
            integrand_vals = s_func_substituted(u_vals, x, y, z, v, alpha, a)
            if max_f > 1e-20:
                norm_vals = integrand_vals / np.max(integrand_vals)
                ax.plot(u_vals, norm_vals, color=color, linewidth=2, label=f"d={d*1e6:.0f}µm")
                ax.axvspan(l_low, l_up, color=color, alpha=0.1)
                ax.axvline(x=l_low, color=color, linestyle=':', alpha=0.8)
                ax.axvline(x=l_up, color=color, linestyle='--', alpha=0.8)

        if i == 0:
            ax.legend(loc='upper right')
        logging.info("") # Empty line between directions

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    out_dir = os.path.join(os.path.dirname(__file__), 'output')
    out_file = os.path.join(out_dir, 'test_sliding_window_logged.png')
    plt.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f">>> Log saved to: {log_file}")
    print(f">>> Plot saved to: {out_file}")

if __name__ == "__main__":
    run_logged_study()