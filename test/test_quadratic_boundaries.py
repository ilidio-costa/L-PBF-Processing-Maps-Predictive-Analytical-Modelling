import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# --- 1. CORE PHYSICS FUNCTION ---
def s_func_substituted(u, x, y, z, v, alpha, a):
    """ Transformed Integrand: s = u^2 """
    u = np.atleast_1d(u)
    s = u**2
    denom = 4 * alpha * s + a**2
    term_z = z**2 / (4 * alpha * s) if z != 0 else np.zeros_like(s)
    term_lat = (y**2 + (x + v * s)**2) / denom
    return 2 * np.exp(-term_z - term_lat) / denom

# --- 2. NEW ALGEBRAIC BOUNDARY LOGIC (WITH NEAR-FIELD FIX) ---
def get_quadratic_window(x, y, z, v, alpha, a, N=8.0):
    """
    Calculates exact integration boundaries algebraically.
    """
    R_sq = x**2 + y**2 + z**2
    
    A = v**2
    B = 2 * (x * v - 2 * alpha * N)
    C = R_sq
    
    discriminant = B**2 - 4 * A * C
    
    # --- DISABLED FAST PASS ---
    # We no longer skip if discriminant < 0. We force a dummy boundary 
    # just to capture whatever microscopic numerical noise is there.
    if discriminant < 0 or (-B + np.sqrt(max(0, discriminant))) <= 0:
        dummy_max = (np.sqrt(R_sq) / v) * 2.0 + 10*(a/v)
        return 0.0, np.sqrt(dummy_max), discriminant, "FORCED"
        
    s_min = (-B - np.sqrt(discriminant)) / (2 * A)
    s_max = (-B + np.sqrt(discriminant)) / (2 * A)
    
    s_min = max(0.0, s_min)
    
    # --- THE NEAR-FIELD FIX ---
    # Because the quadratic approximation ignores the laser radius 'a', 
    # it breaks down at u=0. We check the true integrand at u=0. 
    # If the heat is already active, the limit MUST start at 0.
    val_at_zero = s_func_substituted(0.0, x, y, z, v, alpha, a)[0]
    if val_at_zero > 1e-20:  
        s_min = 0.0

    return np.sqrt(s_min), np.sqrt(s_max), discriminant, "INTEGRATED"

# --- 3. VISUALIZATION SCRIPT ---
def run_quadratic_test():
    v = 0.8          
    a = 40e-6        
    alpha = 5e-6     
    N_decay = 8.0  
    
    # Exact distances requested
    test_cases = {
        "1. Wake (-x)":  [(-d, 0, 0) for d in [50e-6, 400e-6, 1200e-6]],
        "2. Front (+x)": [(d, 0, 0) for d in [40e-6, 100e-6]],
        "3. Side (+y)":  [(0, d, 0) for d in [40e-6, 100e-6]],
        "4. Depth (-z)": [(0, 0, -d) for d in [40e-6, 100e-6]]
    }

    plt.rcParams.update({'font.family': 'serif', 'font.size': 11})
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    cmap = plt.get_cmap('inferno')

    for i, (title, points) in enumerate(test_cases.items()):
        ax = axes[i]
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel("Integration Variable, $u$ ($s^{1/2}$)")
        ax.set_ylabel("Normalized Integrand")
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_ylim(-0.05, 1.15)
        
        # EXTREME INFERNO colors
        colors = cmap(np.linspace(0.1, 0.9, len(points)))

        results = []
        max_u_plot = 0.0
        
        # Pass 1: Get bounds
        for (x, y, z) in points:
            u_min, u_max, delta, status = get_quadratic_window(x, y, z, v, alpha, a, N=N_decay)
            results.append((x, y, z, u_min, u_max, delta, status))
            max_u_plot = max(max_u_plot, u_max * 1.15)
        
        ax.set_xlim(0, max_u_plot)
        u_vals = np.linspace(0, max_u_plot, 1000)

        # Pass 2: Plot curves without skipping
        for j, ((x, y, z, u_min, u_max, delta, status), color) in enumerate(zip(results, colors)):
            dist_um = np.sqrt(x**2 + y**2 + z**2) * 1e6
            
            integrand = s_func_substituted(u_vals, x, y, z, v, alpha, a)
            peak_val = np.max(integrand)
            
            # Prevent Division by Zero for essentially dead curves
            if peak_val > 1e-100:
                norm_vals = integrand / peak_val
            else:
                norm_vals = np.zeros_like(u_vals)
                
            ax.plot(u_vals, norm_vals, color=color, linewidth=2.5, label=f"d={dist_um:.0f} µm")
            
            # Shade the regions calculated by the formula
            if status == "INTEGRATED":
                ax.axvspan(u_min, u_max, color=color, alpha=0.15)
                ax.axvline(x=u_min, color=color, linestyle='--', linewidth=1.5)
                ax.axvline(x=u_max, color=color, linestyle='--', linewidth=1.5)
            elif status == "FORCED":
                # For forced points, plot a faint red box to show the dummy search area
                ax.axvspan(u_min, u_max, color='red', alpha=0.05)
                if j == len(points) - 1:
                    ax.text(0.5, 0.5, "HEAT IS NEGLIGIBLE\n(Shown inside red dummy window)", 
                            horizontalalignment='center', verticalalignment='center', 
                            transform=ax.transAxes, fontsize=12, color=cmap(0.5), fontweight='bold')

        ax.legend(loc='upper right')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, wspace=0.2)
    
    out_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, 'test_forced_boundaries.png')
    plt.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f">>> Test plot saved to: {out_file}\n")

if __name__ == "__main__":
    run_quadratic_test()