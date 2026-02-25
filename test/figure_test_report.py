import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, fixed_quad

# --- STANDALONE PHYSICS FUNCTIONS ---
def s_func(s, x, y, z, v, alpha, a):
    """ Original Integrand with 1/sqrt(s) singularity """
    s = np.atleast_1d(s)
    denom = 4 * alpha * s + a**2
    term_z = z**2 / (4 * alpha * s) if z != 0 else np.zeros_like(s)
    term_lat = (y**2 + (x + v * s)**2) / denom
    return np.exp(-term_z - term_lat) / (np.sqrt(s) * denom)

def s_func_substituted(u, x, y, z, v, alpha, a):
    """ Transformed Integrand: s = u^2 (Removes singularity) """
    u = np.atleast_1d(u)
    s = u**2
    denom = 4 * alpha * s + a**2
    term_z = z**2 / (4 * alpha * s) if z != 0 else np.zeros_like(s)
    term_lat = (y**2 + (x + v * s)**2) / denom
    return 2 * np.exp(-term_z - term_lat) / denom

def get_dynamic_window(x, y, z, v, alpha, a):
    """Calculates the statistical sliding window limits."""
    def f(u):
        val = s_func_substituted(u, x, y, z, v, alpha, a)
        return float(np.squeeze(val))

    upper_calc_limit = np.sqrt((abs(x)/v) * 3 + (x**2+y**2+z**2)/(4*alpha) + 100*(a/v))
    area, _ = quad(f, 0, upper_calc_limit, limit=100)
    
    def u_times_f(u): return u * f(u)
    mean_u, _ = quad(u_times_f, 0, upper_calc_limit, limit=100)
    mean_u /= area
    
    def var_integrand(u): return ((u - mean_u)**2) * f(u)
    variance_u, _ = quad(var_integrand, 0, upper_calc_limit, limit=100)
    std_u = np.sqrt(abs(variance_u / area))
    
    return max(0.0, mean_u - (4.0 * std_u)), mean_u + (4.0 * std_u)

def generate_updated_figure():
    print(">>> GENERATING UPDATED FIGURE 4...")
    
    # --- Academic Plot Formatting ---
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'legend.fontsize': 10,
        'figure.titlesize': 15
    })

    # Parameters
    P, v, a = 200.0, 1.0, 50e-6
    alpha = 8e-6
    t_dwell = a / v
    x, y, z = 0, 0, 0 
    
    # --- EXTREME INFERNO COLOR STYLING ---
    cmap = plt.get_cmap('inferno')
    c_extreme_dark = cmap(0.05)  # Very deep purple/black
    c_dark = cmap(0.2)
    c_bright = cmap(0.85)        # Glowing bright yellow/orange

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    plt.subplots_adjust(hspace=0.35, wspace=0.25)
    
    # ==========================================================
    # PANEL A: Original (Singularity)
    # ==========================================================
    s_limit = 2.0 * t_dwell
    s_vals = np.linspace(0, s_limit, 1000)
    s_vals_safe = np.maximum(s_vals, 1e-12) 
    y_orig = s_func(s_vals_safe, x, y, z, v, alpha, a)
    
    spike_cutoff = s_func(t_dwell * 0.01, x, y, z, v, alpha, a)
    
    # Using the extreme dark color
    axes[0, 0].plot(s_vals * 1e6, y_orig, color=c_extreme_dark, linewidth=2.5)
    axes[0, 0].axvline(t_dwell * 1e6, color=c_dark, linestyle='--', alpha=0.5, label='Dwell Time')
    axes[0, 0].set_title("(a) Original Integrand (Singularity)", fontweight='bold')
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].set_xlabel("Time ($s$, µs)")
    axes[0, 0].set_ylim(0, spike_cutoff)
    axes[0, 0].text(s_limit*1e6*0.05, spike_cutoff*0.9, "$\leftarrow$ Singularity at $s=0$", color=c_extreme_dark, fontweight='bold')
    axes[0, 0].grid(True, linestyle='--', alpha=0.4)
    axes[0, 0].legend()

    # ==========================================================
    # PANEL B: Substituted (Smooth)
    # ==========================================================
    u_vals = np.linspace(0, np.sqrt(s_limit), 1000)
    y_sub = s_func_substituted(u_vals, x, y, z, v, alpha, a)
    
    # Using the extreme bright color
    axes[0, 1].plot(u_vals, y_sub, color=c_bright, linewidth=2.5)
    axes[0, 1].axvline(np.sqrt(t_dwell), color=c_dark, linestyle='--', alpha=0.5, label='$\sqrt{Dwell}$')
    axes[0, 1].set_title("(b) Transformed Integrand ($s=u^2$)", fontweight='bold')
    axes[0, 1].set_ylabel("Amplitude")
    axes[0, 1].set_xlabel("Transformed Variable, $u$ ($s^{1/2}$)")
    axes[0, 1].set_ylim(0, np.max(y_sub)*1.1)
    axes[0, 1].set_xlim(left=0)
    axes[0, 1].grid(True, linestyle='--', alpha=0.4)
    axes[0, 1].legend()

    # ==========================================================
    # PANEL C: Error Convergence
    # ==========================================================
    print("   -> Calculating Convergence...")
    val_truth, _ = quad(s_func, 1e-12, 100 * t_dwell, args=(x, y, z, v, alpha, a), epsabs=1e-12)
    
    lim_s_fixed = 10 * t_dwell
    lim_u_fixed = np.sqrt(lim_s_fixed)
    n_steps_line = [5, 10, 15, 20, 30, 40, 50]
    err_orig_list, err_sub_list = [], []
    
    for n in n_steps_line:
        v1, _ = fixed_quad(s_func, 1e-10, lim_s_fixed, args=(x, y, z, v, alpha, a), n=n)
        err_orig_list.append(abs(v1 - val_truth)/val_truth * 100)
        
        v2, _ = fixed_quad(s_func_substituted, 0, lim_u_fixed, args=(x, y, z, v, alpha, a), n=n)
        err_sub_list.append(abs(v2 - val_truth)/val_truth * 100)

    # High contrast extremes
    axes[1, 0].plot(n_steps_line, err_orig_list, marker='o', color=c_extreme_dark, linewidth=2, label='Original ($s$)')
    axes[1, 0].plot(n_steps_line, err_sub_list, marker='s', color=c_bright, linewidth=2, label='Substituted ($u$)')
    axes[1, 0].axhline(0.1, color='gray', linestyle=':', label='0.1% Error Target')
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_title("(c) Quadrature Error Convergence", fontweight='bold')
    axes[1, 0].set_xlabel("Integration Nodes ($n$)")
    axes[1, 0].set_ylabel("Error % (Log Scale)")
    axes[1, 0].grid(True, which="both", linestyle='--', alpha=0.4)
    axes[1, 0].legend()

    # ==========================================================
    # PANEL D: Dynamic Sliding Window (Multiple Distances)
    # ==========================================================
    print("   -> Calculating Dynamic Boundaries...")
    
    # 1. Define multiple points behind the laser
    distances_micron = [50, 400, 1200]
    
    # Map colors across the extreme range of Inferno (0.1 to 0.9)
    colors_d = cmap(np.linspace(0.1, 0.9, len(distances_micron)))
    
    # 2. Determine universal X-axis for plotting based on the furthest point
    max_d = max(distances_micron) * 1e-6
    max_t_plot_w = (max_d / v) * 2.0
    u_vals_w = np.linspace(0, np.sqrt(max_t_plot_w), 800)
    
    axes[1, 1].set_title("(d) Dynamic Domain Truncation", fontweight='bold')
    
    for d_mic, color in zip(distances_micron, colors_d):
        x_w = -d_mic * 1e-6
        
        # Get boundaries
        l_low, l_up = get_dynamic_window(x_w, 0, 0, v, alpha, a)
        
        # Calculate curve
        integrand_vals_w = s_func_substituted(u_vals_w, x_w, 0, 0, v, alpha, a)
        norm_vals_w = integrand_vals_w / np.max(integrand_vals_w) # Normalize so peaks = 1
        
        # Plot curve
        axes[1, 1].plot(u_vals_w, norm_vals_w, color=color, linewidth=2.5, label=f"Wake (-{d_mic} µm)")
        
        # Highlight integration boundaries
        axes[1, 1].axvspan(l_low, l_up, color=color, alpha=0.15)
        axes[1, 1].axvline(x=l_low, color=color, linestyle='--', linewidth=1.5)
        axes[1, 1].axvline(x=l_up, color=color, linestyle='--', linewidth=1.5)

    axes[1, 1].set_xlabel("Transformed Variable, $u$ ($s^{1/2}$)")
    axes[1, 1].set_ylabel("Normalized Integrand")
    axes[1, 1].set_ylim(0, 1.15)
    axes[1, 1].set_xlim(left=0)
    axes[1, 1].grid(True, linestyle='--', alpha=0.4)
    axes[1, 1].legend(loc='upper right')

    # Save output
    out_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, 'figure_4_updated.png')
    
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f">>> Publication-quality Figure 4 saved to: {out_file}")

if __name__ == "__main__":
    generate_updated_figure()