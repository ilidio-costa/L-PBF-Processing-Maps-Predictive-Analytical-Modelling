import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.physics import s_func_substituted

def calculate_dynamic_u_limit(x, y, z, v, alpha, a):
    """
    Direction-Matrix (Vector Projection) based dynamic boundary for u = sqrt(s).
    """
    # 1. Define vectors
    vec_r = np.array([x, y, z])
    vec_v_dir = np.array([1.0, 0.0, 0.0]) # Laser travels purely in +x direction
    
    distance = np.linalg.norm(vec_r)
    t_dwell = a / v
    
    # 2. Vector Projection (Directional Matrix Logic)
    # np.dot(vec_r, vec_v_dir) extracts the x-component.
    # If point is in front (+x), distance - dot_product approaches 0.
    # If point is behind (-x), distance - dot_product approaches 2 * distance.
    directional_lag = (distance - np.dot(vec_r, vec_v_dir)) / v
    
    # 3. Calculate Limit
    # Base dwell time guarantees we always capture the core laser spot itself (for small distances)
    t_base = 15.0 * t_dwell 
    
    # We multiply the directional lag by a scaling factor (e.g., 2.5) to ensure we 
    # integrate far enough down the "tail" of the curve, not just stop at the peak.
    t_limit = t_base + (2.5 * directional_lag)
    
    return np.sqrt(t_limit)

def run_matrix_boundary_study():
    # --- 1. Set up Base Parameters ---
    v = 0.8          # Scan speed [m/s]
    a = 40e-6        # Laser radius [m]
    alpha = 5e-6     # Thermal diffusivity [m^2/s]
    
    # Distances to test in microns
    distances_micron = np.array([40, 200, 500, 1000, 2000])
    distances = distances_micron * 1e-6
    
    cmap = cm.get_cmap('inferno')
    colors = cmap(np.linspace(0.1, 0.8, len(distances)))

    # Define the 6 directions to test in 3D space
    # NOTE: Z is strictly negative (down into the powder bed)
    inv_sqrt2 = 1.0 / np.sqrt(2)
    directions = {
        "1. Wake (-x)":                 lambda d: (-d, 0, 0),
        "2. Front (+x)":                lambda d: (d, 0, 0),
        "3. Side (+y)":                 lambda d: (0, d, 0),
        "4. Depth (-z)":                lambda d: (0, 0, -d),
        "5. Diagonal Wake (-x, -z)":    lambda d: (-d * inv_sqrt2, 0, -d * inv_sqrt2),
        "6. Diagonal Front (+x, -z)":   lambda d: (d * inv_sqrt2, 0, -d * inv_sqrt2)
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    fig.suptitle(f"Vector-Directional Boundary Study (Z is negative)\n(v={v} m/s, a={a*1e6:.0f} µm, max_d=2000 µm)", fontsize=16)

    for i, (title, coord_func) in enumerate(directions.items()):
        ax = axes[i]
        ax.set_title(title)
        ax.set_xlabel("Integration Variable (u = √s)")
        ax.set_ylabel("Normalized Integrand")
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Calculate maximum needed x-axis for plotting
        max_dist = np.max(distances)
        if "-x" in title:
            max_t_plot = (max_dist / v) * 2.5
        else:
            max_t_plot = (max_dist / v) * 1.5 + (20 * (a/v))
            
        u_vals = np.linspace(0, np.sqrt(max_t_plot), 1000)

        for j, (d, color) in enumerate(zip(distances, colors)):
            x, y, z = coord_func(d)
            
            # Evaluate kernel
            integrand_vals = s_func_substituted(u_vals, x, y, z, v, alpha, a)
            
            # Normalize for visualization
            max_val = np.max(integrand_vals)
            if max_val > 0:
                normalized_vals = integrand_vals / max_val
            else:
                normalized_vals = integrand_vals

            # Plot curve
            label = f"d = {d*1e6:.0f} µm"
            ax.plot(u_vals, normalized_vals, color=color, linewidth=2, label=label)
            
            # Calculate and plot the new vector-based dynamic limit
            dynamic_u_limit = calculate_dynamic_u_limit(x, y, z, v, alpha, a)
            ax.axvline(x=dynamic_u_limit, color=color, linestyle='--', alpha=0.8)
            
            # Dot exactly where the cut-off happens
            idx_limit = (np.abs(u_vals - dynamic_u_limit)).argmin()
            if idx_limit < len(u_vals):
                ax.plot(dynamic_u_limit, normalized_vals[idx_limit], marker='o', color=color)

        if i == 0:
            ax.legend(loc='upper right')

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save Output
    out_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, 'test_boundaries_matrix_directional.png')
    
    plt.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f">>> Vector-Directional Boundary study saved to: {out_file}")

if __name__ == "__main__":
    run_matrix_boundary_study()