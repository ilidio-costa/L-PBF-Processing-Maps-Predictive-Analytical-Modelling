import sys
import os
import logging
import numpy as np
import matplotlib.pyplot as plt

# --- Set up Project Paths ---
test_dir = os.path.dirname(__file__)
logs_dir = os.path.join(test_dir, 'logs')
output_dir = os.path.join(test_dir, 'output')

os.makedirs(logs_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# --- 1. Setup Logging ---
log_file = os.path.join(logs_dir, 'model_comparison_bounds.log')
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
    
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(test_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src import physics

def standalone_rubenchik_interpolation(P, v, a, material, T_ambient=0):
    """
    Standalone version of Equation 11 with internal logging and the ABS fix for depth.
    """
    _, B, p = physics.rubenchik_variables(0, 0, 0, material, P, v, a, T_ambient)
    
    # Check if we are violating the paper's valid interpolation bounds
    if not (1 <= B <= 20):
        logging.warning(f"  -> [BOUNDS WARNING] B ({B:.2f}) is outside the paper's valid range (1 to 20).")
    if not (0.1 <= p <= 5):
        logging.warning(f"  -> [BOUNDS WARNING] p ({p:.2f}) is outside the paper's valid range (0.1 to 5).")

    # Eq 11 Math
    depth = (a / np.sqrt(p)) * (
        0.008 - 0.0048 * B - 0.047 * p - 0.099 * B * p 
        + (0.32 + 0.015 * B) * p * np.log(p) 
        + np.log(B) * (0.0056 - 0.89 * p + 0.29 * p * np.log(p))
    )
    
    length = (a / p**2) * (
        0.0053 - 0.21 * p + 1.3 * p**2 + (-0.11 - 0.17 * B) * p**2 * np.log(p)
        + B * (-0.0062 + 0.23 * p + 0.75 * p**2)
    )
    
    width = (a / (B * p**3)) * (
        0.0021 - 0.047 * p + 0.34 * p**2 - 1.9 * p**3 - 0.33 * p**4
        + B * (0.00066 - 0.0070 * p - 0.00059 * p**2 + 2.8 * p**3 - 0.12 * p**4)
        + B**2 * (-0.00070 + 0.015 * p - 0.12 * p**2 + 0.59 * p**3 - 0.023 * p**4)
        + B**3 * (0.00001 - 0.00022 * p + 0.0020 * p**2 - 0.0085 * p**3 + 0.0014 * p**4)
    )
    
    # Returning the Absolute Value of depth to fix the negative coordinate issue
    return max(0, length), max(0, width), abs(depth)


def run_logged_comparison():
    logging.info("Starting Multi-Model Comparison Test (Including Integrals)")
    
    material = {
        'rho': 7900.0,
        'C_p': 500.0,
        'k': 15.0,
        'T_m': 1673.0,
        'T_b': 3100.0,
        'A': 0.4
    }
    material['alpha'] = material['k'] / (material['rho'] * material['C_p'])
    
    P = 500.0
    a = 50e-6
    T_ambient = 0
    
    speeds = np.linspace(0.1, 1, 15)
    logging.info(f"Testing {len(speeds)} speeds from {speeds[0]:.2f} to {speeds[-1]:.2f} m/s at {P}W\n")
    
    # Storage
    et_depths, et_widths, et_lengths = [], [], []
    gs_depths = []
    rub_int_depths, rub_int_widths, rub_int_lengths = [], [], []
    rub_eq_depths, rub_eq_widths, rub_eq_lengths = [], [], []
    
    for v in speeds:
        logging.info(f"--- Processing Speed: {v:.3f} m/s ---")
        
        # 1. Rubenchik Interpolated (Equation 11)
        L_eq, W_eq, D_eq = standalone_rubenchik_interpolation(P, v, a, material, T_ambient)
        rub_eq_depths.append(D_eq * 1e6)
        rub_eq_widths.append(W_eq * 1e6)
        rub_eq_lengths.append(L_eq * 1e6)
        
        # 2. Rubenchik Integral (from physics.py)
        L_int, W_int, D_int, _, _ = physics.get_rubenchik_dimensions(P, v, a, material, T_ambient=T_ambient, resolution=50)
        rub_int_depths.append(D_int * 1e6)
        rub_int_widths.append(W_int * 1e6)
        rub_int_lengths.append(L_int * 1e6)

        # 3. Eagar-Tsai
        L_et, W_et, D_et, _, _ = physics.get_eagar_tsai_dimensions(P, v, a, material, T_ambient=T_ambient, resolution=50)
        et_depths.append(D_et * 1e6)
        et_widths.append(W_et * 1e6)
        et_lengths.append(L_et * 1e6)

        # 4. Gladush-Smurov
        D_gs = physics.get_melt_depth_gladush_smurov(P, v, a, material)
        gs_depths.append(max(0.0, D_gs * 1e6))
        
        logging.info(f"  -> Rubenchik (Eq 11)   : Depth={D_eq*1e6:.1f}µm, Width={W_eq*1e6:.1f}µm, Length={L_eq*1e6:.1f}µm")
        logging.info(f"  -> Rubenchik (Integral): Depth={D_int*1e6:.1f}µm, Width={W_int*1e6:.1f}µm, Length={L_int*1e6:.1f}µm\n")


    # --- Plotting ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # 1. Depth Plot
    axes[0].plot(speeds, et_depths, 'g-^', label='Eagar-Tsai')
    axes[0].plot(speeds, gs_depths, 'm-v', label='Gladush-Smurov')
    axes[0].plot(speeds, rub_int_depths, 'b-o', label='Rubenchik (Integral)')
    axes[0].plot(speeds, rub_eq_depths, 'r--s', label='Rubenchik (Eq 11)')
    axes[0].set_title('Melt Pool Depth')
    axes[0].set_xlabel('Scan Speed (m/s)')
    axes[0].set_ylabel('Depth (μm)')
    axes[0].grid(True); axes[0].legend()
    
    # 2. Width Plot
    axes[1].plot(speeds, et_widths, 'g-^', label='Eagar-Tsai')
    axes[1].plot(speeds, rub_int_widths, 'b-o', label='Rubenchik (Integral)')
    axes[1].plot(speeds, rub_eq_widths, 'r--s', label='Rubenchik (Eq 11)')
    axes[1].set_title('Melt Pool Width')
    axes[1].set_xlabel('Scan Speed (m/s)')
    axes[1].set_ylabel('Width (μm)')
    axes[1].grid(True); axes[1].legend()

    # 3. Length Plot
    axes[2].plot(speeds, et_lengths, 'g-^', label='Eagar-Tsai')
    axes[2].plot(speeds, rub_int_lengths, 'b-o', label='Rubenchik (Integral)')
    axes[2].plot(speeds, rub_eq_lengths, 'r--s', label='Rubenchik (Eq 11)')
    axes[2].set_title('Melt Pool Length')
    axes[2].set_xlabel('Scan Speed (m/s)')
    axes[2].set_ylabel('Length (μm)')
    axes[2].grid(True); axes[2].legend()
    
    plt.tight_layout()
    
    # Save the plot
    plot_file = os.path.join(output_dir, 'model_comparison_plot_with_integral.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    logging.info(f"Plot successfully saved to: {plot_file}")
    
    plt.show()

if __name__ == "__main__":
    run_logged_comparison()