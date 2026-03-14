import sys
import os
import matplotlib.pyplot as plt

# --- Set up Project Paths ---
test_dir = os.path.dirname(__file__)
output_dir = os.path.join(test_dir, 'output')
os.makedirs(output_dir, exist_ok=True)

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(test_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now Python can find the src modules
from src import plots

def run_hybrid_plot_test():
    print("Starting Hybrid Dimension Plot Test...")
    
    # 1. Define Material Properties (Approximating SS316L)
    material = {
        "name": "NiTi",
        "rho": 6100,
        "C_p": 510,
        "k": 4.4,
        "T_b": 3033,
        "T_m": 1583,
        "A": 0.32
    }
    material['alpha'] = material['k'] / (material['rho'] * material['C_p'])
    
    # 2. Define Plot Parameters
    fixed_params = {'a': 50e-6}  # Fixed spot radius (50 µm)
    y_range = (50, 500)          # Power range (50W to 350W)
    x_range = (0.35, 3.5)         # Velocity range (0.2 m/s to 2.0 m/s)
    
    # 3. Generate the Plot
    # Notice we pass `use_max_gs_et=True` to trigger the new hybrid depth logic
    fig = plots.plot_melt_pool_dimensions(
        x_var='v', 
        y_var='P', 
        x_range=x_range, 
        y_range=y_range, 
        fixed_params=fixed_params, 
        material=material,
        T_ambient=293.0,
        use_rubenchik=True,        # Use Rubenchik for Length/Width
        use_max_gs_et=True,        # Use the hybrid GS/ET envelope for Depth
        resolution=30              # Keep resolution moderate for quick testing
    )
    
    # 4. Save and Display
    plot_file = os.path.join(output_dir, 'hybrid_dimension_map.png')
    fig.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Plot successfully generated and saved to: {plot_file}")
    
    plt.show()

if __name__ == "__main__":
    run_hybrid_plot_test()