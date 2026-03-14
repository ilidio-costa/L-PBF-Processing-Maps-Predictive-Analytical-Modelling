import sys
import os
import matplotlib
# Force matplotlib to not use any Xwindows backend / block GUI popups
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.plots import plot_safe_zone_evolution
from src.data_loader import load_material

def test_safe_zone_evolution_a():
    print("Starting Safe Zone Evolution 3D Map generation...")
    
    # 1. Setup base parameters (matching test_all_defects_individually.py)
    Power_range = (50, 500)       
    Scan_Speed_range = (0.1, 3.5) 
    resolution = 35              

    material = load_material("NiTi_Sheikh.json")
    # Base parameters; 'a' will be dynamically overridden by the plotting function
    process_parameters = {'t': 30e-6, 'h': 80e-6, 'a': 40e-6} 
    
    # 2. Define the new dimension to vary and the steps
    z_var = 'a' # Laser beam radius
    z_values = [20e-6, 30e-6, 40e-6, 50e-6, 60e-6, 70e-6, 80e-6] # Slices at 20, 40, 60, and 80 microns
    
    # Ensure output directory exists
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)

    # 3. Compute and plot
    try:
        print(f"Evaluating 3D stack for variable '{z_var}' over values: {z_values}")
        fig = plot_safe_zone_evolution(
            Power_range=Power_range,
            Scan_Speed_range=Scan_Speed_range,
            material=material,
            base_process_parameters=process_parameters,
            z_var=z_var,
            z_values=z_values,
            resolution=resolution
        )
        
        # 4. Save the plot
        save_path = os.path.join(output_dir, "safe_zone_evolution_a.png")
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Successfully saved 3D evolution plot to: {save_path}")
        
    except Exception as e:
        print(f"FAILED to plot Safe Zone Evolution. Error: {e}")

if __name__ == "__main__":
    test_safe_zone_evolution_a()