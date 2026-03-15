import sys
import os
import matplotlib
# Force matplotlib to not use any Xwindows backend / block GUI popups
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.plots import plot_safe_zone_evolution
from src.data_loader import load_material

def test_all_safe_zone_evolutions():
    print("Starting comprehensive Safe Zone Evolution 3D Map generation...")
    
    Power_range = (50, 500)       
    Scan_Speed_range = (0.1, 3.5) 
    resolution = 40              

    material = load_material("NiTi_Sheikh.json")
    
    # SAFEGUARD: Ensure the material has electrical resistivity for the wavelength sweep
    if 'electrical_resistivity' not in material:
        material['electrical_resistivity'] = 8.2e-7 
        
    # Base parameters; the plotting function will copy this and dynamically 
    # overwrite the specific variable being swept.
    base_process_parameters = {'t': 30e-6, 'h': 80e-6, 'a': 40e-6, 'T_ambient': 298.0} 
    
    # Define the variables to test and the specific values to stack
    sweeps = {
        'wavelength': [300e-9, 500e-9, 700e-9, 1000e-9],        
        'A': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
        'h': [50e-6, 70e-6, 90e-6, 110e-6],
        'a': [20e-6, 30e-6, 40e-6, 50e-6, 60e-6, 70e-6, 80e-6],
        'T_ambient': [0, 100, 200, 300, 400, 500],
        't': [20e-6, 30e-6, 40e-6, 50e-6]
    }
    
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)

    for z_var, z_values in sweeps.items():
        try:
            print(f"\nEvaluating 3D stack for variable '{z_var}' over values: {z_values}")
            
            fig = plot_safe_zone_evolution(
                Power_range=Power_range,
                Scan_Speed_range=Scan_Speed_range,
                material=material,
                base_process_parameters=base_process_parameters,
                z_var=z_var,
                z_values=z_values,
                resolution=resolution
            )
            
            # Save each plot with a descriptive filename
            save_path = os.path.join(output_dir, f"safe_zone_evolution_{z_var}.png")
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig) # Prevent memory leaks
            
            print(f"  --> Successfully saved 3D evolution plot to: {save_path}")
            
        except Exception as e:
            print(f"  --> FAILED to plot Safe Zone Evolution for {z_var}. Error: {e}")

if __name__ == "__main__":
    test_all_safe_zone_evolutions()