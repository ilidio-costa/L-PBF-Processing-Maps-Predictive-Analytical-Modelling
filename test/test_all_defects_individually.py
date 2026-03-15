import sys
import os
import glob
import time
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.physics import compute_printability_map
from src.plots import plot_deterministic_map
from src.data_loader import load_material

def test_all_individual_defects():
    print("Starting individual defect criteria maps generation...")
    
    Power_range = (50, 500)       
    Scan_Speed_range = (0.1, 3.5) 
    resolution = 50              

    material = load_material("NiTi_Sheikh.json")
    
    # Updated to include T_ambient
    process_parameters = {'t': 30e-6, 'h': 80e-6, 'a': 40e-6, 'T_ambient': 298.0}
    
    output_dir = os.path.join(os.path.dirname(__file__), 'output', 'individual_defects')
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    log_file_path = os.path.join(log_dir, 'individual_defects_timing.log')
    defects_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'defects')
    defect_files = glob.glob(os.path.join(defects_dir, '*.py'))
    
    total_start_time = time.time()

    with open(log_file_path, 'w') as log_file:
        log_file.write("Starting individual defect criteria maps generation...\n")
        log_file.write(f"Resolution: {resolution}x{resolution}\n")
        log_file.write("-" * 40 + "\n")

        for file_path in defect_files:
            module_name = os.path.basename(file_path).replace('.py', '')
            if module_name.startswith('__'):
                continue
                
            print(f"\nEvaluating defect criteria: {module_name}")
            active_defects = {}
            if module_name.startswith('ball'): active_defects['balling'] = module_name
            elif module_name.startswith('lof'): active_defects['lof'] = module_name
            elif module_name.startswith('key'): active_defects['keyhole'] = module_name
            else: continue

            module_start_time = time.time()

            P_grid, v_grid, defect_map = compute_printability_map(
                Power_range, Scan_Speed_range, material, process_parameters, 
                resolution=resolution, active_defects=active_defects
            )

            try:
                fig = plot_deterministic_map(P_grid, v_grid, defect_map, material_name=f"{material.get('name', 'NiTi')} - {module_name}")
                save_path = os.path.join(output_dir, f"map_{module_name}.png")
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                
                elapsed_time = time.time() - module_start_time
                print(f"  Saved plot to: {save_path} (Took {elapsed_time:.2f}s)")
                log_file.write(f"{module_name}: {elapsed_time:.2f} seconds\n")
            except Exception as e:
                print(f"  FAILED to plot {module_name}. Error: {e}")
                log_file.write(f"{module_name}: FAILED - {e}\n")

        total_elapsed_time = time.time() - total_start_time
        print(f"\nAll individual maps generated successfully in {total_elapsed_time:.2f} seconds!")
        log_file.write("-" * 40 + "\n")
        log_file.write(f"Total execution time: {total_elapsed_time:.2f} seconds\n")

if __name__ == "__main__":
    test_all_individual_defects()