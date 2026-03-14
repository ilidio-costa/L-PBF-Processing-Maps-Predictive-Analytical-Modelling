import sys
import os
import glob
import time
import matplotlib
# Force matplotlib to not use any Xwindows backend / block GUI popups
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.physics import compute_printability_map
from src.plots import plot_deterministic_map
from src.data_loader import load_material

def test_all_individual_defects():
    print("Starting individual defect criteria maps generation...")
    
    # 1. Setup parameters
    Power_range = (50, 500)       
    Scan_Speed_range = (0.1, 3.5) 
    resolution = 40              

    material = load_material("NiTi_Sheikh.json")
    process_parameters = {'t': 30e-6, 'h': 80e-6, 'a': 40e-6}
    
    # Ensure output and log directories exist
    output_dir = os.path.join(os.path.dirname(__file__), 'output', 'individual_defects')
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    log_file_path = os.path.join(log_dir, 'individual_defects_timing.log')

    # 2. Find all defect modules in src/defects/
    defects_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'defects')
    defect_files = glob.glob(os.path.join(defects_dir, '*.py'))
    
    # Start total timer
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
            if module_name.startswith('ball'):
                active_defects['balling'] = module_name
            elif module_name.startswith('lof'):
                active_defects['lof'] = module_name
            elif module_name.startswith('key'):
                active_defects['keyhole'] = module_name
            else:
                skip_msg = f"  Skipping {module_name}: Unknown category prefix."
                print(skip_msg)
                log_file.write(skip_msg + "\n")
                continue

            # Start individual module timer
            module_start_time = time.time()

            # 3. Compute the map
            P_grid, v_grid, defect_map = compute_printability_map(
                Power_range, 
                Scan_Speed_range, 
                material, 
                process_parameters, 
                resolution=resolution, 
                active_defects=active_defects
            )

            # 4. Generate and save the plot
            try:
                fig = plot_deterministic_map(P_grid, v_grid, defect_map, material_name=f"{material.get('name', 'NiTi')} - {module_name}")
                
                save_path = os.path.join(output_dir, f"map_{module_name}.png")
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                
                # Close the figure
                plt.close(fig)
                
                # End individual timer and calculate elapsed time
                module_end_time = time.time()
                elapsed_time = module_end_time - module_start_time
                
                print(f"  Saved plot to: {save_path} (Took {elapsed_time:.2f}s)")
                log_file.write(f"{module_name}: {elapsed_time:.2f} seconds\n")
                
            except Exception as e:
                err_msg = f"  FAILED to plot {module_name}. Error: {e}"
                print(err_msg)
                log_file.write(f"{module_name}: FAILED - {e}\n")

        # End total timer
        total_end_time = time.time()
        total_elapsed_time = total_end_time - total_start_time
        
        final_msg = f"\nAll individual maps generated successfully in {total_elapsed_time:.2f} seconds!"
        print(final_msg)
        
        log_file.write("-" * 40 + "\n")
        log_file.write(f"Total execution time: {total_elapsed_time:.2f} seconds\n")
        print(f"Log saved to: {log_file_path}")

if __name__ == "__main__":
    test_all_individual_defects()