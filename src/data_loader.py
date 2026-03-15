import json
import os
import numpy as np

def load_material(filename):
    """
    Loads a material JSON file from the materials directory.
    Calculates derived properties like Thermal Diffusivity (alpha).
    """
    
    # 1. Get the directory where THIS script is located
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Go up one level to the project root
    project_root = os.path.dirname(current_script_dir)
    
    # 3. Build the full path to the materials folder
    file_path = os.path.join(project_root, "materials", filename)
    
    print(f"Loading material from: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Material file not found at: {file_path}")

    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # FIX: Use 'C_p' (from JSON) instead of 'Cp'.
    # Calculate Diffusivity alpha = k / (rho * Cp)
    # According to the nomenclature, alpha is Thermal Diffusivity 
    if 'alpha' not in data:
        try:
            data['alpha'] = data['k'] / (data['rho'] * data['C_p'])
        except KeyError as e:
            print(f"Error calculating alpha: Missing key {e} in {filename}")
            raise
    
    return data

def calculate_dynamic_absorptivity(wavelength, resistivity):
    """
    Calculates the laser absorptivity of a metallic surface 
    using the simplified Hagen-Rubens approximation.
    """
    # Using the simplified equation A = 0.365 * sqrt(rho_0 / lambda)
    A_effective = 0.365 * np.sqrt(resistivity / wavelength)
    
    # Failsafe: Absorptivity physically cannot exceed 1.0 (100%)
    return min(A_effective, 1.0)