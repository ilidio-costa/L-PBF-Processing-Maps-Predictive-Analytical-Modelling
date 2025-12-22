import json
import os

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