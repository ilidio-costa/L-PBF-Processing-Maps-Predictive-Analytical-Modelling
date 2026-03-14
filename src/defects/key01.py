from re import A
import numpy as np

def check(dimensions, process_parameters, material):
    """
    King et al. Normalized Enthalpy criterion for Keyhole porosity.
    """
    
    A = material.get('A')
    C_p = material.get('C_p')
    T_m = material.get('T_m')
    T_b = material.get('T_b')
    alpha = material.get('alpha')
    rho = material.get('rho')
    P = process_parameters.get('P')
    v = process_parameters.get('v')
    a = process_parameters.get('a', 50e-6)

    term01 = (A * P) / (np.pi * rho * C_p * T_m * np.sqrt(alpha * v * a**3) )
    term02 = (np.pi * T_b) / T_m
    
    return term01 > term02