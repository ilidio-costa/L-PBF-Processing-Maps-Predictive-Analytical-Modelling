import numpy as np

def check(dimensions, process_parameters, material):
    """
    Gan Universal keyhole criterion. 
    """

    A = material.get('A')
    rho = material.get('rho')
    C_p = material.get('C_p')
    T_m = material.get('T_m')
    alpha = material.get('alpha')
    P = process_parameters.get('P')
    v = process_parameters.get('v')
    a = process_parameters.get('a', 50e-6)
    T_ambient = process_parameters.get('T_ambient', 300)

    term = (A * P) / ((T_m - T_ambient) * np.pi * rho * C_p * np.sqrt(alpha * v * a**3))

        
    return  term > 6.0