from math import exp
from re import A
import numpy as np
from pyparsing import alphanums
from scipy.integrate import quad

# Eagar-Tsai Model
def s_func(s, x, y, z, v, alpha, a):
    """
    Integrand for the Eagar-Tsai model.
    This helper must be visible to get_temp_at_point.
    """
    if s <= 0: return 0
    
    term_z = z**2 / (4 * alpha * s)
    numerator_lat = y**2 + (x - v * s)**2
    denominator_lat = 4 * alpha * s + a**2
    term_lateral = numerator_lat / denominator_lat
    
    exp_val = np.exp(-term_z - term_lateral)
    denom_val = (4 * alpha * s + a**2) * np.sqrt(s)
    
    return exp_val / denom_val

def get_temp_at_point(x, y, z, P, v, a, material):
    """
    Calculates T for a single scalar point.
    """
    A = material['A']
    k = material['k']
    alpha = material['alpha']
    
    pre_factor = (A * P / (np.pi * k)) * np.sqrt(alpha / np.pi)

    # Calculate peak location
    integration_points = []
    if x < 0:
        # Peak is at s = |x|/v. Since s = tau^2, tau = sqrt(|x|/v)
        tau_peak = np.sqrt(abs(x) / v)
        integration_points = [tau_peak]  
    # Ensure our peak isn't beyond our limit (unlikely, but safe)  
    if integration_points and integration_points[0] > upper_limit:
        upper_limit = integration_points[0] * 2.0

    # We call s_func here, so s_func must be defined in this file
    integral_val, _ = quad(s_func, 0, 500**10, args=(x, y, z, v, alpha, a), points=integration_points, limit=100)
    
    return (pre_factor * integral_val)

def s_func_substituted(tau, x, y, z, v, alpha, a):
    """
    Substituted integrand (s = tau^2) to remove the 1/sqrt(s) singularity.
    This makes the integration stable at high speeds.
    """
    s = tau**2  # The substitution
    
    # 1. Handle Z-component (Vertical)
    # If s is very small, exp(-z^2/s) -> 0 for any depth z != 0
    if s == 0:
        if z != 0: return 0
        term_z = 0
    else:
        term_z = z**2 / (4 * alpha * s)

    # 2. Handle Lateral Component (X, Y)
    # CHANGE: I switched (x-vs) to (x+vs) to place the tail in the 
    # negative X direction (standard convention). 
    # If you prefer the tail in positive X, switch it back to minus.
    numerator_lat = y**2 + (x + v * s)**2 
    denominator_lat = 4 * alpha * s + a**2
    term_lateral = numerator_lat / denominator_lat
    
    # 3. Combine
    exp_val = np.exp(-term_z - term_lateral)
    
    # 4. New Denominator (The sqrt(s) cancelled out!)
    # We multiply by 2.0 because ds = 2*tau*dtau
    return (2.0 * exp_val) / denominator_lat

def get_temp_at_point_substituted(x, y, z, P, v, a, material):
    A = material['A']
    k = material['k']
    alpha = material['alpha']
    
    pre_factor = (A * P / (np.pi * k)) * np.sqrt(alpha / np.pi)
    
    # Calculate peak location
    integration_points = []
    if x < 0:
        # Peak is at s = |x|/v. Since s = tau^2, tau = sqrt(|x|/v)
        tau_peak = np.sqrt(abs(x) / v)
        integration_points = [tau_peak]

    # FIX: Use a finite upper limit (e.g., 5.0) instead of np.inf
    # tau = 5.0 corresponds to s = 25 seconds, which is effectively infinite for a melt pool
    upper_limit = 10e9
    
    # Ensure our peak isn't beyond our limit (unlikely, but safe)
    if integration_points and integration_points[0] > upper_limit:
        upper_limit = integration_points[0] * 2.0

    integral_val, _ = quad(
        s_func_substituted, 
        0, 
        upper_limit,  # <--- CHANGED from np.inf
        args=(x, y, z, v, alpha, a), 
        points=integration_points, 
        limit=100
    )
    
    return (pre_factor * integral_val)

# Rubenchik Model
def g_func(t, xi, yi, zi, p):
    """
    Integrand for the dimensionless temperature function g.
    FIX: 't' is now the first argument as required by scipy.integrate.quad.
    """
    if t == 0: return 0

    # These terms are now stable because p and t are guaranteed to be positive
    exp_term = (-(zi**2 / (4 * t)) - (yi**2 + (xi - t)**2) / (4 * p * t + 1))
    denominator = (4 * p * t + 1) * np.sqrt(t)
    
    return (np.exp(exp_term) / denominator)

def rubenchik(x, y, z, material, P, v, a, T_ambient=0):
    """
    Computes the dimensionless variables xi, yi, zi for Rubenchik's model.
    """
    rho = material['rho']
    Cp = material['C_p']
    T_m = material['T_m']
    alpha = material['alpha']
    A_val = material['A']
    
    # FIX 1: xi must be dimensionless (x / a)
    # The previous (v*x)/(alpha*a) had units of 1/m and was far too large.
    xi = x / a
    yi = y / a
    zi = z / a

    # p value is correctly alpha / (v * a)
    p_val = alpha / (v * a)

    # Power equivalent
    P_e = P * (T_m - T_ambient) / T_m

    # B value (Dimensionless intensity)
    numerator = A_val * P_e
    denominator = np.pi * rho * Cp * T_m * np.sqrt(alpha * v * a**3)
    B_val = numerator / denominator

    coords = (xi, yi, zi)
    return coords, B_val, p_val

def get_temp_rubenchik(coords, B, p, material, T_ambient=0):
    """
    Computes the temperature in Kelvin using the Rubenchik model.
    """
    xi, yi, zi = coords

    # Numerical integration of the dimensionless function g
    # limit=500 is a safe upper bound for dimensionless time t
    g_value, _ = quad(g_func, 0, 500**10, args=(xi, yi, zi, p), limit=50)

    # FIX 2: Scale the dimensionless rise (B * g_value) by T_m to get Kelvin.
    # The previous code was dividing by T_m, resulting in ~0.
    T_rise_kelvin = B * g_value * material['T_m']

    return T_rise_kelvin + T_ambient

def rubenchik_field(P,v,a, material, T_ambient):
    """
    Scalings for melt pool parameters
    """
    rho = material['rho']
    Cp = material['C_p']
    T_m = material['T_m']
    alpha = material['alpha']
    A_val = material['A']

    # p value is correctly alpha / (v * a)
    p = alpha / (v * a)

    # Power equivalent
    P_e = P * (T_m - T_ambient) / T_m

    # B value (Dimensionless intensity)
    numerator = A_val * P_e
    denominator = np.pi * rho * Cp * T_m * np.sqrt(alpha * v * a**3)
    B = numerator / denominator



    # Length
    length = a/p**2 * (0.0053 - 0.21*p + 1.3*p**2 + (-0.11 - 0.17*B)*p**2 *np.log(p) + B*(-0.0062 + 0.23*p + 0.75*p**2))

    # Width
    width = a/(B * p**3) * (0.0021 - 0.047*p + 0.34*p**2 - 1.9*p**3 - 0.33*p**4 + B*(0.00066 - 0.0070*p - 0.00059*p**2 + 2.8*p**3 - 0.12*p**4) + B**2 * (-0.00070 + 0.015*p - 0.12*p**2 + 0.59*p**3 - 0.023*p**4) + B**3 * (0.00001 - 0.00022*p + 0.0020*p**2 - 0.0085*p**3 + 0.0014*p**4))

    # Depth
    depth = -a/np.sqrt(p) * (0.008 - 0.0048*B - 0.047*p - 0.099*B*p + (0.32+0.015*B)*p *np.log(p) + np.log(B)*(0.0056 - 0.89*p + 0.29*p*np.log(p)))

    return length, width, depth

# Gladush-Smurov Model for Depth
def get_melt_depth_gladush(P, v, a, material):
    '''
    Computes melt pool depth using the Gladush-Smurov empirical model.
    '''
    A_val = material['A']
    k_val = material['k']
    T_b = material['T_b']
    alpha = material['alpha']

    C1 = A_val*P / (2 * np.pi * k_val * T_b)

    Depth = C1 * np.log( (a + alpha/v) / a**2 )

    return Depth

# Defect Criteria Functions
def get_defect_masks(P, v, a, L, W, D, material, layer_t, hatch_s):
    """
    Implements the 3 specific defect criteria from the project report.
    """
    # 1. Balling (Yadroitsev) - True if unstable
    # Stability: (pi * W / L) > sqrt(2/3)
    balling_mask = (np.pi * W / L) <= np.sqrt(2/3)
    
    # 2. Keyhole (Rubenchik) - True if in keyhole mode
    # Normalized Enthalpy > (pi * Tb / Tm)
    rho, Cp, Tm, Tb, alpha, A = material['rho'], material['C_p'], material['T_m'], \
                                material['T_b'], material['alpha'], material['A']
    
    norm_enthalpy = (A * P) / (np.pi * rho * Cp * Tm * np.sqrt(alpha * v * a**3))
    keyhole_mask = norm_enthalpy > (np.pi * Tb / Tm)
    
    # 3. Lack of Fusion (Seede) - True if insufficient overlap
    # (h/W)^2 + t/(t+D) >= 1
    lof_val = (hatch_s / W)**2 + (layer_t / (layer_t + D))
    lof_mask = lof_val >= 1
    
    return balling_mask, keyhole_mask, lof_mask
    """
    Seede Criterion: Lack of fusion if (h/W)^2 + t/(t+D) >= 1
    """
    term1 = (hatch_spacing / width) ** 2
    term2 = layer_thickness / (layer_thickness + depth)
    return (term1 + term2) >= 1.0