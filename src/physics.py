import numpy as np
from scipy.integrate import quad, fixed_quad
from scipy.optimize import minimize_scalar, brentq, newton

## ======================= Eagar-Tsai Model ======================= ##

def s_func_substituted(u, x, y, z, v, alpha, a):
    """ Transformed Integrand: s = u^2 (Removes singularity) """
    u = np.atleast_1d(u)
    s = u**2
    denom = 4 * alpha * s + a**2
    term_z = z**2 / (4 * alpha * s) if z != 0 else np.zeros_like(s)
    term_lat = (y**2 + (x + v * s)**2) / denom
    return 2 * np.exp(-term_z - term_lat) / denom

def eagar_tsai_temp(x, y, z, P, v, a, material, T_ambient=0):
    """
    Calculates T using the optimized u-substitution method 
    with high-speed dynamic spatial boundaries (Fast Fixed Quad).
    """
    A_val = material['A']
    k = material['k']
    alpha = material['alpha']
    
    # Pre-calculated constant factor
    pre_factor = (A_val * P / (np.pi * k)) * np.sqrt(alpha / np.pi)

    # --- 1. FAST-PASS UNDERFLOW FILTER ---
    u_peak_guess = np.sqrt(max(0, -x / v) + (a / v))
    max_val = float(np.squeeze(s_func_substituted(u_peak_guess, x, y, z, v, alpha, a)))
    
    if max_val < 1e-20:
        return T_ambient
        
    # --- 2. DYNAMIC INTEGRATION BOUNDARY ---
    upper_calc_limit = np.sqrt((abs(x)/v) * 3 + (x**2+y**2+z**2)/(4*alpha) + 100*(a/v))

    # --- 3. ROBUST HIGH-SPEED INTEGRATION ---
    from scipy.integrate import fixed_quad
    integral_val, _ = fixed_quad(
        s_func_substituted, 
        0.0, 
        upper_calc_limit, 
        args=(x, y, z, v, alpha, a),
        n=250  # Validated node count for optimal speed/accuracy ratio
    )
    
    return (pre_factor * integral_val) + T_ambient

def get_eagar_tsai_dimensions(P, v, a, material, T_ambient=0, resolution=100):
    """
    Calculates Melt Pool Dimensions by performing a global search for 
    max width and max depth along the melt pool length.
    
    Returns: length, width, depth, x_tail, x_front
    """
    Tm = material['T_m']
    
    # --- 1. DYNAMIC TOLERANCE ---
    # High resolution needed for gradients
    if resolution:
        tol = a / float(resolution) 
    else:
        tol = 1e-6 

    # --- 2. FIND PEAK & CHECK EXISTENCE ---
    # We first find the peak temperature on the surface centerline
    res_peak = minimize_scalar(
        lambda x: -eagar_tsai_temp(x, 0, 0, P, v, a, material, T_ambient=T_ambient), 
        bounds=(-5*a, a), 
        method='bounded',
        options={'xatol': tol}
    )
    x_peak = res_peak.x
    T_max = -res_peak.fun
    
    if T_max < Tm:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    # --- 3. DEFINE LENGTH BOUNDARIES (x_tail, x_front) ---
    # We solve T(x, 0, 0) = Tm to find the start and end of the pool
    func_x = lambda x: eagar_tsai_temp(x, 0, 0, P, v, a, material, T_ambient=T_ambient) - Tm
    
    # -- Find Front (Start) --
    step = a
    x_scan_fwd = x_peak + step
    # Scan forward until temp drops below Tm
    while eagar_tsai_temp(x_scan_fwd, 0, 0, P, v, a, material, T_ambient=T_ambient) > Tm:
        x_scan_fwd += step
        step *= 1.5
        if x_scan_fwd > x_peak + 0.1: break # Safety break

    try:
        x_front = brentq(func_x, x_peak, x_scan_fwd, xtol=tol)
    except ValueError:
        x_front = x_peak

    # -- Find Tail (End) --
    step = a
    x_scan_bwd = x_peak - step
    # Scan backward until temp drops below Tm
    while eagar_tsai_temp(x_scan_bwd, 0, 0, P, v, a, material, T_ambient=T_ambient) > Tm:
        x_scan_bwd -= step
        step *= 1.5
        if x_scan_bwd < x_peak - 0.1: break

    try:
        x_tail = brentq(func_x, x_scan_bwd, x_peak, xtol=tol)
    except ValueError:
        x_tail = x_peak
    
    length = x_front - x_tail

    # === 4. GLOBAL DEPTH OPTIMIZATION ===
    # Find the deepest point z_min along the entire length [x_tail, x_front]
    
    def get_depth_at_x(x_loc):
        # If surface is below Tm, depth is 0
        if eagar_tsai_temp(x_loc, 0, 0, P, v, a, material, T_ambient=T_ambient) < Tm:
            return 0.0
        
        # Search vertically for T = Tm
        func_z = lambda z: eagar_tsai_temp(x_loc, 0, z, P, v, a, material, T_ambient=T_ambient) - Tm
        
        # Find bracket
        z_scan = -a
        step_z = a
        while eagar_tsai_temp(x_loc, 0, z_scan, P, v, a, material, T_ambient=T_ambient) > Tm:
            z_scan -= step_z
            step_z *= 1.5
            if z_scan < -0.01: break
            
        try:
            z_root = brentq(func_z, z_scan, 0, xtol=tol)
            return abs(z_root)
        except ValueError:
            return 0.0

    # Maximize depth (minimize negative depth)
    res_d = minimize_scalar(
        lambda x: -get_depth_at_x(x), 
        bounds=(x_tail, x_front), 
        method='bounded',
        options={'xatol': tol}
    )
    depth = -res_d.fun

    # === 5. GLOBAL WIDTH OPTIMIZATION ===
    # Find the widest point y_max along the entire length [x_tail, x_front]

    def get_width_at_x(x_loc):
        if eagar_tsai_temp(x_loc, 0, 0, P, v, a, material, T_ambient=T_ambient) < Tm:
            return 0.0
        
        # Search laterally for T = Tm
        func_y = lambda y: eagar_tsai_temp(x_loc, y, 0, P, v, a, material, T_ambient=T_ambient) - Tm
        
        y_scan = a
        step_y = a
        while eagar_tsai_temp(x_loc, y_scan, 0, P, v, a, material, T_ambient=T_ambient) > Tm:
            y_scan += step_y
            step_y *= 1.5
            if y_scan > 0.01: break
        
        try:
            y_edge = brentq(func_y, 0, y_scan, xtol=tol) 
            return y_edge * 2.0 # Total width
        except ValueError:
            return 0.0

    # Maximize width
    res_w = minimize_scalar(
        lambda x: -get_width_at_x(x), 
        bounds=(x_tail, x_front), 
        method='bounded',
        options={'xatol': tol} 
    )
    width = -res_w.fun

    return length, width, depth, x_tail, x_front


## ======================= Rubenchik Model ====================== ##




def g_func(t, xi, yi, zi, p):
    """
    Integrand for the dimensionless temperature function g.
    t: Dimensionless time
    xi, yi, zi: Dimensionless coordinates (x/a, y/a, z/a)
    p: Diffusivity parameter
    """
    # Safety check (though we integrate from 1e-12, quad might sample close to 0)
    if t <= 0: return 0.0

    # 1. Denominator Terms
    # Geometric spreading: 4*p*t + 1
    denom_geom = 4 * p * t + 1
    
    # 2. Exponentials
    # Vertical component: -zi^2 / (4t)
    term_z = zi**2 / (4 * t)
    
    # Lateral component: -(yi^2 + (xi - t)**2) / (4pt + 1)
    # The source moves in +x direction in this dimensionless frame
    term_lateral = (yi**2 + (xi + t)**2) / denom_geom
    
    exp_val = np.exp(-term_z - term_lateral)
    
    # 3. Final Combination
    # Singularity at t=0 comes from sqrt(t)
    return exp_val / (denom_geom * np.sqrt(t))

def rubenchik_variables(x, y, z, material, P, v, a, T_ambient=0):
    """
    Computes the dimensionless variables xi, yi, zi for Rubenchik's model.
    """
    rho = material['rho']
    Cp = material['C_p']
    T_m = material['T_m']
    alpha = material['alpha']
    A_val = material['A']
    
    # Dimensionless coordinates (normalized by beam radius a)
    xi = x / a
    yi = y / a
    zi = z / a

    # Diffusivity parameter
    p_val = alpha / (v * a)

    # Power equivalent
    P_e = P * (T_m - T_ambient) / T_m

    # B value (Dimensionless intensity)
    numerator = A_val * P_e
    denominator = np.pi * rho * Cp * T_m * np.sqrt(alpha * v * a**3)
    B_val = numerator / denominator

    coords = (xi, yi, zi)
    return coords, B_val, p_val

def rubenchik_temp(coords, B, p, material, T_ambient=0):
    """
    Computes the temperature in Kelvin using the Rubenchik model.
    """
    xi, yi, zi = coords

    # --- Robust Integration Limits ---
    # Lower Limit: 1e-12 avoids the 1/sqrt(t) singularity at 0
    # Upper Limit: 10000 is sufficiently large for the wake to decay 
    # (Dimensionless time t=10000 >> 1)
    
    g_value, error = quad(
        g_func, 
        1e-12,      # Start slightly above 0 to avoid singularity
        10000.0,    # Finite upper limit avoids "slow convergence" warnings
        args=(xi, yi, zi, p), 
        limit=200   # Increased limit to handle sharp peaks
    )

    # Scale the dimensionless rise (B * g_value) by T_m
    T_rise_kelvin = B * g_value * material['T_m']

    return T_rise_kelvin + T_ambient

def rubenchik_field(P,v,a, material, T_ambient):
    """
    Scalings for melt pool parameters (Analytical Approximations)
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

    # Length (Analytical Fit)
    length = a/p**2 * (0.0053 - 0.21*p + 1.3*p**2 + (-0.11 - 0.17*B)*p**2 *np.log(p) + B*(-0.0062 + 0.23*p + 0.75*p**2))

    # Width (Analytical Fit)
    width = a/(B * p**3) * (0.0021 - 0.047*p + 0.34*p**2 - 1.9*p**3 - 0.33*p**4 + B*(0.00066 - 0.0070*p - 0.00059*p**2 + 2.8*p**3 - 0.12*p**4) + B**2 * (-0.00070 + 0.015*p - 0.12*p**2 + 0.59*p**3 - 0.023*p**4) + B**3 * (0.00001 - 0.00022*p + 0.0020*p**2 - 0.0085*p**3 + 0.0014*p**4))

    # Depth (Analytical Fit)
    depth = -a/np.sqrt(p) * (0.008 - 0.0048*B - 0.047*p - 0.099*B*p + (0.32+0.015*B)*p *np.log(p) + np.log(B)*(0.0056 - 0.89*p + 0.29*p*np.log(p)))

    return length, width, depth

## ================= Gladush-Smurov Model for Depth ==================== ##
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