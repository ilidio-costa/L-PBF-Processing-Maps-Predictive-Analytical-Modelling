from math import exp, pi, sqrt, log
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar, brentq

## ======================= Eagar-Tsai Model ======================= ##

def s_func(s, x, y, z, v, alpha, a):
    """
    Original Eagar-Tsai Integrand.
    s = Time lag (t - t')
    x, y, z = Spatial coordinates
    v = Scan speed [m/s]
    alpha = Thermal diffusivity [m^2/s]
    a = Laser beam radius [m]
    """
    if z >0:
        print("Warning: positive z depth in s_func")
    
    # 1. Denominator Terms
    # The spreading of the beam: 4*alpha*s + a^2
    denom_lateral = 4 * alpha * s + a**2
    
    # 2. Exponentials
    # Vertical component (Z): exp(-z^2 / 4*alpha*s)
    term_z = z**2 / (4 * alpha * s)
    
    # Lateral component (X, Y): 
    # The source moves, so the center is at (x - v*s)
    # Note: If x is positive ahead of the laser, (x-vs) is correct.
    term_lateral = (y**2 + (x + v * s)**2) / denom_lateral
    
    exp_val = np.exp(-term_z - term_lateral)
    
    # 3. Final Combination
    # Denominator includes sqrt(s) which is the source of the singularity
    denom_total = denom_lateral * np.sqrt(s)
    
    return exp_val / denom_total

def eagar_tsai_temp(x, y, z, P, v, a, material):
    """
    Calculates T using the original Eagar-Tsai formulation.
    Robust at high speeds due to 'Smart Limits'.
    x, y, z = Spatial coordinates
    P = Laser Power [W]
    v = Scan speed [m/s]
    a = Laser beam radius [m]
    material = Material properties dictionary
    A_val = Absorptivity (0-1)
    k = Thermal conductivity [W/m.K]
    alpha = Thermal diffusivity [m^2/s]
    """
    A_val = material['A']
    k = material['k']
    alpha = material['alpha']
    
    # Pre-calculated constant factor
    pre_factor = (A_val * P / (np.pi * k)) * np.sqrt(alpha / np.pi)

   # --- 1. FIND PEAK LOCATION ---
    # The peak contribution occurs when the laser passes the point.
    if x < 0:
        s_peak = abs(x) / v
        points_of_interest = [s_peak]
    else:
        s_peak = 0
        points_of_interest = []

    # --- 2. SMART UPPER LIMIT ---
    # Stop integrating when heat dissipates. 
    # Integrating to infinity causes "slowly convergent" warnings.
    if s_peak > 0:
        upper_limit = s_peak * 5.0 
    else:
        upper_limit = (a / v) * 10.0 
        
    upper_limit = max(upper_limit, 0.001) 

    # --- 3. ROBUST INTEGRATION ---
    # Start at 1e-9 instead of 0 to strictly avoid the 1/sqrt(s) singularity.
    # This is numerically safe and physically accurate.
    integral_val, error = quad(
        s_func, 
        1e-9,           # Lower limit > 0
        upper_limit,    # Finite upper limit
        args=(x, y, z, v, alpha, a),
        points=points_of_interest, 
        limit=100
    )
    
    return (pre_factor * integral_val)

def get_eagar_tsai_dimensions(P, v, a, material, resolution=100):
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
        lambda x: -eagar_tsai_temp(x, 0, 0, P, v, a, material), 
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
    func_x = lambda x: eagar_tsai_temp(x, 0, 0, P, v, a, material) - Tm
    
    # -- Find Front (Start) --
    step = a
    x_scan_fwd = x_peak + step
    # Scan forward until temp drops below Tm
    while eagar_tsai_temp(x_scan_fwd, 0, 0, P, v, a, material) > Tm:
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
    while eagar_tsai_temp(x_scan_bwd, 0, 0, P, v, a, material) > Tm:
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
        if eagar_tsai_temp(x_loc, 0, 0, P, v, a, material) < Tm:
            return 0.0
        
        # Search vertically for T = Tm
        func_z = lambda z: eagar_tsai_temp(x_loc, 0, z, P, v, a, material) - Tm
        
        # Find bracket
        z_scan = -a
        step_z = a
        while eagar_tsai_temp(x_loc, 0, z_scan, P, v, a, material) > Tm:
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
        if eagar_tsai_temp(x_loc, 0, 0, P, v, a, material) < Tm:
            return 0.0
        
        # Search laterally for T = Tm
        func_y = lambda y: eagar_tsai_temp(x_loc, y, 0, P, v, a, material) - Tm
        
        y_scan = a
        step_y = a
        while eagar_tsai_temp(x_loc, y_scan, 0, P, v, a, material) > Tm:
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

def get_melt_pool_dimensions(P, v, a, material, melt_temp=None):
    """
    Highly efficient calculation of melt pool dimensions using 1D root-finding 
    along principal axes instead of global surface optimization.
    
    Assumptions (Geometric Logic):
    - Max Temperature occurs on the scan centerline (y=0, z=0).
    - Max Depth occurs directly beneath the thermal peak (x=x_peak, y=0).
    - Max Width occurs at the thermal peak cross-section (x=x_peak, z=0).
    
    Parameters:
        P (float): Laser Power [W]
        v (float): Scan Speed [m/s]
        a (float): Beam Radius [m]
        material (dict): Material properties
        melt_temp (float, optional): Melting point. Defaults to material['T_m'].
        
    Returns:
        tuple: (length, width, depth, x_tail, x_front)
    """
    # 0. Setup
    
    melt_temp = material['T_m']
        
    # Helper for cleaner root finding calls
    def temp_diff(var, axis):
        # axis 0=x, 1=y, 2=z
        if axis == 0: return eagar_tsai_temp(var, 0, 0, P, v, a, material) - melt_temp
        if axis == 1: return eagar_tsai_temp(x_peak, var, 0, P, v, a, material) - melt_temp
        if axis == 2: return eagar_tsai_temp(x_peak, 0, var, P, v, a, material) - melt_temp

    # --- 1. FIND PEAK LOCATION (x_peak) ---
    # The thermal peak lags slightly behind the laser center (x=0) due to moving heat source.
    # We search in a small window [-5a, a].
    res_peak = minimize_scalar(
        lambda x: -eagar_tsai_temp(x, 0, 0, P, v, a, material), 
        bounds=(-5*a, a), 
        method='bounded',
        options={'xatol': 1e-7} # High precision for peak location
    )
    x_peak = res_peak.x
    T_max = -res_peak.fun
    
    # Check if melting occurs
    if T_max < melt_temp:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    # --- 2. FIND LENGTH (x_tail to x_front) ---
    # We search along the X-axis (y=0, z=0) for points where T = T_melt
    
    # Bracket for Front (Scanning direction, x > x_peak)
    # We step forward until T < melt_temp
    step = a
    x_scan_fwd = x_peak + step
    while eagar_tsai_temp(x_scan_fwd, 0, 0, P, v, a, material) > melt_temp:
        x_scan_fwd += step
        step *= 1.5
    
    try:
        x_front = brentq(temp_diff, x_peak, x_scan_fwd, args=(0,))
    except ValueError:
        x_front = x_peak # Fallback if very small pool

    # Bracket for Tail (Trailing edge, x < x_peak)
    step = a
    x_scan_bwd = x_peak - step
    while eagar_tsai_temp(x_scan_bwd, 0, 0, P, v, a, material) > melt_temp:
        x_scan_bwd -= step
        step *= 1.5
        
    try:
        x_tail = brentq(temp_diff, x_scan_bwd, x_peak, args=(0,))
    except ValueError:
        x_tail = x_peak

    length = x_front - x_tail

    # --- 3. FIND WIDTH (At x_peak) ---
    # We search along the Y-axis at x=x_peak. The profile is symmetric, so we find +y.
    # Bracket: y=0 (T=T_max) to y=large (T < T_melt)
    
    y_scan = a
    step_y = a
    # Quick expansion to find bracket
    while eagar_tsai_temp(x_peak, y_scan, 0, P, v, a, material) > melt_temp:
        y_scan += step_y
        step_y *= 1.5

    try:
        y_edge = brentq(temp_diff, 0, y_scan, args=(1,))
        width = y_edge * 2.0
    except ValueError:
        width = 0.0

    # --- 4. FIND DEPTH (At x_peak) ---
    # We search along the Z-axis (downwards) at x=x_peak.
    # Bracket: z=0 (T=T_max) to z=negative_large.
    # Note: eagar_tsai_temp handles z as signed coordinate? 
    # Provided code often treats depth as absolute or negative. 
    # Looking at provided s_func: "term_z = z**2...", so it is symmetric/insensitive to sign.
    # We will search for z < 0.
    
    z_scan = -a
    step_z = a
    while eagar_tsai_temp(x_peak, 0, z_scan, P, v, a, material) > melt_temp:
        z_scan -= step_z
        step_z *= 1.5

    try:
        z_bottom = brentq(temp_diff, z_scan, 0, args=(2,))
        depth = abs(z_bottom)
    except ValueError:
        depth = 0.0

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