import math

def check(dimensions, process_parameters, material):
    """
    Yadroitsev Stability criterion for Balling.
    Returns True if the track fragments into droplets.
    """
    L = dimensions.get('L', 0)
    W = dimensions.get('W', 0)
    
    if L == 0:
        return True
        
    return (math.pi * W / L) <= math.sqrt(2 / 3)