def check(dimensions, process_parameters, material):
    """
    Geometric Overlap criterion for Lack of Fusion.
    Returns True if voids form between adjacent tracks.
    """

    D = dimensions.get('D', 0)
    t = process_parameters.get('t', 0)
        
    return D/t < 1.5