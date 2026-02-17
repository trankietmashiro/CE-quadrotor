def quadrotor_param():
    """Returns a dict of fixed physical parameters for a quadrotor."""
    params = {}

    # Inertia around x-axis (kg·m²)
    params['Ixx'] = 7e-3

    # Inertia around y-axis (kg·m²)
    params['Iyy'] = 7e-3

    # Inertia around z-axis (kg·m²)
    params['Izz'] = 12e-3

    # Lift constant (N·s²)
    params['k'] = 8.55e-6 * 91.61

    # Distance from rotor to center of mass (m)
    params['l'] = 0.17

    # Mass of the quadrotor (kg)
    params['m'] = 0.716

    # Drag constant (N·m·s²)
    params['b'] = 1.6e-2 * 91.61

    # Gravitational acceleration (m/s²)
    params['g'] = 9.81

    return params