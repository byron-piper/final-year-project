import math
import logging

import numpy as np

from helper import cubic_bezier, closest_idx, slope_eq_at_idx

def generate_naca4(chord: float = 1, M: int = 2, P: int = 4, T: int = 12, num_points: int = 100, closed_trailing_edge: bool = True, cosine_spacing: bool = True) -> np.ndarray:
    """
    Some documentation.
    """
    
    logging.info(f"Generating NACA-4 aerofoil with configuration: chord = {chord}, M = {M}, P = {P}, T = {T}, num_points = {num_points}")

    # Re-scale NACA-4 digits to appropriate digits
    M = 0.09 if M > 9 else M / 100
    P = 0.9 if P > 9 else P / 10
    T = 0.4 if T > 40 else T / 100

    # Instantiate 1D vector for x, use cosine spacing if requested
    if cosine_spacing:
        beta = np.linspace(0, math.pi, num_points)
        x = (1 - np.cos(beta)) / 2
    else:
        x = np.linspace(0, 1, num_points)

    a4 = -0.1015 if not closed_trailing_edge else -0.1036

    # Generate the thickness distribution
    yt = (T/0.2) * (0.2969*x**0.5 - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 + a4*x**4)
    
    # Generate camber line
    if M > 0:
        yc = np.where(x < P, (M/P**2)*(2*P*x - x**2), (M/(1 - P)**2)*(1 - 2*P + 2*P*x - x**2))
        dyc_dx = np.where(x < P, ((2*M)/P**2)*(P - x), ((2*M) / (1 - P)**2)*(P - x))
    else:
        yc = np.zeros_like(x)
        dyc_dx = np.zeros_like(x)
    
    theta = np.arctan(dyc_dx)

    # [::-1] used to reverse array, [1:-1] used to exclude first and last index else duplicate points will be added
    # Aerofoil is constructed starting from the trailing edge and made anti-clockwise
    x = np.concat((x[::-1], x[1:-1]))
    y = np.concat((yc[::-1] + yt[::-1] * np.cos(theta[::-1]), yc[1:-1] - yt[1:-1] * np.cos(theta[1:-1])))

    coords = np.column_stack((x, y, np.zeros(2*num_points-2)))

    return coords

def aerofoil_to_3element(chord: float, coords: np.ndarray, slat_geom: list, flap_geom: list, num_t: int = 50):
    logging.info(f"Converting aerofoil to 3-element: Slat geometry = {slat_geom}, Flap geometry = {flap_geom}")

    coords_half_length = int(len(coords)/2)

    coords_x = coords[:, 0]
    coords_y = coords[:, 1]

    coords_x_upr = coords_x[coords_y >= 0]
    coords_x_lwr = coords_x[coords_y < 0]

    #region # ==== GENERATE FLAP COORDINATES ==== #

    # Determine Point-A by extrapolating from nearest point on aerofoil
    B_x = flap_geom["Bx"] * chord
    A_m, A_c = slope_eq_at_idx(coords, closest_idx(coords_x_upr, B_x))
    A = np.array([
        B_x,
        A_m*B_x + A_c,
        0
    ])
    
    # Determine Point-C
    C_x = flap_geom["Cx"] * chord
    C_idx = closest_idx(coords_x_upr, C_x)
    C_m, C_c = slope_eq_at_idx(coords, C_idx)
    C = np.array([
        C_x,
        C_m*C_x + C_c,
        0
    ])

    # Determine Point-D by extrapolating from nearest point on aerofoil
    D_m, D_c = slope_eq_at_idx(coords, closest_idx(coords_x_lwr, B_x) + coords_half_length)
    D = np.array([
        B_x,
        D_m*B_x + D_c,
        0
    ])

    # Determine Point-B
    B = np.array([
        B_x, 
        D[1] + flap_geom["By"] * (A[1] - D[1]), 
        0
    ])

    # Determine Point-E
    E = coords[-1] # Can be ignored as is not needed

    # Determine Point-F
    F_x = flap_geom["Fx"] * chord
    F_idx = closest_idx(coords_x_lwr, F_x) + coords_half_length
    F_m, F_c = slope_eq_at_idx(coords, F_idx)
    F = np.array([
        F_x,
        F_m*F_x + F_c,
        0
    ])

    # Determine Point-G
    G_x = D[0] + flap_geom["Gx"] * (F[0] - D[0])
    G_m, G_c = slope_eq_at_idx(coords, closest_idx(coords_x_lwr, G_x) + coords_half_length)
    G = np.array([
        G_x,
        G_m*G_x + G_c,
        0
    ])

    # Determine Point-L
    L = np.array([
        B[0],
        B[1] + flap_geom["Ly"] * (D[1] - B[1]),
        0
    ])

    # Determine Point-M
    M = np.array([
        B[0],
        B[1] + flap_geom["My"] * (A[1] - B[1]),
        0
    ])

    # Determine Point-N
    N_x = A[0] + flap_geom["Nx"] * (C[0] - A[0])
    N_m, N_c = slope_eq_at_idx(coords, closest_idx(coords_x_upr, N_x))
    N = np.array([
        N_x,
        N_m*N_x + N_c,
        0
    ])

    # Create array of t values used to create bezier curve
    t = np.linspace(0, 1, num_t)

    # Create Bezier curve subtending points B, M, N, C and points B, L, G, F
    bezier_BMNC = np.zeros(shape=(len(t), 3))
    bezier_BLGF = np.zeros(shape=(len(t), 3))
    for i, ti in enumerate(t):
        bezier_BMNC[i] = cubic_bezier(ti, B, M, N, C)
        bezier_BLGF[i] = cubic_bezier(ti, B, L, G, F)

    # Determine Point-P
    P_x = B[0] + flap_geom["Px"] * (C[0] - B[0])
    P_idx = closest_idx(bezier_BMNC[:, 0], P_x)
    P_m, P_c = slope_eq_at_idx(bezier_BMNC, P_idx)
    P = np.array([
        P_x,
        P_m*P_x + P_c,
        0
    ])

    # Determine Point-S
    S_x = flap_geom["Sx"] * chord
    S_idx = closest_idx(coords_x_lwr, S_x) + coords_half_length
    S_m, S_c = slope_eq_at_idx(coords, S_idx)
    S = np.array([
        S_x,
        S_m*S_x + S_c,
        0
    ])
 
    # Determine Point-P1
    P1_x = S[0] + flap_geom["P1x"] * (P[0] - S[0])
    P_m, P_c = slope_eq_at_idx(bezier_BMNC, closest_idx(bezier_BMNC[:, 0], P[0]))
    P1 = np.array([
        P1_x,
        P_m*P1_x + P_c,
        0
    ])

    # Determine Point-S1
    S1_x = S[0] + flap_geom["S1x"] * (P[0] - S[0])
    S_m, S_c = slope_eq_at_idx(coords, closest_idx(coords_x_lwr, S[0]) + coords_half_length)
    S1 = np.array([
        S1_x,
        S_m*S1_x + S_c,
        0
    ])
    
    # Create Bezier curve subtending points S, S1, P1, P
    bezier_SS1P1P = np.zeros(shape=(len(t), 3))
    for i, ti in enumerate(t):
        bezier_SS1P1P[i] = cubic_bezier(ti, S, S1, P1, P)

    #endregion

    #region # ==== GENERATE SLAT COORDINATES ==== #

    T_x = slat_geom["Tx"] * chord
    T_idx = closest_idx(coords_x_upr, T_x)
    T_m, T_c = slope_eq_at_idx(coords, T_idx)
    T = np.array([
        T_x,
        T_m*T_x + T_c,
        0
    ])

    U_x = slat_geom["Ux"] * T_x
    U_idx = closest_idx(coords_x_lwr, U_x) + coords_half_length
    U_m, U_c = slope_eq_at_idx(coords, U_idx)
    U = np.array([
        U_x,
        U_m*U_x + U_c,
        0
    ])

    V_x = slat_geom["Vx"] * T_x
    V_m, V_c = slope_eq_at_idx(coords, closest_idx(coords_x_upr, V_x))
    V = np.array([
        V_x,
        V_m*V_x + V_c,
        0
    ])

    W_x = slat_geom["Wx"] * U_x
    W_m, W_c = slope_eq_at_idx(coords, closest_idx(coords_x_lwr, W_x) + coords_half_length)
    W = np.array([
        W_x,
        W_m*W_x + W_c,
        0
    ])

    slat_leading_edge = coords[T_idx+1:U_idx+1]

    # Create Bezier curve subtending points T, V, W, U
    t = np.linspace(0, 1, len(slat_leading_edge))
    bezier_TVWU = np.zeros(shape=(len(slat_leading_edge), 3))
    for i, ti in enumerate(t):
        bezier_TVWU[i] = cubic_bezier(ti, T, V, W, U)

    #endregion

    #region # ==== SEPARATE COORDINATES TO ELEMENTS ==== #

    flap_coords = np.concat((coords[:C_idx], bezier_BMNC[::-1][:-1], bezier_BLGF, coords[F_idx+1:]))

    slat_coords = np.concat((slat_leading_edge, bezier_TVWU[::-1]))

    aerofoil_coords = np.concat((coords[C_idx:S_idx], bezier_SS1P1P, bezier_BMNC[P_idx:][1:]))

    #endregion

    #region # ==== DETERMINE ELEMENTS DEPLOYMENT CHARACTERISTICS ==== #

    # In order to establish an appropriate gap between the slat and the base aerofoil, a new point is found
    # where the trailing edge of the slat (that has been moved to its x-overlap position) projects onto the surface
    # of the base aerofoil. The y-gap value is then added to the height of the aerofoil at this point to determine the gap.
    T_prime = T + np.array([slat_geom["x_overlap"]*chord - T[0], 0, 0])
    Q_idx = closest_idx(aerofoil_coords[:, 0], T_prime[0])
    Q_m, Q_c = slope_eq_at_idx(aerofoil_coords, Q_idx)
    
    Q_normal_theta = math.atan(-1 / Q_m)

    Q_prime = np.array([
        T_prime[0] - slat_geom["y_gap"]*chord * math.cos(Q_normal_theta),
        Q_m*T_prime[0] + Q_c - slat_geom["y_gap"]*chord * math.sin(Q_normal_theta),
        0
    ])

    C_prime = aerofoil_coords[0]

    slat_offset = Q_prime - T
    flap_offset = np.array([
        C_prime[0] - B[0] - flap_geom["x_overlap"]*chord,
        0,
        0
    ])

    #endregion

    #region # ==== APPLY TRANSFORMATIONS ==== #

    slat_coords = rotate_element(slat_coords, slat_geom["deflection"], T)
    flap_coords = rotate_element(flap_coords, flap_geom["deflection"], B)

    slat_coords = translate_element(slat_coords, slat_offset)
    flap_coords = translate_element(flap_coords, flap_offset)

    flap_upr_le = rotate_element(bezier_BMNC, flap_geom["deflection"], B)
    flap_upr_le = translate_element(flap_upr_le, flap_offset)

    Z_idx = closest_idx(flap_upr_le[:, 0], C_prime[0])
    Z_y = flap_upr_le[Z_idx][1]

    flap_offset = np.array([
        0,
        (C_prime[1] - Z_y) - flap_geom["y_gap"]*chord,
        0
    ])

    flap_coords = translate_element(flap_coords, flap_offset)

    #endregion

    return aerofoil_coords, slat_coords, flap_coords, len(slat_leading_edge)

def rotate_element(coords: np.ndarray, aoa: float, pivot: np.ndarray = np.zeros(shape=())):
    aoa_radians = math.radians(aoa)
    rotation_matrix = [
        [math.cos(aoa_radians), math.sin(aoa_radians), 0],
        [-math.sin(aoa_radians), math.cos(aoa_radians), 0],
        [0, 0, 0]
    ]

    if pivot.shape == ():
        pivot = np.zeros(shape=coords.shape)

    new_coords = coords - pivot

    new_coords = np.dot(new_coords, rotation_matrix)

    new_coords += pivot
    
    return new_coords

def translate_element(coords: np.ndarray, offset: np.ndarray):
    return coords + offset
    