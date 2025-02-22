import math

import numpy as np

from helper import slope_eq_at_idx, closest_point_idx, cubic_bezier, rotate_coords

def generate_naca4(chord:float=1, M:int=2, P:int=4, T:int=12, num_points:int=100, closed_trailing_edge:bool=True, cosine_spacing:bool=True) -> np.ndarray:

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

	x *= chord

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

def generate_slat_coords(chord:float, coords:np.ndarray, slat_geom:dict) -> tuple[np.ndarray, np.ndarray]:
	coords_half_length = len(coords[coords[:, 1] >= 0])

	coords_x = coords[:, 0]
	coords_y = coords[:, 1]

	coords_x_upr = coords_x[coords_y >= 0]
	coords_x_lwr = coords_x[coords_y <= 0]

	# T_x is the upper anchor point on the aerofoil for slat curve
	T_x = slat_geom["Tx"] * chord
	T_idx = closest_point_idx(coords_x_upr, T_x)
	T_m, T_c = slope_eq_at_idx(coords, T_idx)
	T = np.array([T_x, T_m*T_x + T_c, 0])

	# U_x is the lower anchor point on the aerofoil for slat curve
	U_x = slat_geom["Ux"] * chord
	U_idx = closest_point_idx(coords_x_lwr, U_x) + coords_half_length
	U_m, U_c = slope_eq_at_idx(coords, U_idx)
	U = np.array([U_x, U_m*U_x + U_c, 0])

	# V_x is the control point for the slat bezier curve extrending from T_x
	V_x = slat_geom["Vx"] * T_x
	V_m, V_c = slope_eq_at_idx(coords, closest_point_idx(coords_x_upr, V_x))
	V = np.array([V_x, V_m*V_x + V_c, 0])

	# W_x is the control point for the slat bezier curve extrending from U_x
	W_x = slat_geom["Wx"] * U_x
	W_m, W_c = slope_eq_at_idx(coords, closest_point_idx(coords_x_lwr, W_x) + coords_half_length)
	W = np.array([W_x, W_m*W_x + W_c, 0])

	# Create new numpy array of points of slat leading edge
	slat_leading_edge = coords[T_idx:U_idx+1]

	# Create Bezier curve subtending points T, V, W, U
	t = np.linspace(0, 1, len(slat_leading_edge))
	bezier_TVWU = np.zeros(shape=(len(slat_leading_edge), 3))
	for i, ti in enumerate(t):
		bezier_TVWU[i] = cubic_bezier(ti, T, V, W, U)

	if np.cross(slat_leading_edge[-1] - slat_leading_edge[-2], bezier_TVWU[-1] - slat_leading_edge[-1])[2] < 0:
		temp = bezier_TVWU[:-1]
		slat_coords = np.concat((slat_leading_edge, temp[::-1]))
	else:
		slat_coords = np.concat((slat_leading_edge, bezier_TVWU[::-1]))

	coords_wo_slat = np.concat((coords[:T_idx+1], bezier_TVWU, coords[U_idx:]))

	# APPLY TRANSFORMATIONS

	slat_coords = rotate_coords(slat_coords, slat_geom["deflection"], T)
	
	# Shift slat coordinates to given slat x offset
	aerofoil_le_x = np.min(bezier_TVWU, axis=0)[0]
	delta_x = -np.max(slat_coords, axis=0)[0] + aerofoil_le_x + slat_geom["x_offset"] * chord
	slat_coords += np.array((delta_x, 0, 0))

	# T_prime is defined as the position T now exists at given the previous two transformations
	T_prime = slat_coords[-1]
	T_m_prime, T_c_prime = slope_eq_at_idx(bezier_TVWU, closest_point_idx(bezier_TVWU[bezier_TVWU[:, 1] > 0], T_prime))
	T_prime_theta = math.atan(1 / T_m_prime)
	delta_y = (slat_geom["slot_gap"] * chord - (T_prime[1] - (T_m_prime*T_prime[0] + T_c_prime))) * math.sin(T_prime_theta)
	slat_coords += np.array((0, delta_y, 0))

	return slat_coords, coords_wo_slat, W

def generate_flap_coords(chord:float, coords:np.ndarray, flap_geom:dict) -> tuple[np.ndarray, dict]:
	coords_half_length = len(coords[coords[:, 1] >= 0])

	coords_x = coords[:, 0]
	coords_y = coords[:, 1]

	coords_x_upr = coords_x[coords_y >= 0]
	coords_x_lwr = coords_x[coords_y <= 0]

	coords_half_length = len(coords_x_upr)

	# Determine Point-A by extrapolating from nearest point on aerofoil
	B_x = flap_geom["Bx"] * chord
	A_m, A_c = slope_eq_at_idx(coords, closest_point_idx(coords_x_upr, B_x))
	A = np.array([B_x, A_m*B_x + A_c, 0])
	
	# Determine Point-C
	C_x = flap_geom["Cx"] * chord
	C_idx = closest_point_idx(coords_x_upr, C_x)
	C_m, C_c = slope_eq_at_idx(coords, C_idx)
	C = np.array([C_x, C_m*C_x + C_c, 0])

	# Determine Point-D by extrapolating from nearest point on aerofoil
	D_m, D_c = slope_eq_at_idx(coords, closest_point_idx(coords_x_lwr, B_x) + coords_half_length)
	D = np.array([B_x, D_m*B_x + D_c, 0])

	# Determine Point-B
	B = np.array([B_x, D[1] + flap_geom["By"] * (A[1] - D[1]), 0])

	# Determine Point-E
	E = coords[-1] # Can be ignored as is not needed

	# Determine Point-F
	F_x = flap_geom["Fx"] * chord
	F_idx = closest_point_idx(coords_x_lwr, F_x) + coords_half_length
	F_m, F_c = slope_eq_at_idx(coords, F_idx)
	F = np.array([F_x, F_m*F_x + F_c, 0])

	# Determine Point-G
	G_x = D[0] + flap_geom["Gx"] * (F[0] - D[0])
	G_m, G_c = slope_eq_at_idx(coords, closest_point_idx(coords_x_lwr, G_x) + coords_half_length)
	G = np.array([G_x, G_m*G_x + G_c, 0])

	# Determine Point-L
	L = np.array([B[0], B[1] + flap_geom["Ly"] * (D[1] - B[1]), 0])

	# Determine Point-M
	M = np.array([B[0], B[1] + flap_geom["My"] * (A[1] - B[1]), 0])

	# Determine Point-N
	N_x = A[0] + flap_geom["Nx"] * (C[0] - A[0])
	N_m, N_c = slope_eq_at_idx(coords, closest_point_idx(coords_x_upr, N_x))
	N = np.array([N_x, N_m*N_x + N_c, 0])

	# Create array of t values used to create bezier curve
	t = np.linspace(0, 1, 25)

	# Create Bezier curve subtending points B, M, N, C and points B, L, G, F
	bezier_BMNC = np.zeros(shape=(len(t), 3))
	bezier_BLGF = np.zeros(shape=(len(t), 3))
	for i, ti in enumerate(t):
		bezier_BMNC[i] = cubic_bezier(ti, B, M, N, C)
		bezier_BLGF[i] = cubic_bezier(ti, B, L, G, F)

	# Determine Point-P
	P_x = B[0] + flap_geom["Px"] * (C[0] - B[0])
	P_idx = closest_point_idx(bezier_BMNC[:, 0], P_x)
	P_m, P_c = slope_eq_at_idx(bezier_BMNC, P_idx)
	P = np.array([P_x, P_m*P_x + P_c, 0])

	# Determine Point-S
	S_x = flap_geom["Sx"] * chord
	S_idx = closest_point_idx(coords_x_lwr, S_x) + coords_half_length
	S_m, S_c = slope_eq_at_idx(coords, S_idx)
	S = np.array([S_x, S_m*S_x + S_c, 0])

	# Determine Point-P1
	P1_x = S[0] + flap_geom["P1x"] * (P[0] - S[0])
	P_m, P_c = slope_eq_at_idx(bezier_BMNC, closest_point_idx(bezier_BMNC[:, 0], P[0]))
	P1 = np.array([P1_x, P_m*P1_x + P_c, 0])

	# Determine Point-S1
	S1_x = S[0] + flap_geom["S1x"] * (P[0] - S[0])
	S_m, S_c = slope_eq_at_idx(coords, closest_point_idx(coords_x_lwr, S[0]) + coords_half_length)
	S1 = np.array([S1_x, S_m*S1_x + S_c, 0])
	
	# Create Bezier curve subtending points S, S1, P1, P
	bezier_SS1P1P = np.zeros(shape=(len(t), 3))
	for i, ti in enumerate(t):
		bezier_SS1P1P[i] = cubic_bezier(ti, S, S1, P1, P)

	# Point F is created by determining a new point closest to some existing point by extrapolating along a straight line defined by nearest points on aerofoil
	# Therefore, F_idx does not guarantee a realistic index and may point to a point that should not exist on the curve. This simple if statement
	# accounts for this
	if F[0] > coords[F_idx][0]:
		F_idx += 1

	flap_coords = np.concat((coords[:C_idx], bezier_BMNC[::-1][:-1], bezier_BLGF, coords[F_idx:]))

	# coords_wo_flap = [
	#     coords[C_idx:S_idx+3],
	#     np.concat((bezier_SS1P1P, bezier_BMNC[P_idx:][1:]))
	# ]

	coords_wo_flap = np.concat((coords[C_idx:S_idx], bezier_SS1P1P, bezier_BMNC[P_idx:][1:]))

	# APPLY TRANSFORMATIONS

	flap_coords = rotate_coords(flap_coords, flap_geom["deflection"], B)

	delta_x = -B_x + C_x + flap_geom["x_offset"] * chord
	flap_coords += np.array((delta_x, 0, 0))

	C_m_prime, C_c_prime = slope_eq_at_idx(flap_coords, closest_point_idx(flap_coords, C))
	C_prime_theta = math.atan(-1 / C_m_prime)
	delta_y = (flap_geom["slot_gap"] * chord - (C[1] - (C_m_prime*C[0] + C_c_prime))) * math.sin(C_prime_theta)
	flap_coords += np.array((0, delta_y, 0))

	return flap_coords, coords_wo_flap
