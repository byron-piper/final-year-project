import math
import os

import numpy as np
from shapely.geometry import Polygon

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
	
	# camber_x = np.concat((dyc_dx[::-1], dyc_dx[1:-1]))
	# camber_y = np.concat((yc[::-1], yc[1:-1]))
	# camber_line = np.column_stack((camber_x, camber_y))

	return coords

def generate_slat_coords(chord:float, coords:np.ndarray, slat_geom:dict) -> tuple[np.ndarray, np.ndarray]:
	leading_edge_index = np.argmin(coords[:, 0])

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
	U_idx = closest_point_idx(coords_x_lwr, U_x) + leading_edge_index
	U_m, U_c = slope_eq_at_idx(coords, U_idx)
	U = np.array([U_x, U_m*U_x + U_c, 0])

	# V_x is the control point for the slat bezier curve extrending from T_x
	V_x = slat_geom["Vx"] * T_x
	V_m, V_c = slope_eq_at_idx(coords, closest_point_idx(coords_x_upr, V_x))
	V = np.array([V_x, V_m*V_x + V_c, 0])

	# W_x is the control point for the slat bezier curve extrending from U_x
	W_x = slat_geom["Wx"] * U_x
	W_m, W_c = slope_eq_at_idx(coords, closest_point_idx(coords_x_lwr, W_x) + leading_edge_index)
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
	delta_y = (slat_geom["y_offset"] * chord - (T_prime[1] - (T_m_prime*T_prime[0] + T_c_prime))) * math.sin(T_prime_theta)
	slat_coords += np.array((0, delta_y, 0))

	slat_coords = remove_points_within_radius(slat_coords, 0, 0.0075)
	slat_coords = remove_points_within_radius(slat_coords, len(slat_leading_edge)-3, 0.01)

	slat_coords = np.concat((slat_coords, [slat_coords[0]]))

	return slat_coords, coords_wo_slat

def generate_flap_coords(chord:float, coords:np.ndarray, flap_geom:dict) -> tuple[np.ndarray, dict]:
	leading_edge_idx = np.argmin(coords[:, 0])

	coords_x = coords[:, 0]
	coords_y = coords[:, 1]
 
	coords_x_upr = coords_x[:leading_edge_idx+1]
	coords_x_lwr = coords_x[leading_edge_idx:]

	# Determine Point-A by extrapolating from nearest point on aerofoil
	B_x = flap_geom["Bx"] * chord
	A_m, A_c = slope_eq_at_idx(coords, closest_point_idx(coords_x_upr, B_x))
	A = np.array([B_x, A_m*B_x + A_c, 0])
	
	# Determine Point-C
	C_x = B_x + flap_geom["Cx"] * (1 - B_x)
	C_idx = closest_point_idx(coords_x_upr, C_x)
	C_m, C_c = slope_eq_at_idx(coords, C_idx)
	C = np.array([C_x, C_m*C_x + C_c, 0])

	# Determine Point-D by extrapolating from nearest point on aerofoil
	D_m, D_c = slope_eq_at_idx(coords, closest_point_idx(coords_x_lwr, B_x) + leading_edge_idx)
	D = np.array([B_x, D_m*B_x + D_c, 0])

	# Determine Point-B
	B = np.array([B_x, D[1] + flap_geom["By"] * (A[1] - D[1]), 0])

	# Determine Point-E
	E = coords[-1] # Can be ignored as is not needed

	# Determine Point-F
	F_x = B_x + flap_geom["Fx"] * (1 - B_x)
	F_idx = closest_point_idx(coords_x_lwr, F_x) + leading_edge_idx
	F_m, F_c = slope_eq_at_idx(coords, F_idx)
	F = np.array([F_x, F_m*F_x + F_c, 0])

	# Determine Point-G
	G_x = D[0] + flap_geom["Gx"] * (F[0] - D[0])
	G_m, G_c = slope_eq_at_idx(coords, closest_point_idx(coords_x_lwr, G_x) + leading_edge_idx)
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
	S_idx = closest_point_idx(coords_x_lwr, S_x) + len(coords_x_upr)
	S_m, S_c = slope_eq_at_idx(coords, S_idx)
	S = np.array([S_x, S_m*S_x + S_c, 0])

	# Determine Point-P1
	P1_x = S[0] + flap_geom["P1x"] * (P[0] - S[0])
	P_m, P_c = slope_eq_at_idx(bezier_BMNC, closest_point_idx(bezier_BMNC[:, 0], P[0]))
	P1 = np.array([P1_x, P_m*P1_x + P_c, 0])

	# Determine Point-S1
	S1_x = S[0] + flap_geom["S1x"] * (P[0] - S[0])
	S_m, S_c = slope_eq_at_idx(coords, closest_point_idx(coords_x_lwr, S[0]) + leading_edge_idx)
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

	coords_wo_flap = remove_points_within_radius(coords_wo_flap, 0, 0.05)
	flap_coords = remove_points_within_radius(flap_coords, 0, 0.005)

	# APPLY TRANSFORMATIONS

	flap_coords = rotate_coords(flap_coords, -flap_geom["deflection"], B)

	# Update C_x as coordinates were removed
	C_x = np.max(coords_wo_flap[:, 0])

	delta_x = -B_x + C_x - flap_geom["x_offset"]
	flap_coords += np.array((delta_x, 0, 0))

	# Update C as coordinates were removed
	C = coords_wo_flap[np.argmax(coords_wo_flap[:, 0])]

	C_m_prime, C_c_prime = slope_eq_at_idx(flap_coords, closest_point_idx(flap_coords, C))
	C_prime_theta = math.atan(-1 / C_m_prime)
	delta_y = (flap_geom["y_offset"] * chord - (C[1] - (C_m_prime*C[0] + C_c_prime))) * math.sin(C_prime_theta)
	flap_coords += np.array((0, delta_y, 0))

	coords_wo_flap = np.concat((coords_wo_flap, [coords_wo_flap[0]]))
	flap_coords = np.concat((flap_coords, [flap_coords[0]]))

	return flap_coords, coords_wo_flap

def normalise_aerofoil_coords(coords:list[np.ndarray], trailing_edge_x:float) -> list[np.ndarray]:
	combined_coords = np.concat(coords)
	
	max_x = np.max(combined_coords[:, 0])
	mean_y = np.mean(combined_coords[:, 1])
	
	normalised_coords = []
	for coord in coords:
		coord[:, 0] += trailing_edge_x - max_x
		coord[:, 1] -= mean_y
		normalised_coords.append(coord)
		
	return normalised_coords

def save_aerofoil(aerofoil_id:str, aerofoil_coords:list, directory:str):
	aerofoil_coords[0] = [f"1\t{row[0]}\t{row[1]}\n" for row in aerofoil_coords[0]]
	aerofoil_coords[1] = [f"1\t{row[0]}\t{row[1]}\n" for row in aerofoil_coords[1]]
	aerofoil_coords[2] = [f"1\t{row[0]}\t{row[1]}\n" for row in aerofoil_coords[2]]

	# Intialise folder path and create directory if it doesn't exist already
	aerofoil_folder = os.path.join(directory, aerofoil_id)
	if not os.path.exists(aerofoil_folder):
		os.mkdir(aerofoil_folder)

	for i, element in enumerate(["base", "flap", "slat"]):
		filepath = os.path.join(aerofoil_folder, f"{element}.txt")
		with open(filepath, "w") as f:
			f.write("polyline=true\n")
			f.writelines(aerofoil_coords[i])

def remove_points_within_radius(airfoil, idx, radius):
    coords = np.array(airfoil)
    
    distances_sq = np.sum((coords - coords[idx]) ** 2, axis=1)

    mask = distances_sq > radius**2

    return coords[mask]

def aerofoil_elements_intersecting(base_coords:np.ndarray, flap_coords:np.ndarray, slat_coords:np.ndarray):
	base_polygon = Polygon(base_coords)
	flap_polygon = Polygon(flap_coords)
	slat_polygon = Polygon(slat_coords)

	base_intersects_flap = base_polygon.intersects(flap_polygon)
	base_intersects_slat = base_polygon.intersects(slat_polygon)
	slat_intersects_flap = slat_polygon.intersects(flap_polygon)

	return base_intersects_flap or base_intersects_slat or slat_intersects_flap

def surface_normal(coords:np.ndarray, x_offset:float, y_offset:float):
	coords_idx = closest_point_idx(coords, x_offset)

	# Get surface tangents at each point along the aerofoil element
	tangents = np.diff(coords[coords_idx:coords_idx+2], axis=0)
	tangents = tangents / np.linalg.norm(tangents, axis=1, keepdims=True)

	z_vector = np.array((0, 0, 1))

	normals = np.cross(tangents, z_vector)
    #normals /= np.linalg.norm(normals)

	normals = coords[coords_idx:coords_idx+2] + normals * y_offset

	return normals