import math

import numpy as np

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

	print(leading_edge_index)