import math
import numpy as np

def yt(yh: float, G: float, N: int):
	return yh * (1 - G**N) / (1 - G)

def bl_height(x:float, Re:float) -> float:
	return 0.37 * (x / Re**0.2)

def boundary_layer_characteristics(Re, H, h0, chord):
	δ99 = bl_height(Re, chord)
	flag_h = False
	flag_δ = False
	G = 1.21 #First guess G
	N_levels = 0
	while flag_h == False or flag_δ == False:
		flag_h = False
		flag_δ = False

		if G > 1.1:
			G = G - 0.01
		else:
			h0 = h0 - 0.1 * h0

		if yt(h0, G, 25) < δ99:
			flag_δ = True

		res = []
		Nx = []

		for N in range(30, 151, 1):
			Nx.append(N)
			h = yt(h0, G, N)
			res.append(H - h)

		abs_res = np.abs(res)
		idx = np.where(abs_res == np.min(abs_res))[0][0]

		N_guess = Nx[idx]
		H_guess = yt(h0, G, N_guess)

		if (H_guess >= H - 0.01 or H_guess <= H + 0.01) and flag_δ == True:
			N_levels = N_guess
			flag_h = True

	H_levels = yt(h0, G, N_levels)

	return H_levels, N_levels, G, h0

def calc_reynolds(x:float, U:float, nu:float) -> float:
	return (U * x) / nu
	
def calc_freestream_velocity(x:float, Re:float, nu:float) -> float:
	return (Re * nu) / (x)

def calc_bl_height(x:float, Re:float) -> float:
	return 0.37 * (x / np.pow(Re, 0.2))

def calc_wall_spacing(Re, yplus, dynamic_viscosity, freestream_velocity, freestream_density):
	ut = calc_friction_velocity(Re, freestream_velocity, freestream_density)
	return (yplus * dynamic_viscosity) / (freestream_density * ut)

def calc_skin_friction(Re):
	return 0.0576 * Re**-0.2

def calc_friction_velocity(Re, freestream_velocity, freestream_density):
	Cf = calc_skin_friction(Re)
	wall_stress = 0.5 * Cf * freestream_density * freestream_velocity**2
	return np.sqrt(wall_stress / freestream_density)

# def calc_bl_height(Re, yplus, freestream_velocity, freestream_density, viscosity):
#     ut = calc_friction_velocity(Re, freestream_velocity, freestream_density)
#     return (yplus * viscosity) / ut

def refinement_parameters(chord:float, Re:float, h0:float=-1):
	if Re < 0 and h0 < 0:
		return 0.35, 100, 1.12
	else:
		H = 0.35 * chord
		if h0 < 0:
			h0 = chord * np.sqrt(74) * np.pow(Re, -13 / 14)
		
		H_levels, N_levels, G, h0 = boundary_layer_characteristics(Re, H, h0, chord)

		return H_levels, N_levels, G, h0
		
def boundary_layer_stats(Re:float, U:float, rho:float, nu:float, yplus:float):
	ut = calc_friction_velocity(Re, U, rho)
	return (yplus * nu) / (rho * ut)
	
def calc_first_cell_height(Re:float, U:float, nu:float, yplus:float):
	mu = 1.81e-5
	
	rho = mu / nu
	
	cf = math.pow(2*math.log10(Re) - 0.65, -2.3)
	
	tau_w = 0.5 * rho * math.pow(U, 2) * cf
	
	ut = math.pow(tau_w / rho, 0.5)
	
	yh = (2 * yplus * nu) / (ut)
	
	return yh