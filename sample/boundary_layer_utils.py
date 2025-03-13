import math

def calc_first_cell_height(Re:float, x:float, rho:float, mu:float, yplus:float):
	U = (Re * mu) / (rho * x)

	cf = 0.026 / (math.pow(Re, 1/7))
	
	tau_w = 0.5 * rho * math.pow(U, 2) * cf
	
	ut = math.pow(tau_w / rho, 0.5)
	
	yh = (yplus * mu) / (ut * rho)
	
	return yh