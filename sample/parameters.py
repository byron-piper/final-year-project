project_id = "dev"
log_enabled = True
debug_enabled = False
randomise = False

#region ==== AEROFOIL ==== #

naca = [2, 4, 12]
num_points = 200
chord = 1

#endregion

#region ==== HIGH LIFT DEVICES ==== #

slat_geom = {
    "Tx": 0.15,
    "Ux": 0.3,
    "Vx": 0.6,
    "Wx": 0.25,
    "x_overlap": 0.01,
    "y_gap": 0.027,
    "deflection": 20
}

flap_geom = {
    "Bx": 0.7,
    "By": 0.25,
    "Cx": 0.82,
    "Fx": 0.75,
    "My": 0.5,
    "Nx": 0.5,
    "Ly": 0.8,
    "Gx": 0.4,
    "Px": 0.25,
    "Sx": 0.65,
    "P1x": 0.3,
    "S1x": 0.25,
    "x_overlap": 0.02,
    "y_gap": 0.02,
    "deflection": -30
}

#endregion

#region ==== GMSHER ==== #

mesh_size = 0.2
domain_radius = 5
domain_extension = 15
AoA = 0
N_inlet = 30
refinement_offset = 0.35 * chord

#endregion

