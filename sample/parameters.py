project_id = "dev"
log_enabled = True
debug_enabled = False

#region ==== AEROFOIL ==== #

naca = [2, 4, 12]
num_points = 200
chord = 1

#endregion

#region ==== HIGH LIFT DEVICES ==== #

slat_geom = [0.15, 0.5, 0.5, 1]
flap_geom = [0.3, 0.5, 0.25, 0.1]

#endregion

#region ==== GMSHER ==== #

mesh_size = 0.2
domain_radius = 5
domain_extension = 15
AoA = 0
N_inlet = 30
refinement_offset = 0.35 * chord

#endregion

