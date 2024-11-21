project_id = "dev"

#region ==== AEROFOIL PARAMETERS ==== #

naca = [2, 4, 12]
num_points = 100
chord = 1

#endregion

#region ==== GMSHER PARAMETERS ==== #

mesh_size = 0.2
domain_radius = 5
domain_extension = 15
AoA = 0
N_inlet = 30
refinement_offset = 0.35 * chord

#endregion

