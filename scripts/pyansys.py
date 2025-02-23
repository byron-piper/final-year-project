import ansys.fluent.core as pyfluent
import math

solver = pyfluent.launch_fluent(dimension=2, precision='double', processor_count=4, mode='solver', show_gui=True)

solver.settings.file.read_mesh(file_name = r"C:\Users\honey\Documents\PROJECT\meshes\NACA-0313-56566565202816-2207163230.msh.h5")

solver.settings.setup.models.viscous.model = 'spalart-allmaras'

inlet = solver.settings.setup.boundary_conditions.velocity_inlet['inlet']
inlet.momentum.velocity_specification_method = "Components"
v_x = 14 * math.cos(math.degrees(5))
v_y = 14 * math.sin(math.degrees(5))
inlet.momentum.velocity_components[0] = v_x
inlet.momentum.velocity_components[1] = v_y

solver.settings.setup.reference_values.velocity = math.sqrt(math.pow(v_x, 2) + math.pow(v_y, 2))
solver.settings.setup.reference_values.zone = "fluid"

solver.settings.solution.initialization.defaults['x-velocity'] = v_x
solver.settings.solution.initialization.defaults['y-velocity'] = v_y
solver.settings.solution.initialization.defaults['nut'] = 0.0001460735
solver.settings.solution.initialization.standard_initialize()

solver.settings.solution.monitor.residual.equations['continuity'].absolute_criteria = 1e-6
solver.settings.solution.monitor.residual.equations['x-velocity'].absolute_criteria = 1e-6
solver.settings.solution.monitor.residual.equations['y-velocity'].absolute_criteria = 1e-6
solver.settings.solution.monitor.residual.equations['nut'].absolute_criteria = 1e-6

solver.settings.solution.run_calculation.iter_count = 500
solver.settings.solution.run_calculation.calculate()

solver.settings.file.export.ascii(
    file_name=r"E:\final-year-project\data.csv",
    surface_name_list=["interior:fluid"],
    cell_func_domain=[
        "pressure-coefficient",
        "velocity-magnitude",
        "x-coordinate",
        "y-coordinate",
        "vorticity-mag"
    ]
)
