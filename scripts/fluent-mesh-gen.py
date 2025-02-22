from datetime import datetime
import glob
import json
import logging
import os
import shutil

# !! CHANGE DIRECTORY TO POINT TO CLONED REPO LOCATION !!
with open(r"E:\final-year-project\parameters.json", "r") as f:
	parameters = json.load(f)

if not os.getenv('FLUENT_PROD_DIR'):
	import ansys.fluent.core as pyfluent
	flglobals = pyfluent.setup_for_fluent(product_version="25.1.0", mode="meshing", dimension=3, precision="double", processor_count=4, ui_mode="gui", graphics_driver="dx11")
	globals().update(flglobals)

#region # ==== UNPACK PARAMETERS ==== #
# GLOBAL SIZING
cells_per_gap = parameters["mesh"]["global_sizing"]["cells_per_gap"]
max_element_size = parameters["mesh"]["global_sizing"]["max_element_size"]
min_element_size = parameters["mesh"]["global_sizing"]["min_element_size"]
global_growth_rate = parameters["mesh"]["global_sizing"]["global_growth_rate"]

# LOCAL SIZING
boi_cells_per_gap = parameters["mesh"]["local_sizing"]["boi_cells_per_gap"]
boi_curvature_normal_angle = parameters["mesh"]["local_sizing"]["boi_curvature_normal_angle"]
boi_execution = parameters["mesh"]["local_sizing"]["boi_execution"]
boi_growth_rate = parameters["mesh"]["local_sizing"]["boi_growth_rate"]
boi_size = parameters["mesh"]["local_sizing"]["boi_size"]

# BOUNDARY LAYERS
bl_num_layers = parameters["mesh"]["boundary_layers"]["bl_num_layers"]
bl_first_height = parameters["mesh"]["boundary_layers"]["bl_first_height"]
bl_max_height = parameters["mesh"]["boundary_layers"]["bl_max_height"]
bl_growth_rate = parameters["mesh"]["boundary_layers"]["bl_growth_rate"]

# I/O
geometries_folder = parameters["i/o"]["geometries_folder"]
meshes_folder = parameters["i/o"]["meshes_folder"]
mesh_logs_folder = os.path.join(parameters["i/o"]["logs_folder"], "mesh")

overwrite = parameters["mesh"]["overwrite"]

#endregion

# Create log folder if it doesn't exist already
if not os.path.exists(mesh_logs_folder):
	os.mkdir(mesh_logs_folder)

# Delete unnecessary files generated by Fluent
def cleanup_geometries_folder():
	for file in glob.glob(os.path.join(geometries_folder, "*")):
		# Clean up any folder created by Fluent
		if os.path.isdir(file):
			try:
				shutil.rmtree(file)
			except Exception as e:
				print(f"Error deleting {file}: {e}")
		# Clean up any miscellaneous files created by Fluent
		elif file.endswith(".scdocx"):
			try:
				os.remove(file)
			except Exception as e:
				print(f"Error deleting {file}: {e}")

def cleanup_meshes_folder():
	for file in glob.glob(os.path.join(meshes_folder, "*.msh.h5")):
		os.remove(file)

# Initialise logging
log_filename = os.path.join(mesh_logs_folder,
							datetime.now().strftime("log_%Y-%m-%d_%H-%M-%S.log"))

# Configure logging
logging.basicConfig(filename=log_filename, level=logging.INFO,
					format="%(asctime)s - %(levelname)s - %(message)s")

# Walk through each subfolder within the import location and look for three '.txt. files containing
# coordinates for each aerofoil element. Save the name of the folder and its absolute path
geometries = []
for file in glob.glob(os.path.join(geometries_folder, "*.scdocx")):
	geometries.append((file, os.path.basename(file).split('.')[0]))

# Delete previous mesh files to avoid overwrite message prompt
if overwrite:
	cleanup_meshes_folder()

for i, geometry in enumerate(geometries):
	geometry_path = geometry[0]
	mesh_name = geometry[1]
	mesh_export_path = os.path.join(meshes_folder, mesh_name)
	progress = "[{0}/{1}]".format(i+1, len(geometries))
	
	workflow.InitializeWorkflow(WorkflowType=r'2D Meshing')
	meshing.GlobalSettings.LengthUnit.set_state(r'm')
	meshing.GlobalSettings.AreaUnit.set_state(r'm^2')
	meshing.GlobalSettings.VolumeUnit.set_state(r'm^3')
	logging.info("{0} : {1} | Import geometry...".format(progress, mesh_name))
	start_mesh_generation = datetime.now()
	workflow.TaskObject['Load CAD Geometry'].Arguments.set_state({r'FileName': geometry_path, r'LengthUnit': r'm',})
	workflow.TaskObject['Load CAD Geometry'].Execute()
	logging.info("{0} : {1} | Geometry imported.".format(progress, mesh_name))
	logging.info("{0} : {1} | Updating regions...".format(progress, mesh_name))
	workflow.TaskObject['Update Regions'].Execute()
	logging.info("{0} : {1} | Regions updated.".format(progress, mesh_name))
	logging.info("{0} : {1} | Updating boundaries...".format(progress, mesh_name))
	workflow.TaskObject['Update Boundaries'].Arguments.set_state({r'BoundaryLabelList': [r'inlet'],r'BoundaryLabelTypeList': [r'velocity-inlet'],r'OldBoundaryLabelList': [r'inlet'],r'OldBoundaryLabelTypeList': [r'pressure-inlet'],})
	workflow.TaskObject['Update Boundaries'].Execute()
	logging.info("{0} : {1} | Boundaries updated.".format(progress, mesh_name))
	logging.info("{0} : {1} | Defining global sizing...".format(progress, mesh_name))
	workflow.TaskObject['Define Global Sizing'].Arguments.set_state({r'CellsPerGap': cells_per_gap,r'GrowthRate': global_growth_rate, r'MaxSize': max_element_size, r'MinSize': min_element_size,})
	workflow.TaskObject['Define Global Sizing'].Execute()
	logging.info("{0} : {1} | Global sizing defined.".format(progress, mesh_name))
	logging.info("{0} : {1} | Defining local sizing...".format(progress, mesh_name))
	workflow.TaskObject['Add Local Sizing'].Arguments.set_state({r'AddChild': r'yes',r'BOICellsPerGap': boi_cells_per_gap, r'BOICurvatureNormalAngle': boi_curvature_normal_angle, r'BOIExecution': boi_execution, r'BOIGrowthRate': boi_growth_rate, r'BOISize': boi_size, r'BOIZoneorLabel': r'label',r'EdgeLabelList': [r'part-boundary'],})
	workflow.TaskObject['Add Local Sizing'].AddChildAndUpdate(DeferUpdate=False)
	logging.info("{0} : {1} | Local sizing defined.".format(progress, mesh_name))
	logging.info("{0} : {1} | Adding boundary layers...".format(progress, mesh_name))
	workflow.TaskObject['Add 2D Boundary Layers'].Arguments.set_state({r'AddChild': r'yes',r'FirstLayerHeight': bl_first_height, r'MaxLayerHeight': bl_max_height, r'NumberOfLayers': bl_num_layers, r'Rate': bl_growth_rate,})
	workflow.TaskObject['Add 2D Boundary Layers'].AddChildAndUpdate(DeferUpdate=False)
	logging.info("{0} : {1} | Boundary layers added.".format(progress, mesh_name))
	logging.info("{0} : {1} | Generating surface mesh...".format(progress, mesh_name))
	workflow.TaskObject['Generate the Surface Mesh'].Arguments.set_state({r'EnableMultiThreading': True,r'ProjectOnGeometry': False,})
	workflow.TaskObject['Generate the Surface Mesh'].Execute()
	logging.info("{0} : {1} | Surface mesh updated.".format(progress, mesh_name))
	logging.info("{0} : {1} | Exporting mesh...".format(progress, mesh_name))
	workflow.TaskObject['Export Fluent 2D Mesh'].Arguments.set_state({r'FileName': '{0}.msh.h5'.format(mesh_export_path),})
	workflow.TaskObject['Export Fluent 2D Mesh'].Execute()
	logging.info("{0} : {1} | Mesh exported.".format(progress, mesh_name))
	end_mesh_generation = datetime.now()
	time_elapsed = end_mesh_generation - start_mesh_generation
	logging.info("{0} : {1} | Total time elapsed: {2}".format(progress, mesh_name, time_elapsed))