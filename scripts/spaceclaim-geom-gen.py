import glob
import json
import os

# !! CHANGE DIRECTORY TO POINT TO CLONED REPO LOCATION !!
with open(r"E:\final-year-project\parameters.json", "r") as f:
	parameters = json.load(f)

# C-Domain geometry properties
domain_radius = parameters["geometry"]["domain_radius"]
domain_length = parameters["geometry"]["domain_length"]

# I/O
coords_folder = parameters["i/o"]["coords_folder"]
geometries_folder = parameters["i/o"]["geometries_folder"]

# Walk through each subfolder within the import location and look for three '.txt. files containing
# coordinates for each aerofoil element. Save the name of the folder and its absolute path
aerofoils = []
for folder in glob.glob(os.path.join(coords_folder, "*")):
	if os.path.isdir(folder):
		files = sorted(glob.glob(os.path.join(folder, "*.txt")))
		file_names = tuple(os.path.splitext(os.path.basename(f))[0] for f in files)
		
		# If three '.txt' files exist, add to aerofoil entry
		if len(file_names) == 3:  
			aerofoils.append((os.path.basename(folder), os.path.abspath(folder), file_names))
		else:
			print("Aerofoil '{0}' has invalid coordinate configuration!")

def delete_objects():
	# Delete Objects
	selection = Selection.Create(GetRootPart().Curves[2])
	result = Delete.Execute(selection)
	# EndBlock

	# Delete Objects
	selection = Selection.Create(GetRootPart().Curves[1])
	result = Delete.Execute(selection)
	# EndBlock

	# Delete Objects
	selection = Selection.Create(GetRootPart().Curves[0])
	result = Delete.Execute(selection)
	# EndBlock

	# Delete Objects
	selection = BodySelection.Create(GetRootPart().Bodies[0])
	result = Delete.Execute(selection)
	# EndBlock


# Delete all objects if they already exist
if len(GetRootPart().Bodies) > 1:
	delete_objects()

for aerofoil in aerofoils:
	#region # ==== CREATE C-DOMAIN ==== #

	# Set Sketch Plane
	sectionPlane = Plane.PlaneXY
	result = ViewHelper.SetSketchPlane(sectionPlane, None)
	# EndBlock

	# Sketch Rectangle
	point1 = Point2D.Create(M(1), M(domain_radius))
	point2 = Point2D.Create(M(1 + domain_length), M(domain_radius))
	point3 = Point2D.Create(M(1 + domain_length), M(-domain_radius))
	result = SketchRectangle.Create(point1, point2, point3)
	# EndBlock

	# Create Sweep Arc
	origin = Point2D.Create(M(1), M(0))
	start = Point2D.Create(M(1), M(domain_radius))
	end = Point2D.Create(M(1), M(-domain_radius))
	senseClockWise = False
	result = SketchArc.CreateSweepArc(origin, start, end, senseClockWise)

	baseSel = SelectionPoint.Create(GetRootPart().DatumPlanes[0].Curves[4].GetChildren[ICurvePoint]()[1])
	targetSel = SelectionPoint.Create(GetRootPart().DatumPlanes[0].Curves[2].GetChildren[ICurvePoint]()[1])

	result = Constraint.CreateCoincident(baseSel, targetSel)

	baseSel = SelectionPoint.Create(GetRootPart().DatumPlanes[0].Curves[4], 3.14159265358979)
	targetSel = SelectionPoint.Create(GetRootPart().DatumPlanes[0].Curves[2], domain_radius)

	result = Constraint.CreateTangent(baseSel, targetSel)

	baseSel = SelectionPoint.Create(GetRootPart().DatumPlanes[0].Curves[4].GetChildren[ICurvePoint]()[0])
	targetSel = SelectionPoint.Create(GetRootPart().DatumPlanes[0].Curves[3].GetChildren[ICurvePoint]()[1])

	result = Constraint.CreateCoincident(baseSel, targetSel)

	baseSel = SelectionPoint.Create(GetRootPart().DatumPlanes[0].Curves[4].GetChildren[ICurvePoint]()[1])
	targetSel = SelectionPoint.Create(GetRootPart().DatumPlanes[0].Curves[3].GetChildren[ICurvePoint]()[0])

	result = Constraint.CreateCoincident(baseSel, targetSel)
	# EndBlock
	
	# Delete Selection
	selection = Selection.Create(GetRootPart().DatumPlanes[0].Curves[3])
	result = Delete.Execute(selection)
	# EndBlock
	
   # Solidify Sketch
	mode = InteractionMode.Solid
	result = ViewHelper.SetViewMode(mode, None)
	# EndBlock

	#endregion

	#region # ==== CREATE AEROFOIL GEOMETRY ==== #
	
	# Coordinate filepaths
	base_filepath = "{0}.txt".format(os.path.join(aerofoil[1], aerofoil[2][0]))
	flap_filepath = "{0}.txt".format(os.path.join(aerofoil[1], aerofoil[2][1]))
	slat_filepath = "{0}.txt".format(os.path.join(aerofoil[1], aerofoil[2][2]))
	
	# Insert From Base Element File
	importOptions = ImportOptions.Create()
	DocumentInsert.Execute(base_filepath, importOptions, GetMaps("ac3e73e4"))
	# EndBlock
	
	# Insert From Flap Element File
	importOptions = ImportOptions.Create()
	DocumentInsert.Execute(flap_filepath, importOptions, GetMaps("ac3e73e4"))
	# EndBlock
	
	# Insert From Slat Element File
	importOptions = ImportOptions.Create()
	DocumentInsert.Execute(slat_filepath, importOptions, GetMaps("ac3e73e4"))
	# EndBlock

	# Fill Base Curve
	selection = Selection.Create(GetRootPart().Curves[0])
	secondarySelection = Selection.Empty()
	options = FillOptions()
	result = Fill.Execute(selection, secondarySelection, options, FillMode.ThreeD, None)
	# EndBlock
	
	# Intersecting bodies
	targets = BodySelection.Create(GetRootPart().Bodies[0])
	tools = BodySelection.Create(GetRootPart().Bodies[1])
	options = MakeSolidsOptions()
	options.KeepCutter = False
	options.SubtractFromTarget = True
	result = Combine.Intersect(targets, tools, options)
	# EndBlock

	# Delete Selection
	selection = FaceSelection.Create(GetRootPart().Bodies[0].Faces[1])
	result = Delete.Execute(selection)
	# EndBlock
	
	# Fill
	selection = Selection.Create(GetRootPart().Curves[1])
	secondarySelection = Selection.Empty()
	options = FillOptions()
	result = Fill.Execute(selection, secondarySelection, options, FillMode.ThreeD, None)
	# EndBlock
	
	# Intersecting bodies
	targets = BodySelection.Create(GetRootPart().Bodies[0])
	tools = BodySelection.Create(GetRootPart().Bodies[1])
	options = MakeSolidsOptions()
	options.KeepCutter = False
	options.SubtractFromTarget = True
	result = Combine.Intersect(targets, tools, options)
	# EndBlock

	# Delete Selection
	selection = FaceSelection.Create(GetRootPart().Bodies[0].Faces[1])
	result = Delete.Execute(selection)
	# EndBlock
	
	# Fill
	selection = Selection.Create(GetRootPart().Curves[2])
	secondarySelection = Selection.Empty()
	options = FillOptions()
	result = Fill.Execute(selection, secondarySelection, options, FillMode.ThreeD, None)
	# EndBlock
	
	# Intersecting bodies
	targets = BodySelection.Create(GetRootPart().Bodies[0])
	tools = BodySelection.Create(GetRootPart().Bodies[1])
	options = MakeSolidsOptions()
	options.KeepCutter = False
	options.SubtractFromTarget = True
	result = Combine.Intersect(targets, tools, options)
	# EndBlock

	# Delete Selection
	selection = FaceSelection.Create(GetRootPart().Bodies[0].Faces[1])
	result = Delete.Execute(selection)
	# EndBlock
	
	#endregion

	#region # ==== CREATE NAMED SECTIONS ==== #

	# Create Named Selection Group
	primarySelection = Selection.Create(GetRootPart().Curves[0])
	secondarySelection = Selection.Empty()
	result = NamedSelection.Create(primarySelection, secondarySelection, "Base")
	# EndBlock

	# Create Named Selection Group
	primarySelection = Selection.Create(GetRootPart().Curves[1])
	secondarySelection = Selection.Empty()
	result = NamedSelection.Create(primarySelection, secondarySelection, "Flap")
	# EndBlock

	# Create Named Selection Group
	primarySelection = Selection.Create(GetRootPart().Curves[2])
	secondarySelection = Selection.Empty()
	result = NamedSelection.Create(primarySelection, secondarySelection, "Slat")
	# EndBlock

	# Create Named Selection Group
	primarySelection = EdgeSelection.Create(GetRootPart().Bodies[0].Edges[0])
	secondarySelection = Selection.Empty()
	result = NamedSelection.Create(primarySelection, secondarySelection, "Inlet")
	# EndBlock

	# Create Named Selection Group
	primarySelection = EdgeSelection.Create([GetRootPart().Bodies[0].Edges[3],
		GetRootPart().Bodies[0].Edges[1]])
	secondarySelection = Selection.Empty()
	result = NamedSelection.Create(primarySelection, secondarySelection, "Symmetry")
	# EndBlock

	# Create Named Selection Group
	primarySelection = EdgeSelection.Create(GetRootPart().Bodies[0].Edges[2])
	secondarySelection = Selection.Empty()
	result = NamedSelection.Create(primarySelection, secondarySelection, "Outlet")
	# EndBlock

	# Create Named Selection Group
	primarySelection = FaceSelection.Create(GetRootPart().Bodies[0].Faces[0])
	secondarySelection = Selection.Empty()
	result = NamedSelection.Create(primarySelection, secondarySelection, "Fluid")
	# EndBlock
	#endregion

	# Save File to export location with the aerofoil's name
	options = ExportOptions.Create()
	DocumentSave.Execute(r"{0}.scdocx".format(os.path.join(geometries_folder, aerofoil[0])), options)
	# EndBlock

	#delete_objects()
