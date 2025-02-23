# Python Script, API Version = V251
from datetime import datetime
import glob
import json
import logging
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

geometry_logs_folder = os.path.join(parameters["i/o"]["logs_folder"], "geometry")

# Create log folder if it doesn't exist already
if not os.path.exists(geometry_logs_folder):
    os.mkdir(geometry_logs_folder)

# Initialise logging
log_filename = os.path.join(geometry_logs_folder,
                            datetime.now().strftime("log_%Y-%m-%d_%H-%M-%S.log"))

# Configure logging
logging.basicConfig(filename=log_filename, level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()
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
    # Delete Selection
    selection = Selection.Create([GetRootPart().DatumPlanes[0].Curves[0],
        GetRootPart().Bodies[0],
        GetRootPart().Bodies[0].Edges[-1],
        GetRootPart().Bodies[0].Edges[-2],
        GetRootPart().Bodies[0].Edges[-3],
        GetRootPart().Bodies[0].Edges[-4],
        GetRootPart().DatumPlanes[0].Curves[2],
        GetRootPart().DatumPlanes[0].Curves[1]])
    result = Delete.Execute(selection)
    # EndBlock

    # Delete Objects
    selection = Selection.Create(GetRootPart().DatumPlanes[0])
    result = Delete.Execute(selection)
    # EndBlock


# Delete all objects if they already exist
if len(GetRootPart().Bodies) > 1:
    delete_objects()

for i, aerofoil in enumerate(aerofoils):
    progress = "[{0}/{1}]".format(i+1, len(aerofoils))  
    logging.info("{0} : {1} | Importing aerofoils...".format(progress, aerofoil[0]))
    # Coordinate filepaths
    base_filepath = "{0}.txt".format(os.path.join(aerofoil[1], aerofoil[2][0]))
    flap_filepath = "{0}.txt".format(os.path.join(aerofoil[1], aerofoil[2][1]))
    slat_filepath = "{0}.txt".format(os.path.join(aerofoil[1], aerofoil[2][2]))
    
    # Insert From File
    importOptions = ImportOptions.Create()
    DocumentInsert.Execute(base_filepath, importOptions, GetMaps("406d751d"))
    # EndBlock

    # Insert From File
    importOptions = ImportOptions.Create()
    DocumentInsert.Execute(flap_filepath, importOptions, GetMaps("74f728c0"))
    # EndBlock

    # Insert From File
    importOptions = ImportOptions.Create()
    DocumentInsert.Execute(slat_filepath, importOptions, GetMaps("5f814500"))
    # EndBlock
    
    logging.info("{0} : {1} | Import complete.".format(progress, aerofoil[0]))
    logging.info("{0} : {1} | Creating aerofoil named sections...".format(progress, aerofoil[0]))
    
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
    
    logging.info("{0} : {1} | Named sections created.".format(progress, aerofoil[0]))
    logging.info("{0} : {1} | Creating C-Domain...".format(progress, aerofoil[0]))
    
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
    
    # Fill
    selection = Selection.Create([GetRootPart().DatumPlanes[0].Curves[0],
        GetRootPart().DatumPlanes[0].Curves[1],
        GetRootPart().DatumPlanes[0].Curves[2],
        GetRootPart().DatumPlanes[0].Curves[3],
        GetRootPart().DatumPlanes[0].Curves[4],
        GetRootPart().DatumPlanes[0].Curves[5],
        GetRootPart().DatumPlanes[0].Curves[6]])
    secondarySelection = Selection.Empty()
    options = FillOptions()
    result = Fill.Execute(selection, secondarySelection, options, FillMode.Layout, None)
    # EndBlock
    
   # Solidify Sketch
    mode = InteractionMode.Solid
    result = ViewHelper.SetViewMode(mode, None)
    # EndBlock   
    
    # Delete Selection
    selection = FaceSelection.Create(GetRootPart().Bodies[0].Faces[2])
    result = Delete.Execute(selection)
    # EndBlock

    # Delete Selection
    selection = FaceSelection.Create(GetRootPart().Bodies[1].Faces[0])
    result = Delete.Execute(selection)
    # EndBlock

    # Delete Selection
    selection = FaceSelection.Create(GetRootPart().Bodies[0].Faces[1])
    result = Delete.Execute(selection)
    # EndBlock

    # Delete Selection
    selection = FaceSelection.Create(GetRootPart().Bodies[0].Faces[0])
    result = Delete.Execute(selection)
    # EndBlock

    #endregion
    
    logging.info("{0} : {1} | C-Domain created.".format(progress, aerofoil[0]))

    #region # ==== CREATE AEROFOIL GEOMETRY ==== #
    
    logger.info("{0} : {1} | Creating boundary named sections...".format(progress, aerofoil[0]))
    
    #endregion

    #region # ==== CREATE NAMED SECTIONS ==== #

    # Create Named Selection
    primarySelection = EdgeSelection.Create(GetRootPart().Bodies[0].Edges[-4])
    secondarySelection = Selection.Empty()
    result = NamedSelection.Create(primarySelection, secondarySelection, "Inlet")
    # EndBlock

    # Create Named Selection
    primarySelection = EdgeSelection.Create([GetRootPart().Bodies[0].Edges[-1],
        GetRootPart().Bodies[0].Edges[-3]])
    secondarySelection = Selection.Empty()
    result = NamedSelection.Create(primarySelection, secondarySelection, "Symmetry")
    # EndBlock

    # Create Named Selection
    primarySelection = EdgeSelection.Create(GetRootPart().Bodies[0].Edges[-2])
    secondarySelection = Selection.Empty()
    result = NamedSelection.Create(primarySelection, secondarySelection, "Outlet")
    # EndBlock
    
    # Create Named Selection
    primarySelection = BodySelection.Create(GetRootPart().Bodies[0])
    secondarySelection = Selection.Empty()
    result = NamedSelection.Create(primarySelection, secondarySelection, "Fluid")
    # EndBlock
    
    #endregion
    
    logger.info("{0} : {1} | Boundary named sections created.".format(progress, aerofoil[0]))
    
    logger.info("{0} : {1} | Exporting geometry file...".format(progress, aerofoil[0]))

    # Save File to export location with the aerofoil's name
    options = ExportOptions.Create()
    DocumentSave.Execute(r"{0}.scdocx".format(os.path.join(geometries_folder, aerofoil[0])), options)
    # EndBlock
    
    logger.info("{0} : {1} | Export successful.".format(progress, aerofoil[0]))
    
    delete_objects()