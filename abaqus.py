# -*- coding: mbcs -*-
#
# Abaqus/CAE Release 2020 replay file
# Internal Version: 2019_09_13-20.49.31 163176
# Run by User on Tue Feb  7 21:39:07 2023
#

# from driverUtils import executeOnCaeGraphicsStartup
# executeOnCaeGraphicsStartup()
#: Executing "onCaeGraphicsStartup()" in the site directory ...
from abaqus import *
from abaqusConstants import *
session.Viewport(name='Viewport: 1', origin=(0.0, 0.0), width=183.33235168457, 
    height=127.565101623535)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].maximize()
from caeModules import *
from driverUtils import executeOnCaeStartup
executeOnCaeStartup()
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
    referenceRepresentation=ON)
mdb.saveAs(pathName='D:/Temp/press_hexagon')
#: The model database has been saved to "D:\Temp\press_hexagon.cae".
session.viewports['Viewport: 1'].partDisplay.setValues(sectionAssignments=ON, 
    engineeringFeatures=ON)
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
    referenceRepresentation=OFF)
print("Started...")
mdb.models['Model-1'].Material(name='AL')
mdb.models['Model-1'].materials['AL'].Density(table=((2.555e-09, ), ))
mdb.models['Model-1'].materials['AL'].Elastic(temperatureDependency=ON, table=(
    (56832.9, 0.3468, 300.0), (51416.2, 0.3528, 400.0), (45999.5, 0.3588, 
    500.0)))
mdb.models['Model-1'].materials['AL'].plastic.setValues(table=((10.0, 0.0, 0.0, 
    300.0), (26.8, 0.05, 0.0, 300.0), (35.8, 0.1, 0.0, 300.0), (49.8, 0.18, 
    0.0, 300.0), (62.5, 0.26, 0.0, 300.0), (75.2, 0.34, 0.0, 300.0), (88.7, 
    0.41, 0.0, 300.0), (103.4, 0.47, 0.0, 300.0), (119.7, 0.53, 0.0, 300.0), (
    138.1, 0.59, 0.0, 300.0), (159.2, 0.64, 0.0, 300.0), (183.5, 0.69, 0.0, 
    300.0), (10.0, 0.0, 1.0, 300.0), (36.9, 0.05, 1.0, 300.0), (49.5, 0.1, 1.0, 
    300.0), (68.8, 0.18, 1.0, 300.0), (86.2, 0.26, 1.0, 300.0), (103.8, 0.34, 
    1.0, 300.0), (122.4, 0.41, 1.0, 300.0), (142.7, 0.47, 1.0, 300.0), (165.2, 
    0.53, 1.0, 300.0), (190.6, 0.59, 1.0, 300.0), (219.7, 0.64, 1.0, 300.0), (
    253.3, 0.69, 1.0, 300.0), (10.0, 0.0, 10.0, 300.0), (51.0, 0.05, 10.0, 
    300.0), (68.3, 0.1, 10.0, 300.0), (95.0, 0.18, 10.0, 300.0), (119.1, 0.26, 
    10.0, 300.0), (143.3, 0.34, 10.0, 300.0), (169.0, 0.41, 10.0, 300.0), (
    196.9, 0.47, 10.0, 300.0), (228.0, 0.53, 10.0, 300.0), (263.1, 0.59, 10.0, 
    300.0), (303.3, 0.64, 10.0, 300.0), (349.7, 0.69, 10.0, 300.0), (10.0, 0.0, 
    0.0, 400.0), (22.6, 0.05, 0.0, 400.0), (30.3, 0.1, 0.0, 400.0), (42.2, 
    0.18, 0.0, 400.0), (52.9, 0.26, 0.0, 400.0), (63.8, 0.34, 0.0, 400.0), (
    75.4, 0.41, 0.0, 400.0), (88.0, 0.47, 0.0, 400.0), (102.1, 0.53, 0.0, 
    400.0), (118.1, 0.59, 0.0, 400.0), (136.4, 0.64, 0.0, 400.0), (157.7, 0.69, 
    0.0, 400.0), (10.0, 0.0, 1.0, 400.0), (31.2, 0.05, 1.0, 400.0), (41.8, 0.1, 
    1.0, 400.0), (58.2, 0.18, 1.0, 400.0), (73.1, 0.26, 1.0, 400.0), (88.1, 
    0.34, 1.0, 400.0), (104.0, 0.41, 1.0, 400.0), (121.5, 0.47, 1.0, 400.0), (
    140.9, 0.53, 1.0, 400.0), (163.0, 0.59, 1.0, 400.0), (188.3, 0.64, 1.0, 
    400.0), (217.7, 0.69, 1.0, 400.0), (10.0, 0.0, 10.0, 400.0), (43.1, 0.05, 
    10.0, 400.0), (57.7, 0.1, 10.0, 400.0), (80.3, 0.18, 10.0, 400.0), (100.8, 
    0.26, 10.0, 400.0), (121.6, 0.34, 10.0, 400.0), (143.6, 0.41, 10.0, 400.0), 
    (167.7, 0.47, 10.0, 400.0), (194.5, 0.53, 10.0, 400.0), (225.0, 0.59, 10.0, 
    400.0), (259.9, 0.64, 10.0, 400.0), (300.5, 0.69, 10.0, 400.0), (10.0, 0.0, 
    0.0, 500.0), (19.1, 0.05, 0.0, 500.0), (25.6, 0.1, 0.0, 500.0), (35.7, 
    0.18, 0.0, 500.0), (44.8, 0.26, 0.0, 500.0), (54.1, 0.34, 0.0, 500.0), (
    64.0, 0.41, 0.0, 500.0), (74.9, 0.47, 0.0, 500.0), (87.1, 0.53, 0.0, 
    500.0), (101.0, 0.59, 0.0, 500.0), (116.9, 0.64, 0.0, 500.0), (135.6, 0.69, 
    0.0, 500.0), (10.0, 0.0, 1.0, 500.0), (26.3, 0.05, 1.0, 500.0), (35.3, 0.1, 
    1.0, 500.0), (49.2, 0.18, 1.0, 500.0), (61.9, 0.26, 1.0, 500.0), (74.7, 
    0.34, 1.0, 500.0), (88.4, 0.41, 1.0, 500.0), (103.4, 0.47, 1.0, 500.0), (
    120.2, 0.53, 1.0, 500.0), (139.4, 0.59, 1.0, 500.0), (161.4, 0.64, 1.0, 
    500.0), (187.1, 0.69, 1.0, 500.0), (10.0, 0.0, 10.0, 500.0), (36.3, 0.05, 
    10.0, 500.0), (48.7, 0.1, 10.0, 500.0), (67.9, 0.18, 10.0, 500.0), (85.4, 
    0.26, 10.0, 500.0), (103.1, 0.34, 10.0, 500.0), (122.0, 0.41, 10.0, 500.0), 
    (142.8, 0.47, 10.0, 500.0), (166.0, 0.53, 10.0, 500.0), (192.4, 0.59, 10.0, 
    500.0), (222.8, 0.64, 10.0, 500.0), (258.3, 0.69315, 10.0, 500.0)))
mdb.models['Model-1'].HomogeneousSolidSection(name='Section-1', material='AL', 
    thickness=100.0)
session.viewports['Viewport: 1'].partDisplay.setValues(sectionAssignments=OFF, 
    engineeringFeatures=OFF)
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
    referenceRepresentation=ON)
s = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', 
    sheetSize=200.0)
g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
s.setPrimaryObject(option=STANDALONE)
s.Line(point1=(0.0, 0.0), point2=(7.0, 15.0))
s.Line(point1=(7.0, 15.0), point2=(73.0, 15.0))
s.HorizontalConstraint(entity=g[3], addUndoState=False)
s.Line(point1=(73.0, 15.0), point2=(80.0, 0.0))
s.Line(point1=(80.0, 0.0), point2=(73.0, -15.0))
s.Line(point1=(73.0, -15.0), point2=(7.0, -15.0))
s.HorizontalConstraint(entity=g[6], addUndoState=False)
s.Line(point1=(7.0, -15.0), point2=(0.0, 0.0))
p = mdb.models['Model-1'].Part(name='Hexagon', dimensionality=TWO_D_PLANAR, 
    type=DEFORMABLE_BODY)
p = mdb.models['Model-1'].parts['Hexagon']
p.BaseShell(sketch=s)
s.unsetPrimaryObject()
p = mdb.models['Model-1'].parts['Hexagon']
session.viewports['Viewport: 1'].setValues(displayedObject=p)
del mdb.models['Model-1'].sketches['__profile__']
p = mdb.models['Model-1'].parts['Hexagon']
f = p.faces
faces = f.getSequenceFromMask(mask=('[#1 ]', ), )
p.Set(faces=faces, name='Set-1')
#: The set 'Set-1' has been created (1 face).
p = mdb.models['Model-1'].parts['Hexagon']
s = p.edges
side1Edges = s.getSequenceFromMask(mask=('[#3f ]', ), )
p.Surface(side1Edges=side1Edges, name='Surf-1')
#: The surface 'Surf-1' has been created (6 edges).
session.viewports['Viewport: 1'].partDisplay.setValues(sectionAssignments=ON, 
    engineeringFeatures=ON)
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
    referenceRepresentation=OFF)
p = mdb.models['Model-1'].parts['Hexagon']
region = p.sets['Set-1']
p = mdb.models['Model-1'].parts['Hexagon']
p.SectionAssignment(region=region, sectionName='Section-1', offset=0.0, 
    offsetType=MIDDLE_SURFACE, offsetField='', 
    thicknessAssignment=FROM_SECTION)
session.viewports['Viewport: 1'].partDisplay.setValues(sectionAssignments=OFF, 
    engineeringFeatures=OFF, mesh=ON)
session.viewports['Viewport: 1'].partDisplay.meshOptions.setValues(
    meshTechnique=ON)
elemType1 = mesh.ElemType(elemCode=CPS4R, elemLibrary=EXPLICIT, 
    secondOrderAccuracy=OFF, hourglassControl=DEFAULT, 
    distortionControl=DEFAULT)
elemType2 = mesh.ElemType(elemCode=CPS3, elemLibrary=EXPLICIT)
p = mdb.models['Model-1'].parts['Hexagon']
f = p.faces
faces = f.getSequenceFromMask(mask=('[#1 ]', ), )
pickedRegions =(faces, )
p.setElementType(regions=pickedRegions, elemTypes=(elemType1, elemType2))
p = mdb.models['Model-1'].parts['Hexagon']
p.seedPart(size=5.0, deviationFactor=0.1, minSizeFactor=0.1)
p = mdb.models['Model-1'].parts['Hexagon']
p.seedPart(size=3.0, deviationFactor=0.1, minSizeFactor=0.1)
p = mdb.models['Model-1'].parts['Hexagon']
p.generateMesh()
p = mdb.models['Model-1'].parts['Hexagon']
f = p.faces
pickedRegions = f.getSequenceFromMask(mask=('[#1 ]', ), )
p.deleteMesh(regions=pickedRegions)
p = mdb.models['Model-1'].parts['Hexagon']
f = p.faces
pickedRegions = f.getSequenceFromMask(mask=('[#1 ]', ), )
p.setMeshControls(regions=pickedRegions, algorithm=MEDIAL_AXIS)
p = mdb.models['Model-1'].parts['Hexagon']
p.generateMesh()
session.viewports['Viewport: 1'].partDisplay.setValues(mesh=OFF)
session.viewports['Viewport: 1'].partDisplay.meshOptions.setValues(
    meshTechnique=OFF)
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
    referenceRepresentation=ON)
s1 = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', 
    sheetSize=200.0)
g, v, d, c = s1.geometry, s1.vertices, s1.dimensions, s1.constraints
s1.setPrimaryObject(option=STANDALONE)
s1.Line(point1=(0.0, 0.0), point2=(20.0, 0.0))
s1.HorizontalConstraint(entity=g[2], addUndoState=False)
s1.Line(point1=(20.0, 0.0), point2=(51.18, 18.0))
s1.Line(point1=(51.18, 18.0), point2=(81.18, 18.0))
s1.HorizontalConstraint(entity=g[4], addUndoState=False)
s1.Line(point1=(81.18, 18.0), point2=(112.35, 0.0))
s1.Line(point1=(112.35, 0.0), point2=(132.35, 0.0))
s1.HorizontalConstraint(entity=g[6], addUndoState=False)
session.viewports['Viewport: 1'].view.fitView()
p = mdb.models['Model-1'].Part(name='Puan_Top', dimensionality=TWO_D_PLANAR, 
    type=ANALYTIC_RIGID_SURFACE)
p = mdb.models['Model-1'].parts['Puan_Top']
p.AnalyticRigidSurf2DPlanar(sketch=s1)
s1.unsetPrimaryObject()
p = mdb.models['Model-1'].parts['Puan_Top']
session.viewports['Viewport: 1'].setValues(displayedObject=p)
del mdb.models['Model-1'].sketches['__profile__']
p = mdb.models['Model-1'].parts['Puan_Top']
v1, e, d1, n = p.vertices, p.edges, p.datums, p.nodes
p.ReferencePoint(point=p.InterestingPoint(edge=e[2], rule=MIDDLE))
p = mdb.models['Model-1'].parts['Puan_Top']
r = p.referencePoints
refPoints=(r[2], )
p.Set(referencePoints=refPoints, name='Set-1')
#: The set 'Set-1' has been created (1 reference point).
p = mdb.models['Model-1'].parts['Puan_Top']
s = p.edges
side2Edges = s.getSequenceFromMask(mask=('[#1f ]', ), )
p.Surface(side2Edges=side2Edges, name='Surf-1')
#: The surface 'Surf-1' has been created (5 edges).
s = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', 
    sheetSize=200.0)
g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
s.setPrimaryObject(option=STANDALONE)
s.Line(point1=(0.0, 0.0), point2=(20.0, 0.0))
s.HorizontalConstraint(entity=g[2], addUndoState=False)
s.Line(point1=(20.0, 0.0), point2=(51.18, -18.0))
s.Line(point1=(51.18, -18.0), point2=(81.18, -18.0))
s.HorizontalConstraint(entity=g[4], addUndoState=False)
s.Line(point1=(81.18, -18.0), point2=(112.35, 0.0))
s.Line(point1=(112.35, 0.0), point2=(132.35, 0.0))
s.HorizontalConstraint(entity=g[6], addUndoState=False)
session.viewports['Viewport: 1'].view.fitView()
p = mdb.models['Model-1'].Part(name='Puan_bot', dimensionality=TWO_D_PLANAR, 
    type=ANALYTIC_RIGID_SURFACE)
p = mdb.models['Model-1'].parts['Puan_bot']
p.AnalyticRigidSurf2DPlanar(sketch=s)
s.unsetPrimaryObject()
p = mdb.models['Model-1'].parts['Puan_bot']
session.viewports['Viewport: 1'].setValues(displayedObject=p)
del mdb.models['Model-1'].sketches['__profile__']
p = mdb.models['Model-1'].parts['Puan_bot']
v2, e1, d2, n1 = p.vertices, p.edges, p.datums, p.nodes
p.ReferencePoint(point=p.InterestingPoint(edge=e1[2], rule=MIDDLE))
p = mdb.models['Model-1'].parts['Puan_bot']
r = p.referencePoints
refPoints=(r[2], )
p.Set(referencePoints=refPoints, name='Set-1')
#: The set 'Set-1' has been created (1 reference point).
p = mdb.models['Model-1'].parts['Puan_bot']
s = p.edges
side1Edges = s.getSequenceFromMask(mask=('[#1f ]', ), )
p.Surface(side1Edges=side1Edges, name='Surf-1')
#: The surface 'Surf-1' has been created (5 edges).
a = mdb.models['Model-1'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(
    optimizationTasks=OFF, geometricRestrictions=OFF, stopConditions=OFF)
a = mdb.models['Model-1'].rootAssembly
a.DatumCsysByDefault(CARTESIAN)
p = mdb.models['Model-1'].parts['Hexagon']
a.Instance(name='Hexagon-1', part=p, dependent=ON)
a = mdb.models['Model-1'].rootAssembly
p = mdb.models['Model-1'].parts['Puan_bot']
a.Instance(name='Puan_bot-1', part=p, dependent=ON)
a = mdb.models['Model-1'].rootAssembly
p = mdb.models['Model-1'].parts['Puan_Top']
a.Instance(name='Puan_Top-1', part=p, dependent=ON)
a = mdb.models['Model-1'].rootAssembly
a.translate(instanceList=('Puan_Top-1', ), vector=(0.0, 19.035, 0.0))
#: The instance Puan_Top-1 was translated by 0., 19.035, 0. with respect to the assembly coordinate system
a = mdb.models['Model-1'].rootAssembly
a.translate(instanceList=('Puan_bot-1', ), vector=(0.0, -19.035, 0.0))
#: The instance Puan_bot-1 was translated by 0., -19.035, 0. with respect to the assembly coordinate system
session.viewports['Viewport: 1'].view.fitView()
a = mdb.models['Model-1'].rootAssembly
a.translate(instanceList=('Hexagon-1', ), vector=(1.18, -37.035, 0.0))
#: The instance Hexagon-1 was translated by 1.18, -37.035, 0. with respect to the assembly coordinate system
session.viewports['Viewport: 1'].view.fitView()
a = mdb.models['Model-1'].rootAssembly
a.rotate(instanceList=('Hexagon-1', ), axisPoint=(81.18, -37.035, 0.0), 
    axisDirection=(0.0, 0.0, 1.0), angle=-67.98)
#: The instance Hexagon-1 was rotated by -67.98 degrees about the axis defined by the point 81.18, -37.035, 0. and the vector 0., 0., 1.
session.viewports['Viewport: 1'].view.fitView()
session.viewports['Viewport: 1'].assemblyDisplay.setValues(
    adaptiveMeshConstraints=ON)
mdb.models['Model-1'].ExplicitDynamicsStep(name='Step-1', previous='Initial', 
    improvedDtMethod=ON)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='Step-1')
mdb.models['Model-1'].FieldOutputRequest(name='F-Output-2', 
    createStepName='Step-1', variables=('S', 'E', 'PE', 'PEEQ', 'LE', 'COORD'))
regionDef=mdb.models['Model-1'].rootAssembly.allInstances['Puan_Top-1'].sets['Set-1']
mdb.models['Model-1'].HistoryOutputRequest(name='H-Output-2', 
    createStepName='Step-1', variables=('RF1', 'RF2'), region=regionDef, 
    sectionPoints=DEFAULT, rebar=EXCLUDE)
regionDef=mdb.models['Model-1'].rootAssembly.allInstances['Puan_bot-1'].sets['Set-1']
mdb.models['Model-1'].HistoryOutputRequest(name='H-Output-3', 
    createStepName='Step-1', variables=('RF1', 'RF2'), region=regionDef, 
    sectionPoints=DEFAULT, rebar=EXCLUDE)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(interactions=ON, 
    constraints=ON, connectors=ON, engineeringFeatures=ON, 
    adaptiveMeshConstraints=OFF)
mdb.models['Model-1'].ContactProperty('IntProp-1')
mdb.models['Model-1'].interactionProperties['IntProp-1'].NormalBehavior(
    pressureOverclosure=HARD, allowSeparation=ON, 
    constraintEnforcementMethod=DEFAULT)
mdb.models['Model-1'].interactionProperties['IntProp-1'].TangentialBehavior(
    formulation=PENALTY, directionality=ISOTROPIC, slipRateDependency=OFF, 
    pressureDependency=OFF, temperatureDependency=OFF, dependencies=0, table=((
    0.2, ), ), shearStressLimit=None, maximumElasticSlip=FRACTION, 
    fraction=0.005, elasticSlipStiffness=None)
#: The interaction property "IntProp-1" has been created.
a = mdb.models['Model-1'].rootAssembly
region1=a.instances['Puan_Top-1'].surfaces['Surf-1']
a = mdb.models['Model-1'].rootAssembly
region2=a.instances['Hexagon-1'].surfaces['Surf-1']
mdb.models['Model-1'].SurfaceToSurfaceContactExp(name ='Int-1', 
    createStepName='Step-1', master = region1, slave = region2, 
    mechanicalConstraint=KINEMATIC, sliding=FINITE, 
    interactionProperty='IntProp-1', initialClearance=OMIT, datumAxis=None, 
    clearanceRegion=None)
#: The interaction "Int-1" has been created.
a = mdb.models['Model-1'].rootAssembly
region1=a.instances['Puan_bot-1'].surfaces['Surf-1']
a = mdb.models['Model-1'].rootAssembly
region2=a.instances['Hexagon-1'].surfaces['Surf-1']
mdb.models['Model-1'].SurfaceToSurfaceContactExp(name ='Int-2', 
    createStepName='Step-1', master = region1, slave = region2, 
    mechanicalConstraint=KINEMATIC, sliding=FINITE, 
    interactionProperty='IntProp-1', initialClearance=OMIT, datumAxis=None, 
    clearanceRegion=None)
#: The interaction "Int-2" has been created.
session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=ON, bcs=ON, 
    predefinedFields=ON, interactions=OFF, constraints=OFF, 
    engineeringFeatures=OFF)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='Initial')
a = mdb.models['Model-1'].rootAssembly
region = a.instances['Puan_Top-1'].sets['Set-1']
mdb.models['Model-1'].DisplacementBC(name='BC-1', createStepName='Initial', 
    region=region, u1=SET, u2=UNSET, ur3=SET, amplitude=UNSET, 
    distributionType=UNIFORM, fieldName='', localCsys=None)
a = mdb.models['Model-1'].rootAssembly
region = a.instances['Puan_bot-1'].sets['Set-1']
mdb.models['Model-1'].DisplacementBC(name='BC-2', createStepName='Initial', 
    region=region, u1=SET, u2=UNSET, ur3=SET, amplitude=UNSET, 
    distributionType=UNIFORM, fieldName='', localCsys=None)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='Step-1')
a = mdb.models['Model-1'].rootAssembly
region = a.instances['Puan_Top-1'].sets['Set-1']
mdb.models['Model-1'].VelocityBC(name='BC-3', createStepName='Step-1', 
    region=region, v1=UNSET, v2=-17.5, vr3=UNSET, amplitude=UNSET, 
    localCsys=None, distributionType=UNIFORM, fieldName='')
a = mdb.models['Model-1'].rootAssembly
region = a.instances['Puan_bot-1'].sets['Set-1']
mdb.models['Model-1'].VelocityBC(name='BC-4', createStepName='Step-1', 
    region=region, v1=UNSET, v2=17.5, vr3=UNSET, amplitude=UNSET, 
    localCsys=None, distributionType=UNIFORM, fieldName='')
session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=OFF, bcs=OFF, 
    predefinedFields=OFF, connectors=OFF)
mdb.Job(name='Press_Hex', model='Model-1', description='', type=ANALYSIS, 
    atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, 
    memoryUnits=PERCENTAGE, explicitPrecision=SINGLE, 
    nodalOutputPrecision=SINGLE, echoPrint=OFF, modelPrint=OFF, 
    contactPrint=OFF, historyPrint=OFF, userSubroutine='', scratch='', 
    resultsFormat=ODB, parallelizationMethodExplicit=DOMAIN, numDomains=2, 
    activateLoadBalancing=False, multiprocessingMode=DEFAULT, numCpus=2)
mdb.models['Model-1'].Temperature(name='Predefined Field-1', 
    createStepName='Step-1', region=region, distributionType=UNIFORM, 
    crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(300.0, ))
mdb.save()
session.viewports['Viewport: 1'].partDisplay.setValues(sectionAssignments=ON, 
    engineeringFeatures=ON)
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
    referenceRepresentation=OFF)
p1 = mdb.models['Model-1'].parts['Puan_bot']
session.viewports['Viewport: 1'].setValues(displayedObject=p1)
mdb.save()
mdb.jobs['Press_Hex'].submit(consistencyChecking=OFF)


# print("Job is done...")
# odb = session.odbs['D:/temp/Job-1.odb']
# xy1 = xyPlot.XYDataFromHistory(odb=odb, 
#     outputVariableName='Reaction force: RF2 PI: BOTTOM-1 Node 1 in NSET SET-1', 
#     steps=('Load', ), suppressQuery=True, __linkedVpName__='Viewport: 1')
# c1 = session.Curve(xyData=xy1)
# xyp = session.xyPlots['XYPlot-1']
# chartName = xyp.charts.keys()[0]
# chart = xyp.charts[chartName]
# chart.setValues(curvesToPlot=(c1, ), )
# session.charts[chartName].autoColor(lines=True, symbols=True)
# session.viewports['Viewport: 1'].setValues(displayedObject=xyp)
# #: The contents of viewport "Viewport: 1" have been copied to the clipboard.
# abq_ExcelUtilities.excelUtilities.XYtoExcel(xyDataNames='From Current XY Plot', 
#     trueName='From Current XY Plot')
# #: XY Data sent to Excel



# import cast_test as cast
# ds = cast.Dataset(import_path=xy1, rawData=True)
print("Success")