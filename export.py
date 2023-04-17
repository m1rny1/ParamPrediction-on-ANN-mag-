# -*- coding: mbcs -*-
#
# Abaqus/CAE Release 2022 replay file
# Internal Version: 2021_09_15-20.57.30 176069
# Run by Admin on Fri Apr 14 14:45:05 2023
#

# from driverUtils import executeOnCaeGraphicsStartup
# executeOnCaeGraphicsStartup()
#: Executing "onCaeGraphicsStartup()" in the site directory ...
from abaqus import *
from abaqusConstants import *
session.Viewport(name='Viewport: 1', origin=(0.0, 0.0), width=123.401039123535, 
    height=113.253471374512)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].maximize()
from caeModules import *
from driverUtils import executeOnCaeStartup
executeOnCaeStartup()
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
    referenceRepresentation=ON)
openMdb(pathName='C:/Users/Admin/Downloads/press_hexagon.cae')
#: The model database "C:\Users\Admin\Downloads\press_hexagon.cae" has been opened.
session.viewports['Viewport: 1'].setValues(displayedObject=None)
p = mdb.models['Model-1'].parts['Hexagon']
session.viewports['Viewport: 1'].setValues(displayedObject=p)
a = mdb.models['Model-1'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
o3 = session.openOdb(name='D:/temp/Press_Hex.odb')
#: Model: D:/temp/Press_Hex.odb
#: Number of Assemblies:         1
#: Number of Assembly instances: 0
#: Number of Part instances:     3
#: Number of Meshes:             3
#: Number of Element Sets:       1
#: Number of Node Sets:          5
#: Number of Steps:              1
session.viewports['Viewport: 1'].setValues(displayedObject=o3)
session.viewports['Viewport: 1'].makeCurrent()
a = mdb.models['Model-1'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
session.viewports['Viewport: 1'].setValues(
    displayedObject=session.odbs['D:/temp/Press_Hex.odb'])
session.viewports['Viewport: 1'].assemblyDisplay.setValues(
    optimizationTasks=OFF, geometricRestrictions=OFF, stopConditions=OFF)
odb = session.odbs['D:/temp/Press_Hex.odb']
xyList = xyPlot.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=((
    'COORD', NODAL, ((COMPONENT, 'COOR2'), )), ('PEEQ', INTEGRATION_POINT), ), 
    nodePick=(('PUAN_TOP-1', 1, ('[#1 ]', )), ('HEXAGON-1', 4, (
    '[#0:5 #500 #0:3 #110 ]', )), ), )
xyp = session.XYPlot('XYPlot-1')
chartName = xyp.charts.keys()[0]
chart = xyp.charts[chartName]
curveList = session.curveSet(xyData=xyList)
chart.setValues(curvesToPlot=curveList)
session.charts[chartName].autoColor(lines=True, symbols=True)
session.viewports['Viewport: 1'].setValues(displayedObject=xyp)
import sys
sys.path.insert(8, 
    r'd:/SIMULIA/EstProducts/2022/win_b64/code/python2.7/lib/abaqus_plugins/excelUtilities')
import abq_ExcelUtilities.excelUtilities
abq_ExcelUtilities.excelUtilities.XYtoExcel(
    xyDataNames='_COORD:COOR2 PI: PUAN_TOP-1 N: 1,_PEEQ (Avg: 75%) PI: HEXAGON-1 N: 281,_PEEQ (Avg: 75%) PI: HEXAGON-1 N: 285,_PEEQ (Avg: 75%) PI: HEXAGON-1 N: 309,_PEEQ (Avg: 75%) PI: HEXAGON-1 N: 313', 
    trueName='From Current XY Plot')
#: Multiple XY Data are exported. No chart will be created.
#: XY Data sent to Excel
