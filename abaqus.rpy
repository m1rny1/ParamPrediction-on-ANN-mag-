# -*- coding: mbcs -*-
#
# Abaqus/CAE Release 2022 replay file
# Internal Version: 2021_09_15-20.57.30 176069
# Run by Admin on Sat Mar 11 02:24:10 2023
#

# from driverUtils import executeOnCaeGraphicsStartup
# executeOnCaeGraphicsStartup()
#: Executing "onCaeGraphicsStartup()" in the site directory ...
from abaqus import *
from abaqusConstants import *
session.Viewport(name='Viewport: 1', origin=(1.11979, 1.1169), width=164.833, 
    height=110.796)
session.viewports['Viewport: 1'].makeCurrent()
from driverUtils import executeOnCaeStartup
executeOnCaeStartup()
execfile('abaqus.py', __main__.__dict__)
#: The model database has been saved to "D:\Temp\press_hexagon.cae".
#: Started...
#* AttributeError: 'AbaqusMethod' object has no attribute 'setValues'
#*     mdb.models['Model-1'].materials['AL'].Plastic.setValues(table=((10.0, 
#* 0.0, 0.0,
