#!/bin/bash
from paraview.simple import *

data = LegacyVTKReader(registrationName='data_10000.vtk', FileNames=['C:\\Users\\dmgco\\Downloads\\data\\VTK\\data_10000.vtk'])

pointPlaneInterpolator1 = PointPlaneInterpolator(registrationName='PointPlaneInterpolator1', Input=data,
    Source='Bounded Plane')
pointPlaneInterpolator1.Kernel = 'VoronoiKernel'
pointPlaneInterpolator1.Locator = 'Static Point Locator'

# init the 'Bounded Plane' selected for 'Source'
pointPlaneInterpolator1.Source.BoundingBox = [-10.0, 25.0, -7.0, 7.0, -0.5, 0.5]
pointPlaneInterpolator1.Source.Center = [0.0, 0.0, 0.0]
#pointPlaneInterpolator1.Source.Resolution(

resample_filter = ResampleToImage(Input=pointPlaneInterpolator1)
resample_filter.SamplingDimensions = [400, 200, 1]

