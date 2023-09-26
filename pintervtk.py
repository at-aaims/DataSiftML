from paraview.simple import *
import vtk
#sys.path.insert(0,'/home/danielmtz/miniconda3/envs/tf-cpu/lib/python3.8/site-packages')
import vtk.numpy_interface.dataset_adapter as dsa
from matplotlib import pyplot as plt
#from vtk.util.numpy_support import vtk_to_numpy
import sys
import numpy as np
import os
import re

def extract_integer_from_filename(filename):
    # Using regex to find the first sequence of digits in the filename
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0

# No paraview tools, just vtk
def vtk_structured_to_numpy(structured_data):
    # Get the dimensions of the structured grid
    dimensions = structured_data.GetDimensions()
    num_points = np.prod(dimensions)

    # Get the points from the structured grid
    points = structured_data.GetPoints()
    vtk_data_array = points.GetData()

    # Convert the VTK data array to a NumPy array
    numpy_array = np.array(vtk_data_array).reshape(num_points, 3)

    return numpy_array

# Paraview routine: pvpython
def vtk_to_structnpy(fname, save=False):
        print('Reading {}'.format(fname))
        data = LegacyVTKReader(registrationName=fname, FileNames=['./VTK/'+fname])
        plane = PointPlaneInterpolator(registrationName='plane', Input=data, Source='Bounded Plane')
        plane.Kernel = 'VoronoiKernel'
        plane.Locator = 'Static Point Locator'
        plane.Source.BoundingBox = [-10.0, 40.0, -9.0, 9.0, -0.5, 0.5]
        plane.Source.Center = [0.0, 0.0, 0.0]
        rsfilter = ResampleToImage(Input=plane)
        rsfilter.SamplingDimensions = [800,400,1]
        rsfilter.UpdatePipeline()
        outputData = rsfilter.GetClientSideObject().GetOutputDataObject(0)
        dataU = outputData.GetPointData().GetArray('U')
        datap = outputData.GetPointData().GetArray('p')
        datav = outputData.GetPointData().GetArray('vorticity')
        u = np.array(dataU)
        p = np.array(datap)
        wz = np.array(datav)
        if save:
            # Save into vtk file
            SaveData('VTKstructured/'+name, proxy=rsfilter, ChooseArraysToWrite=1,
            PointDataArrays=['C', 'Cx', 'Cy', 'Cz', 'U', 'p', 'vorticity'])
        return u, p, wz 

def read_vtk_file(file_path):
    reader = vtk.vtkDataSetReader()
    reader.SetFileName(file_path)
    reader.Update()
    #reader.ReadAllScalarsOn()
    return reader.GetOutput()


if __name__ == '__main__':
    save = True
    dpath = './VTK/'
    vtkfiles = [sfile for sfile in os.listdir(dpath) if sfile[-3:] =='vtk']
    sfnames = sorted(vtkfiles, key=extract_integer_from_filename)
    print(sfnames)
    nocases = len(sfnames)
    # Aggregate [u,v,w,p] for each time-step structured grid
    datau = np.zeros((nocases, 800, 400, 2))
    datap = np.zeros((nocases, 800, 400, 1))
    datawz = np.zeros((nocases, 800, 400, 1))
    
    for c, nfile in enumerate(sfnames[:]):
        
        print('Reading {}'.format(dpath+nfile))
        u, p, wz = vtk_to_structnpy(nfile)
        u = u.reshape(800, 400, 3)
        p = p.reshape(800, 400, 1)
        wz = wz.reshape(800, 400, 3)
        datau[c] = u[:,:,:2]
        datap[c] = p
        datawz[c] = np.expand_dims(wz[:,:,2], axis=2)
        print('Flow field U shape: ', u.shape)
        print('Pressure field shape', p.shape)
        print('Vorticity field shape', wz.shape)
        if save:
            np.save('U_800x200.npy', datau)
            np.save('p_800x200.npy', datap)
            np.save('wz_800x200.npy', datawz)

        #if plotp:
        #    # Plot
        #    fig, ax = plt.subplots()
        #    im = ax.imshow(pressure[:,:,0], cmap='jet')
        #    cbar = fig.colorbar(im,ax=ax)
        #    cbar.set_label('Pressure')
        #    ax.set_xlabel('X')
        #    ax.set_ylabel('Y')
        #    ax.set_title('Pressure Field')
        #    plt.savefig('test_p.png')
