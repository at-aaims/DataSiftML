import numpy as np
import os
import re

from constants import SNPDIR
from paraview.simple import LegacyVTKReader, PointPlaneInterpolator, ResampleToImage, SaveData

DPATH = './data/VTK'


def extract_integer_from_filename(filename):
    """Using regex to find the first sequence of digits in the filename"""
    match = re.search(r'(\d+)', filename)
    if match: return int(match.group(1))
    return 0


def vtk_to_structnpy(fname, w, h, save=False):
        """Use pvpython to interpolate onto cutting plane"""
        data = LegacyVTKReader(registrationName=fname, FileNames=[os.path.join(DPATH, fname)])
        plane = PointPlaneInterpolator(registrationName='plane', Input=data, Source='Bounded Plane')
        plane.Kernel = 'VoronoiKernel'
        plane.Locator = 'Static Point Locator'
        plane.Source.BoundingBox = [-10.0, 40.0, -9.0, 9.0, -0.5, 0.5]
        plane.Source.Center = [0.0, 0.0, 0.0]
        rsfilter = ResampleToImage(Input=plane)
        rsfilter.SamplingDimensions = [w, h, 1]
        rsfilter.UpdatePipeline()
        outputData = rsfilter.GetClientSideObject().GetOutputDataObject(0)
        gpd = lambda s : np.array(outputData.GetPointData().GetArray(s))
        x, y, u, p, wz = gpd('Cx'), gpd('Cy'), gpd('U'), gpd('p'), gpd('vorticity')
        if save: # Save to VTK file
            SaveData('VTKstructured/'+fname, proxy=rsfilter, ChooseArraysToWrite=1,
            PointDataArrays=['C', 'Cx', 'Cy', 'Cz', 'U', 'p', 'vorticity'])
        return (x, y, u, p, wz)


if __name__ == '__main__':

    vtkfiles = [sfile for sfile in os.listdir(DPATH) if sfile[-3:] =='vtk']
    sfnames = sorted(vtkfiles, key=extract_integer_from_filename)
    # Aggregate [x, y, u, v, p] for each time-step structured grid
    w, h = 104, 104
    z = lambda n : np.zeros((len(sfnames), w, h, n))
    # Following assumes X is U - which consists of [u, v, w]
    # cv is the collective variables - currently we are using wz
    arrays = { 'x': z(1), 'y': z(1), 'X': z(2), 'Y': z(1), 'cv': z(1) }
    
    for idx, nfile in enumerate(sfnames[:]):
        
        print('Reading {}'.format(os.path.join(DPATH, nfile)))
        x, y, u, p, wz = vtk_to_structnpy(nfile, w, h)
        arrays['x'][idx] = x.reshape(w, h, 1)
        arrays['y'][idx] = y.reshape(w, h, 1)
        arrays['X'][idx] = u.reshape(w, h, 3)[:,:,:2] # only consider u, v for 2D flow
        arrays['Y'][idx] = p.reshape(w, h, 1) # currently using pressure for target
        arrays['cv'][idx] = wz.reshape(w, h, 3)[:, :, 2:3] # vorticity is 3D vector - only consider z-comp

    outfile = os.path.join(SNPDIR, "interpolated.npz")
    np.savez(outfile, **arrays)
    print(f"output {outfile}")
