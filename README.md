# DataSiftML

This is the software repository described in the following paper:

Brewer et al., "Entropy-driven Optimal Sub-sampling of Fluid Dynamics for Developing Machine-learned Surrogates", 
In the 4th International Workshop on Artificial Intelligence and Machine Learning for Scientific Applications (AI4S). IEEE.

### Dataset

Note: OpenFOAM simulation was originally sourced from 
https://github.com/AsmaaHADANE/Youtube-Tutorials/blob/master/flow_around_cylinder.zip
from the online version, the forces and cell-centers were additionally computed. 

    > tar xvfz data.tgz

### Setup environment

    > pip install -r requirements.txt

### Show options

    > python args.py -h 

### By default DataSiftML will use proportional subsampling (this is what we suggest you use as default)

    > python subsample_maxent.py --path ./data --target drag -ns 540 -nc 20 # create maxent subsampled file

    > python subsample_random.py --path ./data --target drag -ns 540 # create randomly subsampled npz file

    > python subsample_full.py --path ./data --target drag # create npz of full dataset

### Sub-sampling usage examples using other methods

When sampling using either 'random', 'random-weighted', 'silhouette', 'equal' you first need to figure out what the minimum number of samples is through all the clusters. To do this just set `-ns 0` then run through one time, and it will tell you the minimum number of samples. Then set your value less than or equal to that value, e.g., 

    > python subsample_maxent.py --path ./data --target drag -ns 0 -nc 10 --subsample equal
    > python subsample_maxent.py --path ./data --target drag -ns 10 -nc 10 --subsample equal

Note that when you use `--subsample equal`, `-ns` is actually the number of samples per cluster, rather than the total number of samples. So, if you want 50 overall samples per timestep, and `-nc 10`, then you would set `-ns 5`. Also, for equalpercentage, we are currently passing the percentage through the `--cutoff` value. So if you want 
to sample 10% of each cluster set `--cutoff 0.1`.

### Workflow for first interpolating to Cartesian grid

Need to first install OpenFOAM. The easiest way to do this if you have Docker:

    > docker pull openfoam/openfoam7-paraview56

Then:

    > docker run -it -v $HOME/foam:/home/openfoam openfoam/openfoam7-paraview56

Run the following command to generate VTK/*.vtk files from within the Docker container,
within the data folder containing the OpenFOAM solution files:

    >> foamToVTK 

Install ParaView from https://www.paraview.org/download/. Then run:

    > /Applications/ParaView-5.11.0.app/Contents/bin/pvpython interpolate.py

This will output the file ./snapshots/interpolated.npz

    > python subsample_maxent.py --path ./data --target drag -ns 540 -nc 10 --dtype structured

    > python subsample_random.py --path ./data --target drag -ns 540 --dtype structured

### Subsampling nekRS CSV data files

To use CSV files as input for subsampling, we add `--dtype csv`:

    > python subsample_maxent.py -ns 1000 -nc 20 --dtype csv --path ./nekrs_data

### Then train the neural network

    > python train.py --epochs 5 --batch 32 

### Temporal forecasting 

Previous examples assume fully connected network (FCN) - time independent samples. 
In order to perform the same analysis with temporal forecasting instead, first do 
subsampling on windowed samples to generate sequences:

    > python subsample_maxent.py --path ./data --target drag -ns 540 -nc 10 --window 3

Then train using LSTM architecture

    > python train.py --epochs 5 --batch 32 --arch lstm --window 3

### Dynamic mode decomposition (DMD)

    > python dmd.py --path ./data

### Create a movie of results from maxent.py

    > ffmpeg -framerate 30 -i frame_%*.png -c:v libx264 -pix_fmt yuv420p -r 30 output.mp4

*** Important Note: that if you change definition of features and target or --window or --dtype you will need to delete snapshots/raw_data.npz otherwise may throw an error or give wrong result. 
