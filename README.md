# DataSiftML

### Dataset

Note: OpenFOAM simulation was originally sourced from 
https://github.com/AsmaaHADANE/Youtube-Tutorials/blob/master/flow_around_cylinder.zip
from the online version, the forces and cell-centers were additionally computed. 

    tar xvfz data.tgz

### Setup environment

    pip install -r requirements.txt

### Show options

    python args.py -h 

### Sub-sampling usage examples

    DPATH=./data

When sampling using either 'random', 'random-weighted', 'silhouette', 'equal' you first need to figure out what the minimum number of samples is through all the clusters. To do this just set `-ns 0` then run through one time, and it will tell you the minimum number of samples. Then set your value less than or equal to that value, e.g., 

    python subsample_maxent.py --path $DPATH --target drag -ns 0 -nc 10 --subsample equal
    python subsample_maxent.py --path $DPATH --target drag -ns 10 -nc 10 --subsample equal

Note that when you use `--subsample equal`, `-ns` is actually the number of samples per cluster, rather than the total number of samples. So, if you want 50 overall samples per timestep, and `-nc 10`, then you would set `-ns 5`. Also, for equalpercentage, we are currently passing the percentage through the `--cutoff` value. So if you want 
to sample 10% of each cluster set `--cutoff 0.1`.

### By default DataSiftML will use proportional subsampling

    python subsample_maxent.py --path $DPATH --target drag -ns 750 -nc 10 # create maxent subsampled file

    python subsample_random.py --path $DPATH --target drag -ns 750 # create randomly subsampled npz file

    python subsample_full.py --path ./data # create npz of full dataset

### Workflow for first interpolating to Cartesian version

    Run `foamToVTK` inside ./data to generate VTK/*.vtk files

    Install ParaView from https://www.paraview.org/download/

    /Applications/ParaView-5.11.0.app/Contents/bin/pvpython pintervtk.py
    -> creates U_800x200.npy, p_800x200.npy, wz_800x200.npy

    python subcart_maxent.py --path $DPATH --target drag -ns 750 -nc 10

### Then train the neural network

    python train.py --epochs 5 --batch 32 

### Temporal forecasting 

Previous examples assume fully connected network (FCN) - time independent samples 
to perform the same analysis with temporal forecasting, first do subsampling on windowed samples to generate sequences

    python subsample_maxent.py --path $DPATH --target drag -ns 750 -nc 10 --window 3

Then train using LSTM architecture

    python train.py --epochs 5 --batch 32 --arch lstm --window 3

### Dynamic mode decomposition (DMD)

    python dmd.py --path $DPATH

### Create a movie of results from maxent.py

    ffmpeg -framerate 30 -i frame_%*.png -c:v libx264 -pix_fmt yuv420p -r 30 output.mp4

*** Important Note: that if you change definition of features and target, and also window size, 
    you will need to delete snapshots/raw_data.npz otherwise may throw an error or give wrong result. 
