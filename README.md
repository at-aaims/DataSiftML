# DataSiftML

This is the software repository described in the following paper:

> Brewer et al., "Entropy-driven Optimal Sub-sampling of Fluid Dynamics for Developing Machine-learned Surrogates", In the 4th International Workshop on Artificial Intelligence and Machine Learning for Scientific Applications (AI4S). IEEE. https://doi.org/10.1145/3624062.3626084

This code has been tested with Python 3.8.15.

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

### Using kNN

To sample the points using MaxEnt and also select k nearest neighbors (kNN) around each point that MaxEnt selects, use:

    > python subsample_maxent.py --path ./data --target p -ns 540 -nc 20 -nn 4 --plot 

The `--plot` here will generate plots/knn.png so you can see a sample of what it looks like.
Using this method with `--dtype interpolated` is not yet supported. 
It may either give incorrect results or crash.

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

    > python subsample_maxent.py --path ./data --target drag -ns 540 -nc 10 --dtype interpolated

    > python subsample_random.py --path ./data --target drag -ns 540 --dtype interpolated

### Subsampling nekRS CSV data files

To use CSV files as input for subsampling, we add `--dtype csv`, e.g., if the CSV data files
are in a folder called `./nekrs_data`, we can run the following:

    > python subsample_maxent.py -ns 1000 -nc 20 --dtype csv --path ./nekrs_data

    > python subsample_random.py -ns 1000 --dtype csv --path ./nekrs_data

    > python subsample_full.py --path ./data --dtype csv --path ./nekrs_data

    > python train.py --epochs 5 --batch 32 --yscaler None

### Then train the neural network

    > python train.py --epochs 5 --batch 32 

### Temporal forecasting 

Previous examples assume fully connected network (FCN) - time independent samples. 
In order to perform the same analysis with temporal forecasting instead, first do 
subsampling on windowed samples to generate sequences:

    > python subsample_maxent.py --path ./data --target drag -ns 540 -nc 10 --window 3

Then train using LSTM architecture

    > python train.py --epochs 5 --batch 32 --arch lstm --window 3 --target drag

Note: the `--target drag` on the train.py command is only to force to 
`args.field_prediction_type` to be set to `FPT_GLOBAL`, which is then passed to 
the call to `dataloader.create_sequences(...)`.

### Create a movie of results from maxent.py

    > ffmpeg -framerate 30 -i frame_%*.png -c:v libx264 -pix_fmt yuv420p -r 30 output.mp4

### Important notes

1. If the definition of features, target, `--window`, or `--dtype` is changed, `snapshots/raw_data.npz` needs to be deleted; otherwise may throw an error or give wrong result. 

2. For reproducibility for Fig. 8 results in the AI4S paper, the following two commands were run five times and averaged for `-ns` values of 540, 1080, 2160. 

        > python subsample_maxent.py --path ./data --target drag -ns 540 -nc 20 -cv wz \
                                     --dtype structured --noseed

        > python train.py --epochs 100 --batch 2 --patience 12 --yscaler None \
                          --yscalefactor 10 --test_frac 0.05

    and similarly for random sub-sampling:

        > python subsample_random.py --path ./data --target drag -ns 540 --dtype structured --noseed

        > python train.py --epochs 100 --batch 2 --patience 12 --yscaler None \
                          --yscalefactor 10 --test_frac 0.05

