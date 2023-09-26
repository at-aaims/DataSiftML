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

when sampling using either 'random', 'random-weighted', 'silhouette', 'equal' you first need to figure out what the minimum number of samples is through all the clusters. To do this just set `-ns 0` then run through one time, and it will tell you the minimum number of samples. Then set your value less than or equal to that value, e.g., 

    python subsample_maxent.py --path $DPATH --target drag -ns 0 -nc 10 --subsample equal
    python subsample_maxent.py --path $DPATH --target drag -ns 10 -nc 10 --subsample equal

note that when you use `--subsample equal`, `-ns` is actually the number of samples per cluster, rather than the total number of samples. So, if you want 50 overall samples per timestep, and `-nc 10`, then you would set `-ns 5`. Also, for equalpercentage, we are currently passing the percentage through the `--cutoff` value. So if you want 
to sample 10% of each cluster set `--cutoff 0.1`.

### By default DataSiftML will use proportional subsampling

    python subsample_maxent.py --path $DPATH --target drag -ns 750 -nc 10 

    python subsample_random.py --path $DPATH --target drag -ns 750

### Then train the neural network

    python train.py --epochs 5 --batch 32 

### Dynamic mode decomposition (DMD)
    python dmd.py --path $DPATH

### Create a movie of results from maxent.py

    ffmpeg -framerate 30 -i frame_%*.png -c:v libx264 -pix_fmt yuv420p -r 30 output.mp4