
# dataset

    Note: data was originally sourced from 
    https://github.com/AsmaaHADANE/Youtube-Tutorials/blob/master/flow_around_cylinder.zip
    from the online version, the forces and cell-centers were additionally computed. 

    tar xvfz data.tgz

# setup environment

    pip install -r requirements.txt

# show options

    python args.py -h 

# jetfog.py is for training hard-to-model functions and predicting their uncertainty, e.g.:

    python jetfog.py -f heaviside -e 200 --plot --test_frac 0.01 --spacing gaussian --noise 0.1

# sub-sampling usage examples

    DPATH=./data

# when sampling using either 'random', 'random-weighted', 'silhouette', 'equal' you first need to figure out 
# what the minimum number of samples is through all the clusters. To do this just set -ns 0 then run through 
# one time, and it will tell you the minimum number of samples. Then set your value less than or equal to that 
# value, e.g., 

    python subsample_maxent.py --path $DPATH --target drag -ns 0 -nc 10 --subsample equal
    python subsample_maxent.py --path $DPATH --target drag -ns 10 -nc 10 --subsample equal

# note that when you use --subsample equal -ns is actually the number of samples per cluster, rather than the 
# total number of samples. So, if you want 50 overall samples per timestep, and -nc 10, then you would set -ns 5

# also, for equalpercentage, we are currently passing the percentage through the --cutoff value. So if you want 
# to sample 10% of each cluster set --cutoff 0.1

# by default it will use proportional subsampling

    python subsample_maxent.py --path $DPATH --target drag -ns 750 -nc 10 

    python subsample_random.py --path $DPATH --target drag -ns 750

# then train the neural network

    python train.py --epochs 5 --batch 32 

# dynamic mode decomposition (DMD)
    python dmd.py --path $DPATH

# create a movie of results from maxent.py

    ffmpeg -framerate 30 -i frame_%*.png -c:v libx264 -pix_fmt yuv420p -r 30 output.mp4
