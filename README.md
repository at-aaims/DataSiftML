
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

# first subsample either using maxent or random method

    python subsample_maxent.py --path $DPATH --target drag -ns 750 -nc 10 --cutoff 0.8

    python subsample_random.py --path $DPATH --target drag -ns 750

# then train the neural network

    python train.py --path $DPATH --epochs 5 --batch 32 

# dynamic mode decomposition (DMD)
    python dmd.py --path $DPATH

# create a movie of results from maxent.py

    ffmpeg -framerate 30 -i frame_%*.png -c:v libx264 -pix_fmt yuv420p -r 30 output.mp4

# push to gitlab

    git push -u origin main

# push to github

    git push -u github main


