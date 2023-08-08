
# dataset

    Note: data was originally sourced from 
    https://github.com/AsmaaHADANE/Youtube-Tutorials/blob/master/flow_around_cylinder.zip
    from the online version, the forces and cell-centers were additionally computed. 

    tar xvfz data.tgz

# setup environment

    pip install -r requirements.txt

# show options

    python args.py -h 

# usage examples

    DPATH=./data

    python jetfog.py -f heaviside -e 200 --plot --test_frac 0.01 --spacing gaussian --noise 0.1

    python maxent3.py --path $DPATH -nc 10 --plot --cutoff 0.5 --target wz

    python train.py --path $DPATH --epochs 5 --batch 32 --target p --subsample

    python dmd.py --path $DPATH

# create a movie of results from maxent3.py

    ffmpeg -framerate 30 -i frame_%04d.png -c:v libx264 -pix_fmt yuv420p -r 30 output.mp4

# push to gitlab

    git push -u origin main

# push to github

    git push -u github main


