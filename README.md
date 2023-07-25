
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

    python jetfog.py -f heaviside -e 200 --plot --test_frac 0.01 --spacing gaussian --noise 0.1

    python maxent.py --path ./data --time 1000 -nc 10 -ns 1000 --plot

    python train.py --path ./data --epochs 5 --batch 32

    python dmd.py --path ./data

# push to github

    git push -u github main
