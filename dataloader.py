"""Usage example: python dataloader.py --path $HOME/foam/cylinder"""
import numpy as np
import pandas as pd

from fluidfoam import readscalar, readvector, readforce


class DataLoader():

    def __init__(self, path, verbose=False):
        self.path = path
        self.verbose = verbose

    def read_solution(self, time):
        # Read solution values from OpenFOAM simulation
        p = readscalar(self.path, time, 'p.gz')
        x, y, z = readvector(self.path, time, 'C.gz')
        Ux, Uy, Uz = readvector(self.path, time, 'U.gz')
        wx, wy, wz = readvector(self.path, time, 'vorticity.gz')
        forces = readforce(self.path, time_name='0', name='forces')

        # Drag force is composed of both a viscous and pressure components
        drag = forces[:, 1] + forces[:, 2]

        if self.verbose:
            print('force.shape:', forces.shape)
            print('drag.shape:', drag.shape)
            print(drag)

        # Add an extra dimension
        p = np.expand_dims(p, axis=1)
        x = np.expand_dims(x, axis=1)
        y = np.expand_dims(y, axis=1)
        Ux = np.expand_dims(Ux, axis=1)
        Uy = np.expand_dims(Uy, axis=1)
        wz = np.expand_dims(wz, axis=1)

        stacked = np.hstack((x, y, p, Ux, Uy, wz))

        if self.verbose:
            print(p.shape)
            print(x.shape)
            print(y.shape)
            print(Ux.shape)
            print(Uy.shape)
            print(wz.shape)
            print(stacked.shape)

        df = pd.DataFrame(stacked, columns=['x', 'y', 'p', 'Ux', 'Uy', 'wz'])
        X = df[['p', 'Ux', 'Uy']].to_numpy()
        #df_sub = df[['p', 'wz']]
        Y = wz

        return X, Y

    def write_to_csv(self, time):
        """Output CSV file named by timestamp, e.g. 1000.csv"""
        file_name = time + '.csv'
        df.to_csv(file_name, index=False)

    def create_sequences(path, sequence_length):
        """Read the CSV files and create sequences"""
        files = sorted([f for f in os.listdir(path) if f.endswith('.csv')])
        sequences = []
        labels = []

        for i in range(sequence_length, len(files) + 1):
            sequence = []
            label_seq = []
            for j in range(i-sequence_length, i):
                file = files[j]
                df = pd.read_csv(os.path.join(path, file))
                label_seq.append(df.iloc[:, 0].values)  # assume the first column is the target
                sequence.append(df.iloc[:, 1:].values)  # rest of the columns are features

            sequences.append(np.array(sequence))
            labels.append(np.array(label_seq))

        return np.array(sequences), np.array(labels)


if __name__ == "__main__":

    from args import args
    
    dl = DataLoader(args.path)
    
    X, Y = dl.read_solution('1000')
    print(X.shape, Y.shape)


