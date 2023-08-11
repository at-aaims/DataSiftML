"""Usage example: python dataloader.py --path $HOME/foam/cylinder"""
import numpy as np
import pandas as pd

from stream import compute_stream_function
from fluidfoam import readscalar, readvector, readforce


class DataLoader():

    def __init__(self, path, verbose=False):
        self.path = path
        self.verbose = verbose

    def load_forces(self, write_interval=100):
        forces = readforce(self.path, time_name='0', name='forces')
        # Drag force is composed of both a viscous and pressure components
        time = forces[:, 0]
        drag = forces[:, 1] + forces[:, 2]
        return time[::write_interval], drag[::write_interval]

    def load_xyz(self):
        x, y, z = readvector(self.path, '0', 'C.gz')
        x = np.expand_dims(x, axis=1)
        y = np.expand_dims(y, axis=1)
        return x, y

    def load_single_timestep(self, time: str, target='p'):
        # Read solution values from OpenFOAM simulation
        stime = str(time)
        p = readscalar(self.path, stime, 'p.gz')
        x, y, z = readvector(self.path, stime, 'C.gz')
        Ux, Uy, Uz = readvector(self.path, stime, 'U.gz')
        wx, wy, wz = readvector(self.path, stime, 'vorticity.gz')

        # Add an extra dimension
        p = np.expand_dims(p, axis=1)
        x = np.expand_dims(x, axis=1)
        y = np.expand_dims(y, axis=1)
        Ux = np.expand_dims(Ux, axis=1)
        Uy = np.expand_dims(Uy, axis=1)
        wz = np.expand_dims(wz, axis=1)

        stacked = np.hstack((x, y, p, Ux, Uy, wz))

        df = pd.DataFrame(stacked, columns=['x', 'y', 'p', 'Ux', 'Uy', 'wz'])
        X = df[['Ux', 'Uy']].to_numpy()

        if target == 'p':
            Y = p
        elif target == 'wz':
            Y = wz
        elif target == 'stream':
            #Y =  df[['p', 'wz']].to_numpy()
            Y = compute_stream_function(u, v, omega)
        else:
            raiseValueError('target not supported')

        return X, Y

    def load_multiple_timesteps(self, write_interval, num_timesteps, target='p'):

        p = readscalar(self.path, str(write_interval), 'p.gz')
        num_pts = p.shape[0]
        
        print(num_pts, num_timesteps)

        p = np.zeros((num_timesteps, num_pts))
        u = np.zeros((num_timesteps, num_pts))
        v = np.zeros((num_timesteps, num_pts))
        wz = np.zeros((num_timesteps, num_pts))

        for i, ts in enumerate(range(write_interval, write_interval*num_timesteps+1, write_interval)):
            print(i, ts)
            p[i, :] = readscalar(self.path, str(ts), 'p.gz')
            u[i, :], v[i, :], _ = readvector(self.path, str(ts), 'U.gz')
            _, _, wz[i, :] = readvector(self.path, str(ts), 'vorticity.gz')

        X = np.stack((u, v), axis=-1)

        if target == 'p':
            Y = p
        elif target == 'wz':
            Y = wz
        elif target == 'pwz':
            Y = np.stack((p, wz), axis=1)
        elif target == 'drag':
            time, drag = self.load_forces()
            Y = np.expand_dims(drag, axis=1)
            print('drag here:', Y.shape)
        elif target == 'stream':
            Y = compute_stream_function(u, v, wz)
        else:
            raiseValueError('target not supported')

        return X, Y

    def to_csv(self, Y, X, time, columns):
        """Output CSV file named by timestamp, e.g. 1000.csv"""
        df = pd.DataFrame(np.concatenate((Y, X), axis=1), columns=columns)
        df.to_csv(str(time) + '.csv', index=False)

def create_sequences_from_csv(path, sequence_length):
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

def create_sequences(X, Y, window_size=3):
    """ Create time sequences of a given window size from the input arrays X and Y """
    nt, nsamples, nvars = X.shape
    num_sequences = nt - window_size + 1

    X_sequences = np.zeros((num_sequences, window_size, nsamples*nvars))
    Y_sequences = np.zeros((num_sequences, window_size))

    for i in range(num_sequences):
        X_sequences[i] = X[i:i+window_size].reshape(window_size, nsamples*nvars)
        Y_sequences[i] = Y[i:i+window_size]

    return (X_sequences, Y_sequences)


if __name__ == "__main__":

    from args import args
    
    dl = DataLoader(args.path)
    
    #X, Y = dl.read_solution('1000')
    #print(X.shape, Y.shape)

#    # Read solution values from OpenFOAM simulation
#    for i, ts in enumerate(range(100, 10100, 100)):
#        print(i, ts)
#        X, Y = dl.load_single_timestep(ts)
#        dl.to_csv(Y, X, ts, columns=['wz', 'p', 'Ux', 'Uy'])

    X, Y = create_sequences(*dl.load_multiple_timesteps(100, 100))
    print(X.shape, Y.shape)

