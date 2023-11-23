"""Usage example: python dataloader.py --path $HOME/foam/cylinder"""
import numpy as np
import os
import pandas as pd

from constants import *
from stream import compute_stream_function
# We do this so that users who are not using OpenFOAM need not install fluidfoam
try: 
    from fluidfoam import readscalar, readvector, readforce
except:
    print("WARNING: fluidfoam not able to be loaded")


class DataLoader():

    def __init__(self, path, verbose=False):
        self.path = path
        self.verbose = verbose

    def to_csv(self, Y, X, time, columns):
        """Output CSV file named by timestamp, e.g. 1000.csv"""
        df = pd.DataFrame(np.concatenate((Y, X), axis=1), columns=columns)
        df.to_csv(str(time) + '.csv', index=False)


class DataLoaderOF(DataLoader):

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

    def load_multiple_timesteps(self, write_interval, num_timesteps, target, cv):

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
    
        params = {'p': p, 'wz': wz, 'pwz': np.stack((p, wz), axis=1)}

        if target == 'drag':
            params['drag'] = self.load_forces()[1].reshape(-1, 1)

        if cv == 'stream': 
            params['stream'] = compute_stream_function(u, v, wz)

        X = np.stack((u, v), axis=-1)
        Y = params[target]
        cv = params[cv]

        return X, Y, cv


class DataLoaderCSV(DataLoader):
    
    def __init__(self, path, verbose=False, prefix='cylinder_t', zwidth=4):
        super().__init__(path, verbose)
        self.prefix = prefix
        self.zwidth = zwidth
    
    def load_xyz(self):
        dfpath = os.path.join(self.path, f'{self.prefix}{str(1).zfill(self.zwidth)}.csv')
        data = pd.read_csv(dfpath)
        x = data["X"].to_numpy()
        y = data["Y"].to_numpy()
        self.num_points = len(x)
        return x, y

    def load_multiple_timesteps(self, write_interval, num_timesteps, target, cv):
        num_pts = self.num_points
        x_labels = ["dudx", "dudy", "dvdx", "dvdy", "vortZ"]
        Y = np.zeros((num_timesteps, num_pts))
        X = np.zeros((num_timesteps, num_pts, len(x_labels)))
        cv = np.zeros((num_timesteps, num_pts))
        
        for i, ts in enumerate(range(write_interval, write_interval*num_timesteps+1, write_interval)):
            dfpath = os.path.join(self.path,f'cylinder_t{str(i+1).zfill(self.zwidth)}.csv')
            data = pd.read_csv(dfpath)
            tke_val = abs(data["TKE"].to_numpy())
            tke_0 = np.where(tke_val <= 1.0e-9)[0]
            tke_val[tke_0] = 1.0e-8
            Y[i, :] = np.log(tke_val)
            X[i, :] = data[x_labels].to_numpy()
            cv[i,:] = data["vortZ"].to_numpy()

        return X, Y, cv


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


def create_sequences(X, Y, window_size=3, overlap=2, field_prediction_type=FPT_GLOBAL):
    """ Create time sequences of a given window size from the input arrays X and Y with specified overlap """
    nt, nsamples, nvars = X.shape
    stride = window_size - overlap
    num_sequences = (nt - window_size) // stride + 1

    X_sequences = np.zeros((num_sequences, window_size, nsamples * nvars))
    if field_prediction_type == FPT_GLOBAL:
        Y_sequences = np.zeros((num_sequences, window_size))
    else:
        Y_sequences = np.zeros((num_sequences, window_size, nsamples))

    for i in range(num_sequences):
        start_index = i * stride
        X_sequences[i] = X[start_index:start_index + window_size].reshape(window_size, nsamples * nvars)
        if field_prediction_type == FPT_GLOBAL:
            Y_sequences[i] = Y[start_index:start_index + window_size].flatten()
        else:
            Y_sequences[i] = Y[start_index:start_index + window_size].reshape(window_size, -1)

    return X_sequences, Y_sequences


if __name__ == "__main__":

    from args import args
    
    dl = DataLoader(args.path)
    
    #X, Y = dl.read_solution('1000')
    #print(X.shape, Y.shape)

    #X, Y = create_sequences(*dl.load_multiple_timesteps(100, 100))
    #print(X.shape, Y.shape)

    x, y = dl.load_xyz()
    X, Y = dl.load_multiple_timesteps(args.write_interval, args.num_timesteps, target=args.target)
    print(X.shape, Y.shape)
