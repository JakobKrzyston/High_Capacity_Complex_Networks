import os
import gc
import itertools
import pickle
import torch
import numpy as np
from typing import Tuple, List, Dict

# Where is the data stored
data_loc = 'path to data'

# Modulation types
MODULATIONS = {
    "8PSK": 0,
    "AM-DSB": 1,
    "AM-SSB": 2,
    "BPSK": 3,
    "CPFSK": 4,
    "GFSK": 5,
    "PAM4": 6,
    "QAM16": 7,
    "QAM64": 8,
    "QPSK": 9,
    "WBFM": 10
}

# Signal-to-Noise Ratios
SNRS = [-20,-18,-16,-14,-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12,14,16,18]

class Data(torch.utils.data.Dataset):

    modulations = MODULATIONS
    snrs = SNRS
    def __init__(self):
        self.n_classes = len(self.modulations)
        self.X, self.y = self.load_data()
        gc.collect()
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """ Load data from file """
        with open(os.path.join(data_loc, "RML2016.10a_dict.pkl"), "rb") as f:
            data = pickle.load(f, encoding="latin1")
        
        X, y = [], []
        for snr in  self.snrs:
            for mod in self.modulations:
                X.append(data[(mod, snr)])

                for i in range(data[(mod, snr)].shape[0]):
                    y.append((mod, snr))

        X = np.vstack(X)
        return X, y

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Load a batch of input and labels """
        x, (mod, snr) = self.X[idx], self.y[idx]
        y = self.modulations[mod]
        x, y = torch.from_numpy(x), torch.tensor(y, dtype=torch.long)
        x = x.to(torch.float).unsqueeze(0)
        return x, y

    def __len__(self) -> int:
        return self.X.shape[0]