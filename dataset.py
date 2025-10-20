import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset, random_split
from geophysics import *
import lasio, segyio
import pandas as pd
import numpy as np

class BaseDataset(Dataset):
    def __init__(self, x, train_ratio=0.7, val_ratio=0.2, batch_size=32):
        super().__init__()
        x = np.array(x, dtype=np.float32)
        self.data = torch.from_numpy(x)

        if self.data.ndim == 2:
            self.data = self.data.unsqueeze(1)


        indices = torch.randperm(len(self.data))
        N = len(self.data)
        n_train = int(0.7 * N)
        n_val = int(0.2 * N)
        n_test = N - n_train - n_val

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train+n_val]
        test_idx = indices[n_train+n_val:]

        self.train_set = Subset(self.data, train_idx)
        self.val_set = Subset(self.data, val_idx)
        self.test_set = Subset(self.data, test_idx)

        self.train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True)
        self.val_loader   = DataLoader(self.val_set, batch_size=batch_size//2, shuffle=False)
        self.test_loader  = DataLoader(self.test_set, batch_size=batch_size//2, shuffle=False)
    
    def get_loaders(self):
        return self.train_loader, self.val_loader, self.test_loader

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return x



class SeismicDataset(BaseDataset):
    def __init__(self, s, s_noise, train_ratio=0.7, val_ratio=0.2, batch_size=32):
        
        self.s      = torch.tensor(np.array(s), dtype=torch.float32).unsqueeze(1)
        self.s_noise  = torch.tensor(np.array(s_noise), dtype=torch.float32).unsqueeze(1)

        full_dataset = TensorDataset(self.s, self.s_noise)

        N = len(full_dataset)
        n_train = int(0.7 * N)
        n_val = int(0.2 * N)
        n_test = N - n_train - n_val

        self.train_set, self.val_set, self.test_set = random_split(
            full_dataset, [n_train, n_val, n_test]
        )

        self.train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True)
        self.val_loader   = DataLoader(self.val_set, batch_size=batch_size//2, shuffle=False)
        self.test_loader  = DataLoader(self.test_set, batch_size=batch_size//2, shuffle=False)
    

    def get_loaders(self):
        return self.train_loader, self.val_loader, self.test_loader

    def __len__(self):
        return len(self.s)

    def __getitem__(self, idx):
        x = self.s[idx], self.s_noise[idx]

        
class TimeShiftDataset(BaseDataset):
    def __init__(self, s, s_syn, ts, train_ratio=0.7, val_ratio=0.2, batch_size=32):
        super().__init__()
        
        self.s      = torch.tensor(s, dtype=torch.float32).unsqueeze(1)
        self.s_syn  = torch.tensor(s_syn, dtype=torch.float32).unsqueeze(1)
        self.ts     = torch.tensor(ts, dtype=torch.float32).unsqueeze(1)

        full_dataset = TensorDataset(self.s, self.s_syn, self.ts)

        N = len(full_dataset)
        n_train = int(0.7 * N)
        n_val = int(0.2 * N)
        n_test = N - n_train - n_val

        self.train_set, self.val_set, self.test_set = random_split(
            full_dataset, [n_train, n_val, n_test]
        )

        self.train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True)
        self.val_loader   = DataLoader(self.val_set, batch_size=batch_size//2, shuffle=False)
        self.test_loader  = DataLoader(self.test_set, batch_size=batch_size//2, shuffle=False)
    

    def get_loaders(self):
        return self.train_loader, self.val_loader, self.test_loader

    def __len__(self):
        return len(self.s)

    def __getitem__(self, idx):
        return
        return self.s[idx], self.s_syn[idx], self.ts[idx]





def seismic_well_dataset_generator(lasdir, syfile, distortions):
    locs = []
    seismics = []
    for lasfile in lasdir.iterdir():
        las = lasio.read(lasfile)
        loc = list(map(float, las.well['LOC'].value.replace(" ", "").split("X=")[1].split("Y=")))
        locs.append(loc)
    locs = np.array(locs) * 10
    with segyio.open(syfile, "r", strict=False) as f:
        xs = np.array([f.header[i][segyio.TraceField.CDP_X] for i in range(f.tracecount)])
        ys = np.array([f.header[i][segyio.TraceField.CDP_Y] for i in range(f.tracecount)])
        for loc in locs:
            distances = np.sqrt((xs - loc[0])**2 + (ys - loc[1])**2)
            closest_trace_idx = np.argmin(distances)
            seismics.append(f.trace[closest_trace_idx])
    n_seis = []
    for s in seismics:
        s = np.array(s, dtype=np.float32)
        for i in range(distortions):
            db_random = np.random.normal(30, 8)
            n_s, shift = add_awgn(s, db_random)
            n_seis.append(n_s)
    return SeismicDataset(n_seis)



def seismic_dataset_generator(lasdir, syfile, size_train):
    n_seis = []
    seis = []
    with segyio.open(syfile, "r", strict=False) as f:
        idx = np.random.choice(f.tracecount, size=size_train, replace=False)
        seis_subset = np.array(f.trace, dtype=np.float32)[idx]
        for s in seis_subset:
            db_random = np.random.normal(30, 8)
            n_s, shift = add_awgn(s, db_random)
            n_seis.append(n_s)
            seis.append(s)
    return SeismicDataset(seis, n_seis)

def timeshift_dataset_generator(vp, rho, s, w, ts):
    z = extract_impedance(rho, vp)
    r = extract_reflectivity(z)
    s_syn = extract_seismic(r, w)

    dataset = TimeShiftDataset(s, s_syn, ts)
    return dataset