import torch
import lasio, segyio
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset, random_split

import random

from welltie.geophysics import *
from utils import adjust_data_length

class BaseDataset(Dataset):
    def __init__(self, x, train_ratio=0.7, val_ratio=0.2, batch_size=32):
        super().__init__()
        x = np.array(x, dtype=np.float32)
        self.data = torch.from_numpy(x)

        if self.data.ndim == 2:
            self.data = self.data.unsqueeze(1)


        indices = torch.randperm(len(self.data))
        N = len(self.data)
        n_train = int(train_ratio * N)
        n_val = int(val_ratio * N)
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
    def __init__(self, args, train_ratio=0.7, val_ratio=0.2, batch_size=32):
        n_seis = []
        with segyio.open(args["syfile"], "r", strict=False) as f:
            idx = np.random.choice(f.tracecount, size=args["train_size"], replace=False)
            seis_subset = np.array(f.trace, dtype=np.float32)[idx]
            self.s      = torch.tensor(seis_subset, dtype=torch.float32).unsqueeze(1)
            for s in seis_subset:
                db_random = np.random.normal(30, 8)
                n_s, shift = add_awgn(s, db_random)
                n_seis.append(n_s)

        self.s_noise  = torch.tensor(np.array(n_seis), dtype=torch.float32).unsqueeze(1)

        full_dataset = TensorDataset(self.s, self.s_noise)

        N = len(full_dataset)
        n_train = int(train_ratio * N)
        n_val = int(val_ratio * N)
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
        return self.s[idx], self.s_noise[idx]


class WaveletExtractorDataset(BaseDataset):
    def __init__(self, args, train_ratio=0.7, val_ratio=0.2, batch_size=32):
        
        with segyio.open(args['syfile'], "r", strict=False) as f:
            idx = np.random.choice(f.tracecount, size=args['train_size'], replace=False)
            seis_subset = np.array(f.trace, dtype=np.float32)[idx]
            self.s = torch.tensor(seis_subset, dtype=torch.float32).unsqueeze(1)
            self.l = args["model"].encode(self.s)
        print(self.s.shape)
        print(self.l.shape)

        full_dataset = TensorDataset(self.s, self.l)

        N = len(full_dataset)
        n_train = int(train_ratio * N)
        n_val = int(val_ratio * N)
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
        return self.s[idx], self.l[idx]


        
class TimeShiftDataset(BaseDataset):
    def __init__(self, s, s_syn, ts, train_ratio=0.7, val_ratio=0.2, batch_size=32):
        super().__init__()
        
        self.s      = torch.tensor(s, dtype=torch.float32).unsqueeze(1)
        self.s_syn  = torch.tensor(s_syn, dtype=torch.float32).unsqueeze(1)
        self.ts     = torch.tensor(ts, dtype=torch.float32).unsqueeze(1)

        full_dataset = TensorDataset(self.s, self.s_syn, self.ts)

        N = len(full_dataset)
        n_train = int(train_ratio * N)
        n_val = int(val_ratio * N)
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
        return self.s[idx], self.s_syn[idx], self.ts[idx]


class MLPDataset(BaseDataset):
    """Dataset para treinar/avaliar um MLP que relaciona sinais sísmicos, wavelets e refletividades.

    Cada amostra é uma tupla (s, w, r):
    - s: traço sísmico (tensor)
    - w: wavelet associada (tensor)
    - r: refletividade (tensor)

    O construtor cria o dataset de treino/val/test a partir de dados sintéticos (quando
    `args['train']` é True) ou a partir de dados reais lidos de arquivos LAS/SEGY.

    Parâmetros:
    - n: número de amostras sintéticas a gerar quando em modo treino
    - args: dicionário com parâmetros necessários (ex.: 'train', 'lasdir', 'syfile', 'train_size', 'model')
    - train_ratio, val_ratio, batch_size: configurações de divisão de conjuntos e loaders
    """
    def __init__(self, n, args, train_ratio=0.7, val_ratio=0.2, batch_size=32):
        self.args = args
        if (args["train"]):
            self.set_train_dataset(n)
        else:
            self.set_dataset()

        full_dataset = TensorDataset(self.s, self.w, self.r)

        N = len(full_dataset)
        n_train = int(train_ratio * N)
        n_val = int(val_ratio * N)
        n_test = N - n_train - n_val

        self.train_set, self.val_set, self.test_set = random_split(
            full_dataset, [n_train, n_val, n_test]
        )

        self.train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True)
        self.val_loader   = DataLoader(self.val_set, batch_size=batch_size//2, shuffle=False)
        self.test_loader  = DataLoader(self.test_set, batch_size=batch_size//2, shuffle=False)
    

    def get_loaders(self):
        """Retorna os DataLoaders para treino, validação e teste.

        Retorno: tupla (train_loader, val_loader, test_loader)
        """
        return self.train_loader, self.val_loader, self.test_loader

    def __len__(self):
        """Retorna o número de amostras no dataset (baseado em `self.s`)."""
        return len(self.s)

    def __getitem__(self, idx):
        """Obtém a amostra no índice `idx`.

        Retorna uma tupla (`s[idx]`, `w[idx]`, `r[idx]`) onde cada elemento é um tensor.
        """
        return self.s[idx], self.w[idx], self.r[idx]
    

    def set_train_dataset(self, n):
        """Gera um dataset sintético para treino.

        Processo:
        - Para cada amostra, escolhe um tamanho de amostra aleatório entre 250 e 400.
        - Gera uma refletividade `r` aleatória normalizada em [-1, 1].
        - Seleciona uma wavelet aleatória via `select_wavelet()`.
        - Constrói o sinal sintético `s = extract_seismic(r, w)` e ajusta o comprimento.
        - Armazena `s` e `w` em listas; `r` fica reservado (inicializado como vazio)

        Parâmetros:
        - n: número de amostras sintéticas a gerar
        """
        ss = []
        ws = []
        for i in range(n):
            sample_size = random.randint(250, 400)

            r = 2 * torch.rand(sample_size) - 1
            r = r / torch.max(torch.abs(r))
            w = self.select_wavelet()
            s = torch.tensor(extract_seismic(r, w))
            s = adjust_data_length(s)
            ss.append(s)
            ws.append(w)
        self.s  = torch.tensor(np.array(ss), dtype=torch.float32)
        self.w  = torch.tensor(np.array(ws), dtype=torch.float32)
        self.r  = torch.empty(self.s.shape, dtype=torch.float32)
    
    def set_dataset(self):
        """Carrega dataset a partir de arquivos LAS (poços) e SEGY (sísmica).

        Processo:
        - Percorre arquivos em `args['lasdir']`, lê cada LAS e extrai localização (LOC), densidade (`RHOB`) e `DT`.
        - Constrói impedância e refletividade a partir de `rho` e `vp` usando funções de `welltie.geophysics`.
        - Para cada localização de poço, encontra a trace SEGY mais próxima (em XY) e adiciona ao conjunto sísmico.
        - Converte listas em tensores `self.s` (sísmica), `self.r` (refletividades). `self.w` é inicializado vazio
          (a ser preenchido posteriormente se necessário).
        """
        lasdir = self.args["lasdir"]
        syfile = self.args["syfile"]
        locs = []
        rs = []
        seismics = []
        for lasfile in lasdir.iterdir():
            las = lasio.read(lasfile)
            loc = list(map(float, las.well['LOC'].value.replace(" ", "").split("X=")[1].split("Y=")))
            locs.append(loc)
            rho = las['RHOB']
            dt = las["DT"]
            print("rho", rho.shape)
            vp = (1 / dt) * 1e6
            z = extract_impedance(rho, vp)
            r = extract_reflectivity(z)
            print("reflexividade: ", r.shape)
            # Ajusta o comprimento da refletividade para o tamanho alvo (usa adjust_data_length)
            # convertendo para numpy para facilitar a construção do array final
            r_adj = adjust_data_length(r)
            rs.append(r_adj.cpu().numpy())
        locs = np.array(locs) * 10
        with segyio.open(syfile, "r", strict=False) as f:
            xs = np.array([f.header[i][segyio.TraceField.CDP_X] for i in range(f.tracecount)])
            ys = np.array([f.header[i][segyio.TraceField.CDP_Y] for i in range(f.tracecount)])
            for loc in locs:
                distances = np.sqrt((xs - loc[0])**2 + (ys - loc[1])**2)
                closest_trace_idx = np.argmin(distances)
                # Ajusta cada trace para o comprimento alvo antes de armazenar
                seis_adj = adjust_data_length(f.trace[closest_trace_idx])
                seismics.append(seis_adj.cpu().numpy())
        # Constrói tensores a partir de arrays numpy já com comprimento uniforme
        self.s = torch.tensor(np.array(seismics), dtype=torch.float32)
        print(self.s.shape)
        self.r = torch.tensor(np.array(rs), dtype=torch.float32)
        self.w = torch.empty(self.s.shape, dtype=torch.float32)

    def select_wavelet(self, dt=0.004, nt=97, device="cpu"):
            """Seleciona e gera uma wavelet aleatória.

        Escolhe aleatoriamente um tipo entre 'ricker', 'gabor', 'ormsby', 'klauder' e 'sinc',
        sorteia parâmetros relevantes (frequências) e retorna a wavelet como um tensor no
        `device` especificado.

        Parâmetros:
        - dt: intervalo de amostragem em segundos
        - nt: número de amostras da wavelet
        - device: dispositivo destino do tensor ('cpu' ou 'cuda')
        """
            choice = random.choice(['ricker', 'gabor', 'ormsby', 'klauder', 'sinc'])
            f = random.uniform(5, 125)  # frequência principal

            if choice == 'ricker':
                w = ricker_wavelet(f, dt, nt)
            elif choice == 'gabor':
                w = gabor_wavelet(f, dt, nt)
            elif choice == 'ormsby':
                f1 = random.uniform(5, 20)
                f2 = f1 + random.uniform(5, 10)
                f3 = f2 + random.uniform(20, 40)
                f4 = f3 + random.uniform(20, 40)
                w = ormsby_wavelet(f1, f2, f3, f4, dt, nt)
            elif choice == 'klauder':
                f1 = random.uniform(5, 20)
                f2 = random.uniform(60, 125)
                t = 0.2
                w = klauder_wavelet(f1, f2, t, dt)
            else:  # 'sinc'
                w = sinc_wavelet(f, dt, nt)

            return w.to(device)
    




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

def timeshift_dataset_generator(vp, rho, s, w, ts):
    z = extract_impedance(rho, vp)
    r = extract_reflectivity(z)
    s_syn = extract_seismic(r, w)

    dataset = TimeShiftDataset(s, s_syn, ts)
    return dataset