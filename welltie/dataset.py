import torch
import lasio, segyio
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset, random_split

import random

from welltie.geophysics import *
from utils import adjust_data_length, plot_2j, plot, normalization, r_coefficient, plot_axis

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

        xs, ys, locs = take_coordinates_trace(args["lasdir"], args["syfile"])
        with segyio.open(args["syfile"], "r", strict=False) as f:
            self.dt = float(segyio.tools.dt(f)) * 1e-6
            try:
                t0_ms = float(f.header[0][segyio.TraceField.DelayRecordingTime])
            except Exception:
                t0_ms = 0.0

            t0_s = t0_ms * 1e-3
            self.duration = t0_s + np.arange(int(f.samples.size), dtype=np.float64) * self.dt
            seis_subset = take_closers_trace_well(locs, xs, ys, args["syfile"], k=int(args["train_size"]/len(locs)))
            self.s = torch.tensor(seis_subset, dtype=torch.float32).unsqueeze(1)
            for s in seis_subset:
                db_random = np.random.normal(30, 8)
                n_s, shift = add_awgn(s, db_random)
                n_seis.append(n_s)

        self.s_noise  = torch.tensor(np.array(n_seis), dtype=torch.float32).unsqueeze(1)

        full_dataset = TensorDataset(self.s, self.s_noise)

        N = len(full_dataset)
        n_train = int(train_ratio * N)
        n_val = N - n_train

        self.train_set, self.val_set = random_split(
            full_dataset, [n_train, n_val]
        )

        test_data = take_well_trace(locs, xs, ys, args["syfile"])
        zeros = torch.zeros(test_data.shape)
        self.test_set = TensorDataset(test_data, zeros)


        # n_val = int(val_ratio * N)
        # n_test = N - n_train - n_val

        # self.train_set, self.val_set, self.test_set = random_split(
        #     full_dataset, [n_train, n_val, n_test]
        # )

        self.train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True)
        self.val_loader   = DataLoader(self.val_set, batch_size=batch_size//2, shuffle=False)
        self.test_loader  = DataLoader(self.test_set, batch_size=batch_size//2, shuffle=False)
    

    def get_loaders(self):
        return self.train_loader, self.val_loader, self.test_loader

    def __len__(self):
        return len(self.s)

    def __getitem__(self, idx):
        return self.s[idx], self.s_noise[idx]

        
class TimeShiftDataset(BaseDataset):
    def __init__(self, args, train_ratio=0.7, val_ratio=0.2, batch_size=32, train_distortions=100):
        self._model = args["_model"]
        
        well_data = take_well_data(args["lasdir"])
        s, s_w, ts = self.set_dataset(args["syfile"], well_data, train_distortions)
        
        self.s      = torch.tensor(s, dtype=torch.float32).unsqueeze(1)
        self.s_w    = torch.tensor(s_w, dtype=torch.float32).unsqueeze(1)
        self.ts     = torch.tensor(ts, dtype=torch.float32).unsqueeze(1)

        full_dataset = TensorDataset(self.s, self.s_w, self.ts)

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
    

    def set_dataset(self, syfile, well_data, train_distortions):
        traces = []
        traces_wraped = []
        time_shifts = []

        with segyio.open(syfile, "r", strict=False) as f:
            dt = float(segyio.tools.dt(f)) * 1e-6
            xs = np.array([f.header[i][segyio.TraceField.CDP_X] for i in range(f.tracecount)])
            ys = np.array([f.header[i][segyio.TraceField.CDP_Y] for i in range(f.tracecount)])

            for v in well_data:
                rho0 = v["rho"]
                sonic = v["sonic"]
                depth = v["depth"]
                loc = np.array(v["loc"]) * 10
                kb = v["kb"]
                gl = v["gl"]
                start = v["start"]
                repl_vel = 1600
                water_vel = 1480

                distances = np.sqrt((xs - loc[0])**2 + (ys - loc[1])**2)
                closest_trace_idx = np.argmin(distances)
                trace = f.trace[closest_trace_idx]
                w = self._model.process(trace).squeeze(0)

                plot(w)

                sonic = np.nan_to_num(sonic, nan=np.nanmean(sonic))
                rho0 = np.nan_to_num(rho0, nan=np.nanmean(rho0))

                vp0 = 1e6 / sonic
                
                twt = generate_twt(sonic, depth)


                logs = {'vp': vp0, 'rho': rho0}

                time_seis, logs_res = resample_logs_to_seismic(twt, logs, dt)
                vp = logs_res['vp']
                rho = logs_res['rho']

                dist_water = kb - gl
                print(dist_water)
                twt_water = 0 if gl == 0 else 2.0 * dist_water / water_vel

                dist_gap = start - dist_water

                if dist_gap < 0:
                    print("Warning: Log starts ABOVE the seafloor? Check inputs")
                    dist_gap = 0
                
                twt_gap = 2.0 * dist_gap / repl_vel

                t_start = twt_water + twt_gap

                print(f"--- Geometry Report ---")
                print(f"Water TWT: {twt_water:.4f} s")
                print(f"Gap TWT:   {twt_gap:.4f} s")
                print(f"Log Start: {t_start:.4f} s")

                n_samples_water = int(np.round(twt_water / dt))
                n_samples_gap = int(np.round(twt_gap / dt))

                # [WATER] - VP: 1480, RHO: 1030
                vp_water = np.ones(n_samples_water) * water_vel
                rho_water = np.ones(n_samples_water) * 1030

                # [GAP] - VP: 1600, RHO: 2000
                vp_gap = np.ones(n_samples_gap) * vp[0]
                rho_gap = np.ones(n_samples_gap) * rho[0]

                print("Water N: ", n_samples_water)
                print("Gap N: ", n_samples_gap)


                # [WATER] + [GAP] + [LOG]
                full_vp = np.concatenate([vp_water, vp_gap, vp])
                full_rho = np.concatenate([rho_water, rho_gap, rho])
                print("vp: ", vp.shape, full_vp.shape)
                print("rho: ", rho.shape, full_rho.shape)

                plot_2j(vp0, rho0)
                plot_2j(vp, rho)
                plot_2j(full_vp, full_rho)

                s, ts_ = SeismicModel(full_vp, full_rho, w, t0=time_seis[0], dt=dt)
                
                s = s.squeeze(-1)
                #print(depth.shape, twt.shape)
                #print(twt)
                #plot_axis(depth, twt)

                #synth += np.random.randn(*synth.shape) * 0.01
                #s = extract_seismic(synth, w)

                padding = np.zeros(trace.shape[-1] - s.shape[-1])
                s_pad = np.concatenate([s, padding])
                print(s.shape, trace.shape, s_pad.shape)

                print(r_coefficient(s_pad, trace))
                [n_trace, n_s] = normalization(torch.tensor(np.array([trace, s_pad]), dtype=torch.float32))
                plot_2j(n_trace, n_s)
                
                for _ in range(train_distortions):
                    twt_warped, time_shift = generate_warped_twt(twt)
                    time_seis, logs_res_w = resample_logs_to_seismic(twt_warped, logs, dt)
                    vp_w = logs_res_w['vp']
                    rho_w = logs_res_w['rho']
                    s_w, _ = SeismicModel(vp_w, rho_w, w, t0=time_seis[0], dt=dt)
                    #s_w += np.random.randn(*synth_w.shape) * 0.01
                    #s_w = extract_seismic(synth_w, w)

                    traces.append(s)
                    traces_wraped.append(s_w)
                    time_shifts.append(time_shift)

        traces = torch.tensor(np.array(traces), dtype=torch.float32).unsqueeze(1)
        traces_wraped = torch.tensor(np.array(traces_wraped), dtype=torch.float32).unsqueeze(1)
        time_shifts = torch.tensor(np.array(time_shifts), dtype=torch.float32).unsqueeze(1)
        return traces, traces_wraped, time_shifts

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
    


def take_well_data(lasdir):
    values = []
    for lasfile in lasdir.iterdir():
        las = lasio.read(lasfile)
        loc = list(map(float, las.well['LOC'].value.replace(" ", "").split("X=")[1].split("Y=")))
        rho = las['RHOB']
        sonic = las["DT"]
        depth = las["DEPTH"]
        kb = las.well["EKB"].value
        gl = las.well["EGL"].value
        start = las.well["STRT"].value
        value = {
            "rho": rho,
            "sonic": sonic,
            "depth": depth,
            "loc": loc,
            "kb": kb,
            "gl": gl,
            "start": start
        }
        values.append(value)
    return values

def take_coordinates_trace(lasdir, syfile):
    locs = []
    for lasfile in lasdir.iterdir():
        las = lasio.read(lasfile)
        loc = list(map(float, las.well['LOC'].value.replace(" ", "").split("X=")[1].split("Y=")))
        locs.append(loc)
    locs = np.array(locs) * 10
    with segyio.open(syfile, "r", strict=False) as f:
        xs = np.array([f.header[i][segyio.TraceField.CDP_X] for i in range(f.tracecount)])
        ys = np.array([f.header[i][segyio.TraceField.CDP_Y] for i in range(f.tracecount)])
        return xs, ys, locs

def take_closers_trace_well(locs, xs, ys, syfile, k=1000):
    indices = []
    for loc in locs:
        distances = np.sqrt((xs - loc[0])**2 + (ys - loc[1])**2)
        ind = np.argpartition(distances, k)[:k]
        ind = ind[np.argsort(distances[ind])]
        indices.append(ind)
    indices = np.unique(np.array(indices).reshape(-1))
    seismics = []
    with segyio.open(syfile, "r", strict=False) as f:
        seismics = [f.trace[i] for i in indices]
    return np.array(seismics)


def take_well_trace(locs, xs, ys, syfile):
    seismics = []
    with segyio.open(syfile, "r", strict=False) as f:
        for loc in locs:
            distances = np.sqrt((xs - loc[0])**2 + (ys - loc[1])**2)
            closest_trace_idx = np.argmin(distances)
            seismics.append(f.trace[closest_trace_idx])
    return torch.tensor(np.array(seismics), dtype=torch.float32).unsqueeze(1)