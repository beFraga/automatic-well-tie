import torch
import lasio, segyio
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset, random_split
from scipy.signal import savgol_filter

import random

from welltie.geophysics import *
from utils import adjust_data_length, rolling_window, despike

import matplotlib.pyplot as plt

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
    def __init__(self, args, train_ratio=0.7, val_ratio=0.2, batch_size=32, angola=False):
        n_seis = []

        xs, ys, locs = take_coordinates_trace(args["lasdir"], args["syfile"], angola)
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

        test_data = take_well_traces(locs, xs, ys, args["syfile"])
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
    def __init__(self, syfile, lasdir, extdir, _model, train_ratio=0.7, val_ratio=0.2, batch_size=32, train_distortions=100, test_real=False):
        self._model = _model
        self.t = None
        self.og_t = None
        
        well_data = take_well_data(lasdir)
        s, s_w, ts, mask, t2, z_t, w = self.set_dataset(syfile, extdir, well_data, train_distortions)
        
        s    = torch.tensor(s, dtype=torch.float32).unsqueeze(1)
        s_w  = torch.tensor(s_w, dtype=torch.float32).unsqueeze(1)
        ts   = torch.tensor(ts, dtype=torch.float32).unsqueeze(1)
        mask = torch.tensor(mask, dtype=torch.int32).unsqueeze(1).bool()
        t2   = torch.tensor(t2, dtype=torch.float32).unsqueeze(1)
        z_t  = torch.tensor(z_t, dtype=torch.float32).unsqueeze(1)
        w    = torch.tensor(w, dtype=torch.float32).unsqueeze(1)
        self.s      = s
        self.s_w    = s_w
        self.ts     = ts
        self.mask   = mask
        self.t2     = t2
        self.z_t    = z_t
        self.w      = w

        full_dataset = TensorDataset(self.s, self.s_w, self.ts, self.mask)

        N = len(full_dataset)
        n_train = int(train_ratio * N)
        n_val = N - n_train

        self.train_set, self.val_set = random_split(
            full_dataset, [n_train, n_val]
        )

        if test_real:
            s, s_w, mask, t2, z_t, w = self.set_test(syfile, extdir, well_data)
            s      = torch.tensor(s, dtype=torch.float32).unsqueeze(1)
            s_w    = torch.tensor(s_w, dtype=torch.float32).unsqueeze(1)
            mask   = torch.tensor(mask, dtype=torch.int32).unsqueeze(1).bool()
            t2     = torch.tensor(t2, dtype=torch.float32).unsqueeze(1)
            z_t    = torch.tensor(z_t, dtype=torch.float32).unsqueeze(1)
            w      = torch.tensor(w, dtype=torch.float32).unsqueeze(1)
            ts     = torch.tensor(np.zeros((4,1,1)))
        self.test_set = TensorDataset(s, s_w, ts, mask, t2, z_t, w)

        self.train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True)
        self.val_loader   = DataLoader(self.val_set, batch_size=batch_size//2, shuffle=False)
        self.test_loader  = DataLoader(self.test_set, batch_size=1, shuffle=False)
    

    def set_dataset(self, syfile, extdir, well_data, train_distortions):
        traces = []
        traces_wraped = []
        time_shifts = []
        masks = []
        t2s = []
        zw_ts = []
        ws = []

        with segyio.open(syfile, "r", strict=False) as f:
            dt = float(segyio.tools.dt(f)) * 1e-6
        
        t = np.load(extdir / 'seismic_time.npy')
        og_t = np.load(extdir / 'og_seismic_time.npy')
        self.t = t
        self.og_t = og_t
        w = np.load(extdir / 'wavelet.npy')
        for v in well_data:
            RHO = v["rho"]
            #SONIC = v["sonic"]
            Z = v["depth"]
            VP = v["vp"]
            AI = v["ai"]
            T = v["t"]
            loc = np.array(v["loc"])
            # trace = take_well_trace(syfile, loc[0], loc[1])
            #kb = v["kb"]
            #gl = v["gl"]
            start = v["start"]
            print(f"Well: {v["name"]}")
            # repl_vel = 1600
            # water_vel = 1480
            window = 50


            #w = self._model.process(trace).squeeze(0)


            #repl_int = start - kb + gl
            #EGL_time = 2.0 * np.abs(kb) / water_vel
            #twt_water = 2.0 * np.abs(gl + EGL_time) / water_vel
            #repl_time = 2.0 * repl_int / repl_vel
            #t_start = twt_water + repl_time
            # t_start = 0

            #print(f"--- Geometry Report ---")
            #print(f"Water TWT: {twt_water:.4f} s")
            #print(f"Gap TWT:   {EGL_time:.4f} s")
            #print(f"Log Start: {t_start:.4f} s")
            
            rho = despike_and_smooth(RHO, window=window)
            vp = despike_and_smooth(VP, window=window)
            ai = despike_and_smooth(AI, window=window)
            # sonic = despike_and_smooth(SONIC, window=window)

            z = vp * rho
            #vp = 1e6 / sonic

            # scaled_sonic = 0.1524 * np.nan_to_num(sonic) * 1e-6
            # tcum = 2 * np.cumsum(scaled_sonic)
            # tdr = tcum + t_start
            # tdr2 = generate_twt(sonic, Z, start_twt=t_start)

            z_t = np.interp(x=t, xp=T, fp=ai)
            
            rc_t = extract_reflectivity(z_t)
            rc_t = np.nan_to_num(rc_t)

            s = np.convolve(w, rc_t, mode='same')

            """
            trace_ = np.interp(t, og_t, trace)
            end = s.shape[-1] - 50
            trace__ = trace_[:end]
            s_ = s[:end]
            t_ = t[:end]

            [n_trace, n_s] = normalization(torch.tensor(np.array([trace_, s]), dtype=torch.float32))
            plt.plot(trace__, t_, linewidth=1.5, label='Real')
            plt.fill_betweenx(t_, 0, trace__, where=(trace__ > 0), alpha=0.5)
            plt.plot(s_, t_, linewidth=1.5, label='Synth')
            plt.fill_betweenx(t_, 0, s_, where=(s_ > 0), alpha=0.5)
            plt.gca().invert_yaxis()
            plt.title('Traces')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()
            # """
            interval = (T >= 3350) & (T <= 3850)
            T2 = T[interval]
            mask = ~np.isnan(RHO[interval]).astype(bool)
            mask_ = np.interp(x=t, xp=T2, fp=mask).astype(bool)
            
            for _ in range(train_distortions):
                snr = np.random.uniform(5, 20)
                s, _ = add_awgn(s, snr)
                Tw, shift = generate_warped_twt(T2)
                shift = np.interp(t, T2, shift)

                # plt.plot(t, shift)
                # plt.show()
                # plt.plot(np.fft.fft(shift))
                # plt.show()
                zw_t = np.interp(x=t, xp=Tw, fp=ai[interval]) # TW PRECISA SER DEPTH DOMAIN, SHIFT TEM QUE TER O SHAPE DE T PARA SIMULAR
                rcw_t = extract_reflectivity(zw_t)
                rcw_t = np.nan_to_num(rcw_t)

                sw = np.convolve(w, rcw_t, mode='same')

                """
                shift__ = np.interp(T2, t, shift)
                zb = np.interp(x=T2 + shift__, xp=t, fp=zw_t)
                zb_t = np.interp(x=t, xp=T2, fp=zb)
                rcb_t = extract_reflectivity(zb_t)
                rcb_t = np.nan_to_num(rcb_t)
                plt.plot(t, zb_t)
                # plt.plot(t, rcb_t)
                plt.show()
                sb = np.convolve(w, rcb_t, mode='same')
                s_ = s[mask_]
                t_ = t[mask_]
                sw_ = sw[mask_]
                sb_ = sb[mask_]
                plt.plot(s_, t_, linewidth=1.5, label='Original')
                plt.fill_betweenx(t_, 0, s_, where=(s_ > 0), alpha=0.5)
                plt.plot(sw_, t_, linewidth=1.5, label='Warped')
                plt.fill_betweenx(t_, 0, sw_, where=(sw_ > 0), alpha=0.5)
                plt.plot(sb_, t_, '--', linewidth=1.5, label='Back')
                plt.fill_betweenx(t_, 0, sb_, where=(sb_ > 0), alpha=0.5)
                plt.gca().invert_yaxis()
                plt.title('Traces')
                plt.legend()
                plt.grid(alpha=0.3)
                plt.tight_layout()
                plt.show()
                # """
                traces.append(s)
                traces_wraped.append(sw)
                time_shifts.append(shift)
                masks.append(mask_)
                t2s.append(T2)
                zw_ts.append(zw_t)
                ws.append(w)
        

        maxi = max([i.shape[-1] for i in time_shifts])
        maxj = max([j.shape[-1] for j in t2s])
        # print(time_shifts.shape, masks.shape, t2s.shape, zw_ts.shape)
        # print([[i.shape for i in j] for j in [time_shifts, masks, t2s, zw_ts]])
        time_shifts = [np.pad(i, (0, maxi - i.shape[-1])) for i in time_shifts]
        masks       = [np.pad(i, (0, maxi - i.shape[-1])) for i in masks]
        t2s         = [np.pad(i, (0, maxj - i.shape[-1])) for i in t2s]
        zw_ts       = [np.pad(i, (0, maxi - i.shape[-1])) for i in zw_ts]
        traces          = np.array(traces)
        traces_wraped   = np.array(traces_wraped)
        time_shifts     = np.array(time_shifts)
        masks           = np.array(masks).astype(bool)
        t2s             = np.array(t2s)
        zw_ts           = np.array(zw_ts)
        ws              = np.array(ws)
        return traces, traces_wraped, time_shifts, masks, t2s, zw_ts, ws

    def set_test(self, syfile, extdir, well_data):
        traces = []
        traces_syn = []
        masks = []
        t2s = []
        z_ts = []
        ws = []

        with segyio.open(syfile, "r", strict=False) as f:
            dt = float(segyio.tools.dt(f)) * 1e-6

        w = np.load(extdir / 'wavelet.npy')
        for v in well_data:
            RHO = v["rho"]
            Z = v["depth"]
            VP = v["vp"]
            AI = v["ai"]
            T = v["t"]
            loc = np.array(v["loc"])
            trace = take_well_trace(syfile, loc[0], loc[1])
            trace = np.interp(self.t, self.og_t, trace)
            start = v["start"]
            print(f"Well: {v["name"]}")
            window = 50

            #w = self._model.process(trace).squeeze(0)

            # rho = despike_and_smooth(RHO, window=window)
            # vp = despike_and_smooth(VP, window=window)
            ai = despike_and_smooth(AI, window=window)

            # z = rho * vp

            z_t = np.interp(x=self.t, xp=T, fp=ai)
            
            rc_t = extract_reflectivity(z_t)
            rc_t = np.nan_to_num(rc_t)

            s = np.convolve(w, rc_t, mode='same')

            interval = (T >= 3350) & (T <= 3850)
            T2 = T[interval]
            mask = ~np.isnan(RHO[interval]).astype(bool)
            mask_ = np.interp(x=self.t, xp=T2, fp=mask).astype(bool)
            traces.append(trace)
            traces_syn.append(s)
            masks.append(mask_)
            t2s.append(T2)
            z_ts.append(z_t)
            ws.append(w)

        maxi = max([i.shape[-1] for i in masks])
        maxj = max([j.shape[-1] for j in t2s])
        masks       = [np.pad(i, (0, maxi - i.shape[-1])) for i in masks]
        t2s         = [np.pad(i, (0, maxj - i.shape[-1])) for i in t2s]
        z_ts        = [np.pad(i, (0, maxi - i.shape[-1])) for i in z_ts]
        traces          = np.array(traces)
        traces_syn      = np.array(traces_syn)
        masks           = np.array(masks).astype(bool)
        t2s             = np.array(t2s)
        z_ts            = np.array(z_ts)
        ws              = np.array(ws)

        return traces, traces_syn, masks, t2s, z_ts, ws

    def get_loaders(self):
        return self.train_loader, self.val_loader, self.test_loader

    def __len__(self):
        return len(self.s)

    def __getitem__(self, idx):
        return self.s[idx], self.s_syn[idx], self.ts[idx], self.mask[idx]


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
        #loc = list(map(float, las.well['LOC'].value.replace(" ", "").split("X=")[1].split("Y=")))
        loc = list(map(float, las.well["LOC"].value.replace(" ", "").split(",")))
        rho = las['RHOB']
        #sonic = las["DTC"]
        depth = las["DEPTH"]
        ai = las["AI"]
        vp = las["VP"]
        t = las["TIME"]
        #kb = las.well["EKB"].value
        #gl = las.well["EGL"].value
        start = las.well["STRT"].value
        name = las.well["WELL"].value
        value = {
            "rho": rho,
            #"sonic": sonic,
            "depth": depth,
            "loc": loc,
            "ai": ai,
            "vp": vp,
            "t": t,
            #"kb": kb,
            #"gl": gl,
            "start": start,
            "name": name
        }
        values.append(value)
    return values

def take_coordinates_trace(lasdir, syfile, angola):
    locs = []
    for lasfile in lasdir.iterdir():
        las = lasio.read(lasfile)
        if angola:
            loc = list(map(float, las.well["LOC"].value.replace(" ", "").split(","))) # Angola
        else:
            loc = list(map(float, las.well['LOC'].value.replace(" ", "").split("X=")[1].split("Y="))) #F3
        locs.append(loc)
    locs = np.array(locs)
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


def take_well_traces(locs, xs, ys, syfile):
    seismics = []
    with segyio.open(syfile, "r", strict=False) as f:
        for loc in locs:
            distances = np.sqrt((xs - loc[0])**2 + (ys - loc[1])**2)
            closest_trace_idx = np.argmin(distances)
            seismics.append(f.trace[closest_trace_idx])
    return torch.tensor(np.array(seismics), dtype=torch.float32).unsqueeze(1)


def take_well_trace(syfile, x, y):
    with segyio.open(syfile, "r", strict=False) as f:
        xs = np.array([f.header[i][segyio.TraceField.CDP_X] for i in range(f.tracecount)])
        ys = np.array([f.header[i][segyio.TraceField.CDP_Y] for i in range(f.tracecount)])
        distances = np.sqrt((xs - x)**2 + (ys - y)**2)
        closest_trace_idx = np.argmin(distances)
        trace = f.trace[closest_trace_idx]
        return trace

def despike_and_smooth(Y, window=50):
    Y_ = np.nan_to_num(Y, nan=np.nanmean(Y))
    y_sm = np.median(rolling_window(Y_, window), -1)
    y_sm = np.pad(y_sm, window//2, mode='edge')
    y_ = despike(Y_, y_sm, z=2)
    y = savgol_filter(y_, window_length=11, polyorder=2)
    return y