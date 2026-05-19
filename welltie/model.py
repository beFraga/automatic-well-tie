import pickle
import time

import numpy as np
import torch
from tqdm import tqdm

from scipy.interpolate import interp1d

from welltie.network import DualTaskAE, TimeShiftPredictor, MLPWaveletExtractor
from welltie.losses import DualTaskLoss, TimeShiftLoss, MLPLoss
from welltie.geophysics import extract_seismic, extract_reflectivity

from utils import normalization, apply_ormsby_frequency_domain
from utils_spectrum import get_amplitude_spectra, get_freqs, gaussian_smoothing_1d

class BaseModel:
    def __init__(self, save_dir, dataset, parameters, device=None, state_dict="trained_net_state_dict.pt", save_ckpt=False):
        self.state_dict = state_dict
        self.history_file = "history.pkl"
        
        self.save_dir = save_dir
        if not self.save_dir.is_dir():
            self.save_dir.mkdir(parents=True, exist_ok=True)
        assert self.save_dir.is_dir()

        self._save_ckpt = save_ckpt

        self.start_time = time.time()

        self.start_epoch = 0
        self.cur_epoch = 0

        self.params = parameters
        self.learning_rate = parameters["learning_rate"]
        self.batch_size = parameters["batch_size"]
        self.max_epochs = parameters["max_epochs"]

        self.train_dataset, self.val_dataset, self.test_dataset = dataset.get_loaders()

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.early_stopping = None
        self.schedulers = []

        self.best_val_loss = float("inf")

        self.history = {}


    def train_one_epoch(self):
        raise NotImplementedError()

    def validate_training(self):
        raise NotImplementedError()

    def run_test(self):
        raise NotImplementedError()

    def _append_history(self, metrics, prefix):
        for key, value in metrics.items():

            hist_key = f"{prefix}_{key}"

            if hist_key not in self.history:
                self.history[hist_key] = []

            self.history[hist_key].append(float(value))

    def train(self):
        _div = len(self.train_dataset) / self.batch_size
        _remain = int(len(self.train_dataset) % self.batch_size > 0)
        num_it_per_epoch = _div + _remain

        for e in tqdm(range(self.start_epoch, self.max_epochs)):
            self.cur_epoch = e

            self.net.train()
            train_loss = self.train_one_epoch()

            self.net.eval()
            with torch.no_grad():
                val_loss = self.validate_training()

            self._append_history(train_loss, "train_loss")
            self._append_history(val_loss, "validation_loss")

            if self.schedulers:
                for sche in self.schedulers:
                    sche.step()

            if val_loss["total"] < self.best_val_loss - 1e-4:
                self.best_val_loss = val_loss
                self.save_network()

            if self.early_stopping is not None:
                if self.early_stopping.step(val_loss):
                    print("Early stopping triggered.")
                    break
            
            if self._save_ckpt:
                save_freq = max(1, self.max_epochs // 4)
                if (save_freq == 0) or (e == self.max_epochs - 1):
                    ckpt_path = self.save_dir / f"ckpt_epoch{str(e + 1).zfill(3)}.tar"
                    self.save_model_ckpt(ckpt_path, e)

        self.history["elapsed"] = time.time() - self.start_time
        self.save_history()

    def save_network(self):
        torch.save(self.net.state_dict(), self.save_dir / self.state_dict)

    def load_network(self):
        self.net.load_state_dict(torch.load(self.save_dir / self.state_dict, map_location=self.device))

    def save_history(self):
        with open(self.save_dir / self.history_file, "wb") as fp:
            pickle.dump(self.history, fp)

    def print_history(self):
        if not self.history:
            print("There is no training history")
            return

        if "elapsed" in self.history:
            elapsed = self.history["elapsed"]
            print(f"Elapsed time: {elapsed:.2f} seconds")
        
    def save_model_ckpt(self, path, epoch):
        torch.save({
            'epoch': epoch,
            'net_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
        }, path)

    def restore_model_ckpt(self, ckpt_file):
        ckpt = torch.load(ckpt_file)
        self.net.load_state_dict(ckpt['net_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.start_epoch = ckpt['epoch'] + 1
        self.cur_epoch = self.start_epoch
        self.history = ckpt.get("history", self.history)


class DualModel(BaseModel):
    def __init__(self, save_dir, dataset, parameters, device=None, state_dict="dualmodel_state_dict.pt"):
        super().__init__(save_dir, dataset, parameters, device=device, state_dict=state_dict)

        self.history_file = "dualmodel_history.pkl"

        self.duration = dataset.duration[-1] - dataset.duration[0]
        self.full_dur = dataset.duration
        self.dt = dataset.dt
        self.freqs = get_freqs(self.duration, self.dt)
        self.params["loss"]["dt"] = self.dt
        self.params["loss"]["duration"] = self.duration
        self.params["loss"]["freqs"] = self.freqs


        N = int(self.duration / self.dt)
        if N % 2 != 0:
            N += 1
        N /= 2
        N += 1

        self.net = DualTaskAE(int(N))
        self.net.to(self.device)

        self.pre_train_epochs = self.params["pre_train_epochs"]

        self.loss = DualTaskLoss(self.params["loss"])

        self.optimizer = torch.optim.Adam(
            params=self.net.parameters(), lr=self.learning_rate
        )

        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            parameters["lr_decay_every_n_epoch"],
            gamma=parameters["lr_decay_rate"],
        )
        self.schedulers = [lr_scheduler]

    def train(self):
        _div = len(self.train_dataset) / self.batch_size
        _remain = int(len(self.train_dataset) % self.batch_size > 0)
        num_it_per_epoch = _div + _remain

        for e in tqdm(range(self.start_epoch, self.max_epochs)):
            self.cur_epoch = e

            if self.cur_epoch < self.pre_train_epochs:
                train_loss = self.pre_train()
            else:
                train_loss = self.train_one_epoch()

            self.net.eval()
            with torch.no_grad():
                val_loss = self.validate_training()

            self._append_history(train_loss, "train_loss")
            self._append_history(val_loss, "validation_loss")

            if self.schedulers:
                for sche in self.schedulers:
                    sche.step()

            if val_loss["total"] < self.best_val_loss - 1e-4:
                self.best_val_loss = val_loss["total"]
                self.save_network()

            if self.early_stopping is not None:
                if self.early_stopping.step(val_loss):
                    print("Early stopping triggered.")
                    break
            
            if self._save_ckpt:
                save_freq = max(1, self.max_epochs // 4)
                if (save_freq == 0) or (e == self.max_epochs - 1):
                    ckpt_path = self.save_dir / f"ckpt_epoch{str(e + 1).zfill(3)}.tar"
                    self.save_model_ckpt(ckpt_path, e)

        self.history["elapsed"] = time.time() - self.start_time
        self.save_history()

    def train_one_epoch(self):
        self.net.train()

        loss_numerics = {}
        for key in self.loss.key_names:
            loss_numerics[key] = 0.0

        count_loop = 0
        for s, s_noise in self.train_dataset:
            count_loop += 1
            self.optimizer.zero_grad()

            # Move tensores para a GPU
            s_noise = s_noise.to(self.device)
            s = s.to(self.device)

            s_syn, spec_w = self.net(s_noise)

            s_syn = s_syn.to(self.device)
            spec_w = spec_w.to(self.device)

            loss = self.loss(s, s_syn, spec_w)
            loss["total"].backward()
            self.optimizer.step()

            for key in self.loss.key_names:
                loss_numerics[key] += loss[key].item()

        avg_losses = {}
        for key in self.loss.key_names:
            avg_losses[key] = loss_numerics[key] / count_loop

        return avg_losses

    def pre_train(self):
        self.net.train()

        loss_numerics = {}
        for key in self.loss.key_names:
            loss_numerics[key] = 0.0

        count_loop = 0
        for s, s_noise in self.train_dataset:
            count_loop += 1
            self.optimizer.zero_grad()

            # Move tensores para a GPU
            s_noise = s_noise.to(self.device)
            s = s.to(self.device)

            s_noise = normalization(s_noise)
            s = normalization(s)

            s_syn, spec_w = self.net(s_noise)

            s_syn = s_syn.to(self.device)
            spec_w = spec_w.to(self.device)

            loss = self.loss.pre_train(s, s_syn, spec_w)
            loss["total"].backward()
            self.optimizer.step()
            # if self.cur_epoch == self.pre_train_epochs - 1 and count_loop == 1:
                # plot(spec_w[0].detach().numpy())

            for key in self.loss.key_names:
                loss_numerics[key] += loss[key].item()

        avg_losses = {}
        for key in self.loss.key_names:
            avg_losses[key] = loss_numerics[key] / count_loop

        return avg_losses

    def validate_training(self):
        loss_numerics = {}
        for key in self.loss.key_names:
            loss_numerics[key] = 0.0

        count_loop = 0
        for s, s_noise in self.val_dataset:
            count_loop += 1
            
            s = s.to(self.device)
            s_noise = s_noise.to(self.device)

            s_syn, w = self.net(s_noise)

            loss = self.loss(s, s_syn, w)

            for key in self.loss.key_names:
                loss_numerics[key] += loss[key].item()

        avg_losses = {}
        for key in self.loss.key_names:
            avg_losses[key] = loss_numerics[key] / count_loop

        return avg_losses

    def run_test(self):
        result = {"s": [], "s_syn": [], "w": [], "w_spec": [], "x_f": np.array(self.freqs), "s_spec": [], "x": np.array(self.full_dur), "dt": np.array(self.dt)}
        # print(self.freqs)

        self.net.eval()

        with torch.no_grad():
            for s, _ in self.test_dataset:
                # Move tensores para a GPU
                s = s.to(self.device)
                s_syn, spec_w = self.net(s)

                # s = normalization(s)
                # s_syn = normalization(s_syn)

                spec_w = spec_w.to(self.device)
                # spec_w = get_amplitude_spectra(w, self.duration, self.dt)
                spec_s = get_amplitude_spectra(s, self.duration, self.dt)
                # filtered_w, _ = apply_ormsby_frequency_domain(spec_w, self.freqs)
                # print(pad_w)


                pool_s = gaussian_smoothing_1d(spec_s, kernel_size=65, sigma=10)
                

                filtered_s, _ = apply_ormsby_frequency_domain(pool_s, self.freqs)
                filtered_w, _ = apply_ormsby_frequency_domain(spec_w, self.freqs)

                new_w = torch.fft.irfft(spec_w, n=128)
                new_w = torch.roll(new_w, shifts=new_w.shape[-1] // 2, dims=-1)

                result["s_syn"].append(np.squeeze(s_syn.detach().cpu().numpy()))
                result["w"].append(np.squeeze(new_w.detach().cpu().numpy()))
                result["w_spec"].append(np.squeeze(filtered_w.detach().cpu().numpy()))
                result["s_spec"].append(np.squeeze(filtered_s.detach().cpu().numpy()))
                result["s"].append(np.squeeze(s.detach().cpu().numpy()))

        result["s"] = np.concatenate(result["s"], axis=0)
        result["s_syn"] = np.concatenate(result["s_syn"], axis=0)
        result["w"] = np.concatenate(result["w"], axis=0)
        result["w_spec"] = np.concatenate(result["w_spec"], axis=0)
        result["s_spec"] = np.concatenate(result["s_spec"], axis=0)
        return result
    
    def process(self, s):
        s = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
        s = normalization(s).to(self.device)
        _, w = self.net(s)
        w = w.to(self.device)
        spec_w = get_amplitude_spectra(w, self.duration, self.dt)
        filtered_w, _ = apply_ormsby_frequency_domain(spec_w, self.freqs)
        new_w = torch.fft.irfft(filtered_w, n=w.shape[-1])
        new_w = torch.roll(new_w, shifts=new_w.shape[-1] // 2, dims=-1)

        return new_w



class TimeShiftModel(BaseModel):
    def __init__(self, save_dir, dataset, parameters, device=None, state_dict="timeshift_state_dict.pt", save_ckpt=False):
        super().__init__(save_dir, dataset, parameters, device=device, state_dict=state_dict, save_ckpt=save_ckpt)

        self.history_file = "timeshift_history.pkl"
        self.net = TimeShiftPredictor()
        self.net.to(self.device)

        self.loss = TimeShiftLoss()

        self.optimizer = torch.optim.Adam(
            params=self.net.parameters(), lr=self.learning_rate
        )

        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            parameters["lr_decay_every_n_epoch"],
            gamma=parameters["lr_decay_rate"],
        )
        self.schedulers = [lr_scheduler]
        
        self.t = dataset.t

    def train_one_epoch(self):
        self.net.train()

        loss_numerics = {}
        for key in self.loss.key_names:
            loss_numerics[key] = 0.0

        count_loop = 0
        for s, s_syn, ts, mask in self.train_dataset:
            count_loop += 1
            self.optimizer.zero_grad()

            # Move tensores para a GPU
            s = s.to(self.device)
            s_syn = s_syn.to(self.device)
            ts = ts.to(self.device)

            ts_syn = self.net(s, s_syn)

            loss = self.loss(ts, ts_syn, mask)
            loss["total"].backward()
            self.optimizer.step()

            for key in self.loss.key_names:
                loss_numerics[key] += loss[key].item()

        avg_losses = {}
        for key in self.loss.key_names:
            avg_losses[key] = loss_numerics[key] / count_loop

        return avg_losses

    def validate_training(self):
        loss_numerics = {}
        for key in self.loss.key_names:
            loss_numerics[key] = 0.0

        count_loop = 0
        for s, s_syn, ts, mask in self.val_dataset:
            count_loop += 1

            # Move tensores para a GPU
            s = s.to(self.device)
            s_syn = s_syn.to(self.device)
            ts = ts.to(self.device)

            ts_syn = self.net(s, s_syn)

            loss = self.loss(ts, ts_syn, mask)

            for key in self.loss.key_names:
                loss_numerics[key] += loss[key].item()

        avg_losses = {}
        for key in self.loss.key_names:
            avg_losses[key] = loss_numerics[key] / count_loop

        return avg_losses
    
    def run_test(self):
        # Adicionar ts caso rode na versão sintetica
        result = {'s': [], 's_syn': [], 'ts_syn': [], 'mask': [], 'sb': [], 't': self.t, 'ts': []}

        self.net.eval()

        with torch.no_grad():
            # Colocar ts antes de mask caso rode na versão sintetica
            for s, s_syn, ts, mask, t2, z_t, w in self.test_dataset:
                # Move tensores para a GPU
                s = s.to(self.device)
                s_syn = s_syn.to(self.device)
                ts_syn = self.net(s, s_syn)
                # ts_syn = torch.avg_pool1d(ts_syn, 3, 1, 1).detach().cpu().numpy().squeeze(0).squeeze(0)
                ts_syn = ts_syn.detach().cpu().numpy().squeeze(0).squeeze(0)
                t2 = t2.detach().cpu().numpy().squeeze(0).squeeze(0)
                z_t = z_t.detach().cpu().numpy().squeeze(0).squeeze(0)
                w = w.detach().cpu().numpy().squeeze(0).squeeze(0)
                mask = mask.squeeze(0).squeeze(0)
                ts = ts.squeeze(0).squeeze(0)
                shift = np.interp(t2, self.t, ts_syn) # Faço o shift ser em T2
                zb = np.interp(t2 + shift, self.t, z_t) # A impedância no tempo vai para a versão deslocada
                # zb_t = np.interp(self.t, t2, zb)

                zb_t = interp1d(t2, zb, fill_value="extrapolate")(self.t) # Aqui ocorre o deslocamento, pois considero que a versão deslocada está certa

                rcb_t = extract_reflectivity(zb_t) # Valores muito pequenos
                """
                print(zb)
                print(self.t.shape, t2.shape)
                print(zb_t)
                print(rcb_t)
                rcb_t = np.nan_to_num(rcb_t)
                plt.plot(t2, zb)
                plt.plot(self.t[mask], z_t[mask])
                plt.plot(t2 + shift, zb, '--')
                plt.plot(self.t[mask], zb_t[mask], '--')
                plt.plot(self.t, rcb_t, '--')
                plt.legend(['zb por t2', 'z_t', 'zb', 'zb_t'])
                plt.show()
                # """
                sb = np.convolve(w, rcb_t, mode="same")

                result["s"].append(np.squeeze(s.detach().cpu().numpy()))
                result["s_syn"].append(np.squeeze(s_syn.detach().cpu().numpy()))
                result["ts"].append(np.squeeze(ts.detach().cpu().numpy()))
                result["ts_syn"].append(np.squeeze(ts_syn))
                result["mask"].append(np.squeeze(mask.detach().cpu().numpy()))
                result["sb"].append(np.squeeze(sb))

        result["s"] = np.array(result["s"])
        result["s_syn"] = np.array(result["s_syn"])
        result["ts"] = np.array(result["ts"])
        result["ts_syn"] = np.array(result["ts_syn"])
        result["mask"] = np.array(result["mask"])
        result["sb"] = np.array(result["sb"])
        return result


class MLPWaveletModel(BaseModel):
    """
    Modelo que treina uma MLP para extrair wavelets a partir de traços sísmicos.

    Esta classe encapsula a rede `MLPWaveletExtractor`, a função de perda `MLPLoss`,
    o otimizador e o scheduler de taxa de aprendizado. Fornece métodos para:
    - treinar por época (`train_one_epoch`),
    - validar/avaliar durante o treinamento (`validate_training`),
    - executar inferência em dados de teste (`run_test`).

    Atributos principais (herdados/definidos):
    - `self.net`: instância de `MLPWaveletExtractor` usada para inferência/treinamento.
    - `self.loss`: instância de `MLPLoss` que retorna dicionário com vários termos de perda,
      incluindo a chave `"total"`.
    - `self.optimizer`: otimizador (Adam) ligado a `self.net.parameters()`.
    - `self.schedulers`: lista de schedulers (por ex. `StepLR`) aplicada externamente.
    - `self.history`: dicionário que armazena histórico de perdas por chave (treino/val).
    - `self.train_dataset`, `self.val_dataset`, `self.test_dataset`: iteráveis fornecidos pelo dataset.
    """

    def __init__(self, save_dir, dataset, parameters, device=None):
        """
        Inicializa o modelo MLP para extração de wavelets.

        Parâmetros:
        - save_dir (str): diretório onde salvar checkpoints e histórico.
        - dataset: objeto/dicionário com datasets (treino/val/test) já configurados.
        - parameters (dict): dicionário de hiperparâmetros, deve conter ao menos:
            - "lr_decay_every_n_epoch": intervalo de epochs para decaimento de LR
            - "lr_decay_rate": fator de decaimento do LR
        - device (torch.device ou str, opcional): dispositivo para executar o modelo (CPU/GPU).

        Efeitos colaterais:
        - Cria a rede, o otimizador, o scheduler e inicializa `self.history` com chaves
          de perda definidas em `self.loss.key_names`.
        """
        super().__init__(save_dir, dataset, parameters, device=device)

        self.state_dict = "mlpwavelet_state_dict.pt"
        self.history_file = "mlpwavelet_history.pkl"
        self.net = MLPWaveletExtractor()
        self.net.to(self.device)

        self.loss = MLPLoss()

        self.optimizer = torch.optim.Adam(
            params=self.net.parameters(), lr=self.learning_rate
        )

        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            parameters["lr_decay_every_n_epoch"],
            gamma=parameters["lr_decay_rate"],
        )
        self.schedulers = [lr_scheduler]

    def train_one_epoch(self):
        """
        Executa uma passagem de treino (uma época) sobre `self.train_dataset`.

        Fluxo geral:
        - Coloca a rede em modo treino (`self.net.train()`).
        - Itera sobre amostras do dataset: para cada (s, w_, _):
            - move tensores para o dispositivo,
            - zera gradientes do otimizador,
            - faz forward passando `s` pela rede para obter `w` estimado,
            - calcula as várias componentes de perda via `self.loss(w, w_)`,
            - retropropaga a perda total (`loss["total"].backward()`),
            - executa `optimizer.step()` para atualizar pesos.
        - Acumula as perdas numéricas por chave (`self.loss.key_names`) e ao final
          registra a perda média dessa época em `self.history["train_loss_<key>"]`.

        Observações:
        - Espera-se que cada item de `self.train_dataset` seja uma tupla com `s, w_, _`.
        - Não retorna valor; atualiza `self.history`.
        """
        self.net.train()

        loss_numerics = {key: 0.0 for key in self.loss.key_names}
        count_loop = 0

        for s, w_, _ in self.train_dataset:
            count_loop += 1

            s = s.to(self.device)
            w_ = w_.to(self.device)

            self.optimizer.zero_grad()

            w = self.net(s)

            loss = self.loss(w, w_)
            loss["total"].backward()
            self.optimizer.step()

            for key in self.loss.key_names:
                loss_numerics[key] += loss[key].item()

        for key in self.loss.key_names:
            _avg_numeric_loss = loss_numerics[key] / count_loop
            self.history["train_loss_" + key].append(_avg_numeric_loss)

    def validate_training(self):
        """
        Avalia o modelo no conjunto de validação (`self.val_dataset`) sem gradientes.

        Fluxo:
        - Coloca a rede em modo avaliação (`self.net.eval()`).
        - Itera sobre o conjunto de validação, realiza forward e computa as perdas
          (mesma função de perda usada no treino).
        - Acumula as perdas por chave e armazena a média na história (`self.history`).

        Retorno:
        - Retorna a perda média total (float) sobre o conjunto de validação, i.e.
          `loss_numerics["total"] / n_amostras`.

        Observações:
        - Executado sob `torch.no_grad()` para eficiência.
        - Garante que não haja atualização de pesos durante a validação.
        """
        loss_numerics = {}
        for key in self.loss.key_names:
            loss_numerics[key] = 0.0

        count_loop = 0
        for s, w_, _ in self.val_dataset:
            count_loop += 1

            s = s.to(self.device)
            w_ = w_.to(self.device)

            w = self.net(s)

            loss = self.loss(w, w_)

            for key in self.loss.key_names:
                loss_numerics[key] += loss[key].item()

        avg_losses = {}
        for key in self.loss.key_names:
            avg_losses[key] = loss_numerics[key] / count_loop

        return avg_losses

    def run_test(self):
        """
        Executa inferência no conjunto de teste (`self.test_dataset`) e constrói resultados.

        Para cada amostra (s_, w_, r) do dataset de teste:
        - move tensores para o dispositivo,
        - estima a wavelet `w = self.net(s_)`,
        - converte tensores para numpy e armazena `s_` e `w` nos arrays de saída,
        - reconstrói o sinal sísmico `s` chamando `extract_seismic(r, w)` e armazena.

        Retorno:
        - Um dicionário com chaves:
            - "s": array numpy concatenado com os sinais reconstruídos (shape: N x ...),
            - "w": array numpy concatenado com as wavelets estimadas,
            - "s_": array numpy concatenado com os traços de entrada.
        - As entradas são concatenadas ao longo do eixo 0 para formar arrays completos de teste.

        Observações:
        - Executado sob `torch.no_grad()` (modo avaliação).
        - `extract_seismic(r, w)` deve ser responsável por combinar reflectividade `r`
          com a wavelet estimada `w` para produzir o traço sísmico reconstruído `s`.
        """
        result = {"s": [], "w": [], "s_": []}
        with torch.no_grad():
            for s_, w_, r in self.test_dataset:
                s_ = s_.to(self.device)
                w_ = w_.to(self.device)
                r = r.to(self.device)

                w = self.net(s_)
                result["s_"].append(np.squeeze(s_.detach().cpu().numpy()))
                result["w"].append(np.squeeze(w.detach().cpu().numpy()))
                s = extract_seismic(r, w)
                result["s"].append(np.squeeze(s))
        result["s"] = np.concatenate(result["s"], axis=0)
        result["w"] = np.concatenate(result["w"], axis=0)
        result["s_"] = np.concatenate(result["s_"], axis=0)
        return result
