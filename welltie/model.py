import pickle
import time

import numpy as np
import torch
from tqdm import tqdm

from utils import apply_ormsby_frequency_domain, normalization, plot
from utils_spectrum import get_amplitude_spectra, get_freqs
from welltie.geophysics import extract_seismic
from welltie.losses import (
    DualTaskLoss,
    MLPLoss,
    TimeShiftLoss,
)
from welltie.network import (
    DualTaskAE,
    MLPWaveletExtractor,
    TimeShiftPredictor,
)


class BaseModel:
    def __init__(self, save_dir, dataset, parameters, device=None):
        self.state_dict = "trained_net_state_dict.pt"
        self.history_file = "history.pkl"
        self.save_dir = save_dir

        self.start_time = time.time()

        self.params = parameters
        self.start_epoch = 0
        self.cur_epoch = self.start_epoch
        self.learning_rate = parameters["learning_rate"]
        self.batch_size = parameters["batch_size"]
        self.max_epochs = parameters["max_epochs"]

        self.train_dataset, self.val_dataset, self.test_dataset = dataset.get_loaders()

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.schedulers = []

    def train_one_epoch(self):
        raise NotImplementedError()

    def validate_training(self):
        raise NotImplementedError()

    def run(self):
        raise NotImplementedError()

    def run_test(self):
        raise NotImplementedError()

    def train(self):
        _div = len(self.train_dataset) / self.batch_size
        _remain = int(len(self.train_dataset) % self.batch_size > 0)
        num_it_per_epoch = _div + _remain

        for e in tqdm(range(self.start_epoch, self.max_epochs)):
            self.train_one_epoch()
            # current_val_loss = self.validate_training()

            if self.schedulers:
                for sche in self.schedulers:
                    sche.step()

            self.cur_epoch += 1

        self.history["elapsed"] = time.time() - self.start_time
        self.save_history()
        self.save_network(self.save_dir / self.state_dict)

    def save_network(self, path):
        torch.save(self.net.state_dict(), path)

    def load_network(self, path):
        self.net.load_state_dict(torch.load(path, map_location=self.device))

    def save_history(self):
        with open(self.save_dir / self.history_file, "wb") as fp:
            pickle.dump(self.history, fp)

    def print_history(self):
        if not self.history:
            print("There is no training history")
            return

        if "elapsed" in self.history:
            elapsed = self.history["elapsed"]


class DualModel(BaseModel):
    def __init__(self, save_dir, dataset, parameters, device=None):
        super().__init__(save_dir, dataset, parameters, device=device)

        self.state_dict = "dualmodel_state_dict.pt"
        self.history_file = "dualmodel_history.pkl"
        self.net = DualTaskAE()
        self.net.to(self.device)

        self.params["loss"]["dt"] = dataset.dt
        self.params["loss"]["duration"] = dataset.duration

        self.pre_train_epochs = self.params["pre_train_epochs"]

        s, _ = next(iter(self.train_dataset))
        s = s.to(self.device)
        latent_shape = self.net.encoder(s).shape[-1]
        torch.manual_seed(0)
        self.unit = torch.randn(8, latent_shape).to(self.device)

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

        self.history = {}
        for key in self.loss.key_names:
            self.history["train_loss_" + key] = []
            self.history["val_loss_" + key] = []

    def train(self):
        _div = len(self.train_dataset) / self.batch_size
        _remain = int(len(self.train_dataset) % self.batch_size > 0)
        num_it_per_epoch = _div + _remain

        for e in tqdm(range(self.start_epoch, self.max_epochs + self.pre_train_epochs)):
            if self.cur_epoch < self.pre_train_epochs:
                self.pre_train()
            else:
                self.train_one_epoch()
            # current_val_loss = self.validate_training()

            if self.schedulers:
                for sche in self.schedulers:
                    sche.step()

            self.cur_epoch += 1

        self.history["elapsed"] = time.time() - self.start_time
        self.save_history()
        self.save_network(self.save_dir / self.state_dict)

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

            s_noise = normalization(s_noise)
            s = normalization(s)

            s_syn, w = self.net(s_noise)

            s_syn = s_syn.to(self.device)
            w = w.to(self.device)

            loss = self.loss(s, s_syn, w)
            loss["total"].backward()
            self.optimizer.step()

            for key in self.loss.key_names:
                loss_numerics[key] += loss[key].item()

        for key in self.loss.key_names:
            _avg_numeric_loss = loss_numerics[key] / count_loop
            self.history["train_loss_" + key].append(_avg_numeric_loss)

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

            s_syn, w = self.net(s_noise)

            s_syn = s_syn.to(self.device)
            w = w.to(self.device)

            loss = self.loss.pre_train(s, s_syn, w)
            loss["total"].backward()
            self.optimizer.step()
            if self.cur_epoch == self.pre_train_epochs - 1 and count_loop == 1:
                plot(w[0, 0].detach().cpu().numpy())

            for key in self.loss.key_names:
                loss_numerics[key] += loss[key].item()

        for key in self.loss.key_names:
            _avg_numeric_loss = loss_numerics[key] / count_loop
            self.history["train_loss_" + key].append(_avg_numeric_loss)

    def validate_training(self):
        loss_numerics = {}
        for key in self.loss.key_names:
            loss_numerics[key] = 0.0

        count_loop = 0
        with torch.no_grad():
            self.net.eval()
            for s, s_noise in self.val_dataset:
                count_loop += 1

                s_syn, w = self.net(s_noise)

                loss = self.loss(s, s_syn, w)

                for key in self.loss.key_names:
                    loss_numerics[key] += loss[key].item()

            for key in self.loss.key_names:
                _avg_numeric_loss = loss_numerics[key] / count_loop
                self.history["train_loss_" + key].append(_avg_numeric_loss)

        return loss_numerics["total"] / count_loop

    def run_test(self):
        result = {"s": [], "s_syn": [], "w": [], "w_spec": [], "x": None}

        self.net.eval()

        with torch.no_grad():
            duration = (
                self.params["loss"]["duration"][-1] - self.params["loss"]["duration"][0]
            )
            result["x"] = get_freqs(duration, self.params["loss"]["dt"])
            for s, _ in self.test_dataset:
                # Move tensores para a GPU
                s = normalization(s).to(self.device)
                s_syn, w = self.net(s)
                w = w.to(self.device)
                spec_w = get_amplitude_spectra(w, duration, self.params["loss"]["dt"])
                filtered_w, _ = apply_ormsby_frequency_domain(spec_w, result["x"])
                new_w = torch.fft.irfft(filtered_w, n=w.shape[-1])
                new_w = torch.roll(new_w, shifts=new_w.shape[-1] // 2, dims=-1)

                result["s_syn"].append(np.squeeze(s_syn.detach().cpu().numpy()))
                result["w"].append(np.squeeze(new_w.detach().cpu().numpy()))
                result["w_spec"].append(np.squeeze(filtered_w.detach().cpu().numpy()))
                result["s"].append(np.squeeze(s.detach().cpu().numpy()))

        result["s"] = np.concatenate(result["s"], axis=0)
        result["s_syn"] = np.concatenate(result["s_syn"], axis=0)
        result["w"] = np.concatenate(result["w"], axis=0)
        result["w_spec"] = np.concatenate(result["w_spec"], axis=0)
        return result


class TimeShiftModel(BaseModel):
    def __init__(self, save_dir, dataset, parameters, device=None):
        super().__init__(save_dir, dataset, parameters, device=device)

        self.state_dict = "timeshift_state_dict.pt"
        self.history_file = "timeshift_history.pkl"
        self.net = TimeShiftPredictor()
        self.net.to(self.device)

        self.loss = TimeShiftLoss

        self.optimizer = torch.optim.Adam(
            params=self.net.parameters(), lr=self.learning_rate
        )

        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            parameters["lr_decay_every_n_epoch"],
            gamma=parameters["lr_decay_rate"],
        )
        self.schedulers = [lr_scheduler]

        self.history = {}
        for key in self.loss.key_names:
            self.history["train_loss_" + key] = []
            self.history["val_loss_" + key] = []

    def train_one_epoch(self):
        self.net.train()

        loss_numerics = {}
        for key in self.loss.key_names:
            loss_numerics[key] = 0.0

        count_loop = 0
        for s, s_syn, ts in self.train_dataset:
            count_loop += 1
            self.optimizer.zero_grad()

            # Move tensores para a GPU
            s = s.to(self.device)
            s_syn = s_syn.to(self.device)
            ts = ts.to(self.device)

            ts_syn = self.net(s, s_syn)

            loss = self.loss(ts, ts_syn)
            loss["total"].backward()
            self.optimizer.step()

            for key in self.loss.key_names:
                loss_numerics[key] += loss[key].item()

        for key in self.loss.key_names:
            _avg_numeric_loss = loss_numerics[key] / count_loop
            self.history["train_loss_" + key].append(_avg_numeric_loss)

    def validate_training(self):
        loss_numerics = {}
        for key in self.loss.key_names:
            loss_numerics[key] = 0.0

        count_loop = 0
        with torch.no_grad():
            self.net.eval()
            for s, s_syn, ts in self.val_dataset:
                count_loop += 1

                # Move tensores para a GPU
                s = s.to(self.device)
                s_syn = s_syn.to(self.device)
                ts = ts.to(self.device)

                ts_syn = self.net(s, s_syn)

                loss = self.loss(ts, ts_syn)

                for key in self.loss.key_names:
                    loss_numerics[key] += loss[key].item()

            for key in self.loss.key_names:
                _avg_numeric_loss = loss_numerics[key] / count_loop
                self.history["train_loss_" + key].append(_avg_numeric_loss)

        return loss_numerics["total"] / count_loop


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

        self.history = {}
        for key in self.loss.key_names:
            self.history["train_loss_" + key] = []
            self.history["val_loss_" + key] = []

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
        with torch.no_grad():
            self.net.eval()
            for s, w_, _ in self.val_dataset:
                count_loop += 1

                s = s.to(self.device)
                w_ = w_.to(self.device)

                w = self.net(s)

                loss = self.loss(w, w_)

                for key in self.loss.key_names:
                    loss_numerics[key] += loss[key].item()

            for key in self.loss.key_names:
                _avg_numeric_loss = loss_numerics[key] / count_loop
                self.history["val_loss_" + key].append(_avg_numeric_loss)

        return loss_numerics["total"] / count_loop

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
