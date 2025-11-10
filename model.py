from typing import override
import torch

import time
import pickle
import numpy as np
from tqdm import tqdm

from network import DualTaskAE, TimeShiftPredictor
from losses import DualTaskLoss, TimeShiftLoss


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

    @override
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
            s = s.to(self.device)
            s_noise = s_noise.to(self.device)

            s_syn, w = self.net(s_noise)

            loss = self.loss(s, s_syn, w)
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
        result = {"s": [], "s_syn": [], "w": []}
        with torch.no_grad():
            for s, s_noise in self.test_dataset:
                # Move tensores para a GPU
                s = s.to(self.device)
                s_noise = s_noise.to(self.device)

                s_syn, w = self.net(s_noise)
                result["s_syn"].append(np.squeeze(s_syn.detach().cpu().numpy()))
                result["w"].append(np.squeeze(w.detach().cpu().numpy()))
                result["s"].append(np.squeeze(s.detach().cpu().numpy()))

        result["s"] = np.concatenate(result["s"], axis=0)
        print(result["s"].shape)
        result["s_syn"] = np.concatenate(result["s_syn"], axis=0)
        result["w"] = np.concatenate(result["w"], axis=0)
        print(result["w"].shape)
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
