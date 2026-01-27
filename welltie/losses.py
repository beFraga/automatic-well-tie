import random

import torch
import torch.nn.functional as F

from utils import apply_ormsby_frequency_domain, normalization
from utils_spectrum import get_freqs, get_power_spectra
from welltie.geophysics import ricker_wavelet


class BaseLoss(object):
    key_names = None

    def __init__(self):
        if self.key_names == None:
            raise NotImplementedError(
                "Losses subclasses must implement `key_names` attribute"
            )

        if "total" not in self.key_names:
            raise NotImplementedError("The key `total` must be present for backdrop")


class DualTaskLoss(BaseLoss):
    key_names = ("total", "reconstruction", "spectral")

    def __init__(self, parameters):
        self.key_names = DualTaskLoss.key_names
        super().__init__()

        self.alpha = parameters["alpha"]

        self.dt = parameters["dt"]
        self.duration = parameters["duration"]

        self.f_min = 5.0
        self.f_max = 80.0

    def __call__(self, s, s_rec, w):
        loss_reconstruction = F.mse_loss(s, s_rec)  # ||s - s'|| ^ 2

        loss_spectral = self.spectral_loss(w, s)  # alpha * ||F(w') - F(s)|| ^ 2

        loss_total = loss_reconstruction + self.alpha * loss_spectral

        loss = {
            "total": loss_total,
            "reconstruction": loss_reconstruction,
            "spectral": loss_spectral,
        }
        return loss

    def pre_train(self, s, s_rec, w):
        loss_reconstruction = F.mse_loss(s, s_rec)  # ||s - s'|| ^ 2

        ricker = ricker_wavelet(30, self.dt, w.shape[-1]).to(w.device)
        ricker = ricker.reshape(1, 1, -1)
        ricker_batch = ricker.expand(w.shape[0], -1, -1)

        wmax = torch.max(torch.abs(w), dim=-1, keepdim=True)[0] + 1e-8
        rmax = torch.max(torch.abs(ricker_batch), dim=-1, keepdim=True)[0] + 1e-8

        loss_total = F.mse_loss(w / wmax, ricker_batch / rmax) + loss_reconstruction
        loss = {
            "total": loss_total,
            "reconstruction": loss_reconstruction,
            "spectral": torch.tensor(0.0),
        }
        return loss

    def spectral_loss(self, w, s):
        duration = self.duration[-1] - self.duration[0]
        # spec_s = get_amplitude_spectra(s, duration, self.dt)
        # spec_w = get_amplitude_spectra(w, duration, self.dt)
        spec_s = get_power_spectra(s, duration, self.dt)
        spec_w = get_power_spectra(w, duration, self.dt)

        mean_s = torch.sum(spec_s, dim=0) / spec_s.shape[0]
        # pool_s = moving_average(mean_s, 64)
        pool_s = torch.avg_pool1d(mean_s, kernel_size=65, stride=1, padding=32)
        pool_s = torch.avg_pool1d(pool_s, kernel_size=65, stride=1, padding=32)
        pool_s = torch.avg_pool1d(pool_s, kernel_size=65, stride=1, padding=32)

        freqs = get_freqs(duration, self.dt)
        filtered_s, _ = apply_ormsby_frequency_domain(pool_s, freqs)

        norm_s = normalization(filtered_s)
        norm_w = normalization(spec_w)

        batch_s = norm_s.reshape(1, 1, -1)
        batch_s = batch_s.expand(norm_w.shape[0], -1, -1)
        batch_s = batch_s[..., : norm_w.shape[-1]]

        # plot_2j(norm_s[0].detach().numpy(), norm_w[0, 0].detach().numpy())

        loss = F.mse_loss(batch_s, norm_w)

        # mask = (freqs >= self.f_min) & (freqs <= self.f_max)

        # if mask.sum() == 0:
        #     mask[:] = True

        # max_w = torch.max(spec_w[..., mask], dim=-1, keepdim=True)[0] + 1e-8
        # max_s = torch.max(spec_s[..., mask], dim=-1, keepdim=True)[0] + 1e-8

        # norm_w = spec_w / max_w
        # norm_s = spec_s / max_s

        # plot_2j(norm_s[0, 0].detach().numpy(), norm_w[0, 0].detach().numpy())
        # loss = F.mse_loss(norm_s[..., mask], norm_w[..., mask])

        return loss


class TimeShiftLoss(BaseLoss):
    key_names = "total"

    def __init__(self):
        self.key_names = TimeShiftLoss.key_names
        super().__init__()

    def __call__(self, ts, ts_syn):
        return torch.linalg.norm(ts - ts_syn)


class MLPLoss(BaseLoss):
    key_names = ("total", "logcosh", "cosine_similarity")

    def __init__(self):
        self.key_names = MLPLoss.key_names
        super().__init__()

    def __call__(self, y, y_):
        lch = self.logcosh(y, y_)
        cos = torch.nn.CosineSimilarity()
        # cosine similarity in [ -1, 1 ] -> we want a loss that decreases when
        # similarity increases, so use (1 - mean_cosine)
        cs = torch.mean(cos(y, y_))
        cos_loss = 1.0 - cs
        total = lch + cos_loss
        loss = {"total": total, "logcosh": lch, "cosine_similarity": cs}
        return loss

    def logcosh(self, y, y_):
        return torch.sum(torch.log(torch.cosh(y - y_)))
