import torch
import torch.nn.functional as F
from welltie.geophysics import ricker_wavelet

from utils import normalization, apply_ormsby_frequency_domain, plot_2j, plot
from utils_spectrum import get_power_spectra, get_freqs, get_amplitude_spectra, gaussian_smoothing_1d

class BaseLoss(object):
    key_names = None
    def __init__(self):
        if self.key_names == None:
            raise NotImplementedError("Losses subclasses must implement `key_names` attribute")

        if 'total' not in self.key_names:
            raise NotImplementedError("The key `total` must be present for backdrop")

class DualTaskLoss(BaseLoss):
    key_names = ('total', 'reconstruction', 'spectral')

    def __init__(self, parameters):
        self.key_names = DualTaskLoss.key_names
        super().__init__()

        self.alpha = parameters['alpha']

        self.dt = parameters["dt"]
        self.duration = parameters["duration"]
        self.freqs = parameters["freqs"]

        # self.i = 0

    def __call__(self, s, s_rec, spec_w):
        loss_reconstruction = F.mse_loss(s, s_rec) # ||s - s'|| ^ 2

        loss_spectral = self.spectral_loss(spec_w, s) # alpha * ||F(w') - F(s)|| ^ 2


        loss_total = loss_reconstruction + self.alpha * loss_spectral

        loss = {
                'total': loss_total,
                'reconstruction': loss_reconstruction,
                'spectral': loss_spectral,
               }
        return loss


    def pre_train(self, s, s_rec, spec_w):
        # s = normalization(s)
        # s_rec = normalization(s_rec)
        loss_reconstruction = F.mse_loss(s, s_rec) # ||s - s'|| ^ 2

        ricker = ricker_wavelet(30, self.dt, spec_w.shape[-1]).to(spec_w.device)
        # ricker = ricker.reshape(1, 1, -1)

        spec_r = get_amplitude_spectra(ricker, self.duration, self.dt)
        spec_r, _ = apply_ormsby_frequency_domain(spec_r, self.freqs)
        spec_r = spec_r.expand(spec_w.shape[0], -1)

        # spec_r = normalization(spec_r)

        # plot_2j(spec_w[0].detach().numpy(), spec_r[0].detach().numpy())

        loss_total = F.mse_loss(spec_w, spec_r) + loss_reconstruction
        loss = {
            'total': loss_total,
            'reconstruction': loss_reconstruction,
            'spectral': torch.tensor(0.0)
        }
        return loss

    def spectral_loss(self, spec_w, s):
        #spec_s = get_amplitude_spectra(s, duration, self.dt)
        #spec_w = get_amplitude_spectra(w, duration, self.dt)
        spec_s = get_amplitude_spectra(s, self.duration, self.dt)
        # spec_w = get_power_spectra(w, duration, self.dt)


        mean_s = torch.sum(spec_s, dim=0) / spec_s.shape[0]
        #pool_s = moving_average(mean_s, 64)
        pool_s = gaussian_smoothing_1d(mean_s, kernel_size=65, sigma=10)


        filtered_s, _ = apply_ormsby_frequency_domain(pool_s, self.freqs)

        # batch_s = filtered_s.reshape(1, 1, -1)
        batch_s = filtered_s.expand(spec_w.shape[0], -1)
        batch_s = batch_s[..., :spec_w.shape[-1]]

        # if (self.i % 2000 == 0):
        #     plot_2j(batch_s[0].detach().numpy(), norm_w[0].detach().numpy())
        # self.i += 1
        # batch_s = normalization(batch_s)
        loss = F.mse_loss(batch_s, spec_w)

        return loss


class TimeShiftLoss(BaseLoss):
    key_names = ('total', 'smooth', 'mse')

    def __init__(self):
        self.key_names = TimeShiftLoss.key_names
        super().__init__()

    def __call__(self, ts, ts_syn, mask):
        loss_mse = F.mse_loss(ts[mask],ts_syn[mask])
        diff = ts_syn[:,:,1:] - ts_syn[:,:,:-1]
        loss_smooth = torch.mean(torch.abs(diff))

        loss_total = loss_mse + (0.5 * loss_smooth)

        loss = {
            'total': loss_total,
            'smooth': loss_smooth,
            'mse': loss_mse

        }
        return loss


class MLPLoss(BaseLoss):
    key_names = ('total', 'logcosh', 'cosine_similarity')

    def __init__(self):
        self.key_names = MLPLoss.key_names
        super().__init__()

    def __call__(self, y, y_):
        lch = self.logcosh(y, y_)
        cos = torch.nn.CosineSimilarity()
        cs = torch.mean(cos(y, y_))
        total = lch + cs
        loss = {
            'total': total,
            'logcosh': lch,
            'cosine_similarity': cs
        }
        return loss

    def logcosh(self, y, y_):
        return torch.sum(torch.log(torch.cosh(y - y_)))