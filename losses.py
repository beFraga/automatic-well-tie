import torch
import random
from geophysics import ricker_wavelet

class BaseLoss(object):
    key_names = None
    def __init__(self):
        if self.key_names == None:
            raise NotImplementedError("Losses subclasses must implement `key_names` attribute")

        if 'total' not in self.key_names:
            raise NotImplementedError("The key `total` must be present for backdrop")

class DualTaskLoss(BaseLoss):
    key_names = ('total', 'reconstruction', 'spectral', 'smooth')

    def __init__(self, parameters):
        self.key_names = DualTaskLoss.key_names
        super().__init__()

        self.sigma = parameters['sigma']
        self.alpha = parameters['alpha']
        self.beta = parameters['beta']
        self.gamma = parameters['gamma']

    def __call__(self, s, s_rec, w):
        loss_reconstruction = self.sigma * torch.mean((s - s_rec) ** 2) # sigma * ||s - s'|| ^ 2

        loss_spectral = self.spectral_loss(w, s) # alpha * ||F(w') - F(s)|| ^ 2

        loss_smooth = self.gamma * torch.mean((w[:,:,1:] - w[:,:,:-1]) ** 2) # gamma * sum(w_i - w_i-1) ^ 2
        
        loss_band = self.band_limited_loss(w)

        loss_total = loss_reconstruction + loss_spectral + loss_smooth + loss_band

        loss = {
                'total': loss_total,
                'reconstruction': loss_reconstruction,
                'spectral': loss_spectral,
                'smooth': loss_smooth
               }
        
        return loss


    def spectral_loss(self, w, s):
        Wf = torch.fft.rfft(w, dim=-1)
        Sf = torch.fft.rfft(s, dim=-1)

        # converte em módulo ou potência
        Wp = torch.abs(Wf)**2
        Sp = torch.abs(Sf)**2

        # normaliza cada batch individualmente (para evitar dominância de amplitude)
        Wp = Wp / (torch.max(Wp, dim=-1, keepdim=True).values + 1e-8)
        Sp = Sp / (torch.max(Sp, dim=-1, keepdim=True).values + 1e-8)

        # reamostra para o mesmo tamanho (wavelet e sísmica podem ter tamanhos diferentes)
        if Wp.shape[-1] != Sp.shape[-1]:
            Sp = torch.nn.functional.interpolate(
                Sp, size=Wp.shape[-1], mode="linear", align_corners=False
            ).squeeze(1)

        # loss espectral: ||F(w') - F(s)|| ^ 2
        #loss = torch.linalg.norm(torch.abs(torch.abs(Wp)**2 - torch.abs(Sp)**2))
        loss = torch.mean((torch.log(Wp + 1e-8) - torch.log(Sp + 1e-8)) ** 2)

        return self.alpha * loss
    
    def band_limited_loss(self, w, dt=0.004, f_low=5, f_high=60):
        Wf = torch.fft.rfft(w, dim=-1)
        freqs = torch.fft.rfftfreq(w.shape[-1], d=dt).to(w.device)

        mask = ((freqs < f_low) | (freqs > f_high)).float()
        outside_energy = torch.mean((torch.abs(Wf) * mask) ** 2)
        return self.beta * outside_energy


class SeisAELoss(BaseLoss):
    key_names = ['total']

    def __init__(self):
        self.key_names = SeisAELoss.key_names
        super().__init__()

    def __call__(self, s, s_rec):
        loss_total = torch.norm((s - s_rec), p=2)**2 # ||s - s'|| ^ 2

        loss = {
                'total': loss_total,
               }
        
        return loss


class WaveletDecoderLoss(BaseLoss):
    key_names = ('total', 'spectral', 'smooth')

    def __init__(self, parameters):
        self.key_names = WaveletDecoderLoss.key_names
        super().__init__()

        self.sigma = parameters['sigma']
        self.alpha = parameters['alpha']
        self.beta = parameters['beta']
        self.gamma = parameters['gamma']

    def __call__(self, w, s):
        loss_spectral = self.spectral_loss(w, s) # alpha * ||F(w') - F(s)|| ^ 2

        loss_smooth = self.gamma * torch.mean((w[:,:,1:] - w[:,:,:-1])**2) # gamma * sum(w_i - w_i-1) ^ 2
        
        loss_band = self.band_limited_loss(w)

        loss_total = loss_spectral + loss_smooth + loss_band

        loss = {
                'total': loss_total,
                'spectral': loss_spectral,
                'smooth': loss_smooth
               }
        
        return loss

    def supervised(self, w):
        ricker = ricker_wavelet(random.uniform(5, 125), 0.004, w.shape[-1])
        loss_reconstruction = self.sigma * torch.norm((w - ricker), p=2)**2 # sigma * ||s - s'|| ^ 2

        loss = {
            'total': loss_reconstruction,
            'spectral': torch.tensor(0.0),
            'smooth': torch.tensor(0.0)
        }

        return loss

    def spectral_loss(self, w, s):
        Wf = torch.fft.rfft(w, dim=-1)
        Sf = torch.fft.rfft(s, dim=-1)

        Wp = torch.abs(Wf)
        Sp = torch.abs(Sf)

        Wp = torch.clamp(Wp, min=1e-8)
        Sp = torch.clamp(Sp, min=1e-8)

        Wp = Wp / (torch.sum(Wp, dim=-1, keepdim=True) + 1e-8)
        Sp = Sp / (torch.sum(Sp, dim=-1, keepdim=True) + 1e-8)

        if Wp.shape[-1] != Sp.shape[-1]:
            Sp = torch.nn.functional.interpolate(
                Sp, size=Wp.shape[-1], mode="linear", align_corners=False
            ).squeeze(1)

        loss = torch.mean((torch.log(Wp + 1e-8) - torch.log(Sp + 1e-8))**2)

        return self.alpha * loss
    
    def band_limited_loss(self, w, dt=0.004, f_low=5, f_high=60):
        Wf = torch.fft.rfft(w, dim=-1)
        freqs = torch.fft.rfftfreq(w.shape[-1], d=dt).to(w.device)

        mask = ((freqs < f_low) | (freqs > f_high)).float()
        outside_energy = torch.mean((torch.abs(Wf) * mask) ** 2)
        return self.beta * outside_energy


class TimeShiftLoss(BaseLoss):
    key_names = ('total')

    def __init__(self):
        self.key_names = TimeShiftLoss.key_names
        super().__init__()

    def __call__(self, ts, ts_syn):
        return torch.linalg.norm(ts - ts_syn)



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