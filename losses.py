import torch


class BaseLoss(object):
    key_names = None
    def __init__(self):
        if self.key_names == None:
            raise NotImplementedError("Losses subclasses must implement `key_names` attribute")

        if 'total' not in self.key_names:
            raise NotImplementedError("The key `total` must be present for backdrop")

class DualTaskLoss(BaseLoss):
    key_names = ('total', 'reconstruction', 'spectral') # normalization

    def __init__(self, parameters):
        self.key_names = DualTaskLoss.key_names
        super().__init__()

        self.alpha = parameters['alpha']
        self.beta = parameters['beta']

    def __call__(self, s, s_rec, w):
        loss_reconstruction = torch.linalg.norm(s - s_rec) ** 2 # ||s - s'|| ^ 2

        loss_spectral = self.spectral_loss(w, s) # alpha * ||F(w') - F(s)|| ^ 2


#        loss_norm = self.beta * torch.abs(torch.linalg.norm(w) - 1) # beta * | ||w'|| - 1 |

        loss_total = loss_reconstruction + loss_spectral

        loss = {
                'total': loss_total,
                'reconstruction': loss_reconstruction,
                'spectral': loss_spectral,
#                'normalization': loss_norm
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
        loss = torch.linalg.norm(torch.abs(torch.abs(Wp)**2 - torch.abs(Sp)**2))

        return self.alpha * loss


class TimeShiftLoss(BaseLoss):
    key_names = ('total')

    def __init__(self):
        self.key_names = TimeShiftLoss.key_names
        super().__init__()

    def __call__(self, ts, ts_syn):
        return torch.linalg.norm(ts - ts_syn)