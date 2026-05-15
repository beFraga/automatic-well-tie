import torch
import torch.nn as nn
from utils import normalization

class DualTaskAE(nn.Module):
    def __init__(self, output_length):
        super(DualTaskAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.seismic_decoder = nn.Sequential(
            nn.ConvTranspose1d(8, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 1, kernel_size=4, stride=2, padding=1),
        )

        self.wavelet_branch = nn.Sequential(
            nn.ConvTranspose1d(8, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(8, output_length)
        )


    def forward(self, s_noyse):
        latent = self.encoder(s_noyse)
        s_syn = self.seismic_decoder(latent)
        w = self.wavelet_branch(latent)
        #w = normalization(w)

        t = s_noyse.shape[-1]
        t2 = s_syn.shape[-1]

        if t2 > t:
            s_syn = s_syn[..., :t]
        elif t2 < t:
            pad = t - t2
            s_syn = nn.functional.pad(s_syn, (0, pad))
            
        return s_syn, w




class TimeShiftPredictor(nn.Module):
    def __init__(self):
        super(TimeShiftPredictor, self).__init__()

        kernel_size=9
        filters = 32
        filters2 = 64
        concat_filter = 16

        self.synthetic_network = nn.Sequential(
            nn.Conv1d(1, filters, kernel_size=kernel_size, padding="same"),
            nn.BatchNorm1d(filters),
            nn.ReLU(inplace=True),
            nn.Conv1d(filters, filters2, kernel_size=kernel_size, padding="same", dilation=2),
            nn.BatchNorm1d(filters2),
            nn.ReLU(inplace=True),
        )

        self.truth_network = nn.Sequential(
            nn.Conv1d(1, filters, kernel_size=kernel_size, padding="same"),
            nn.BatchNorm1d(filters),
            nn.ReLU(inplace=True),
            nn.Conv1d(filters, filters2, kernel_size=kernel_size, padding="same", dilation=2),
            nn.BatchNorm1d(filters2),
            nn.ReLU(inplace=True),
        )

        self.concat_network = nn.Sequential(
            nn.Conv1d(2 * filters2, concat_filter, kernel_size=kernel_size, padding="same"),
            nn.BatchNorm1d(concat_filter),
            nn.ReLU(inplace=True),
            nn.Conv1d(concat_filter, 1, kernel_size=kernel_size, padding="same"),
        )

    def forward(self, s, s_syn):
        conv_s = self.truth_network(s)
        conv_s_syn = self.synthetic_network(s_syn)

        x = torch.cat([conv_s, conv_s_syn], dim=1)

        return self.concat_network(x)
    



class MLPWaveletExtractor(nn.Module):
    def __init__(self):
        super(MLPWaveletExtractor, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(300, 300),
            nn.Tanh(),
            nn.Linear(300, 300),
            nn.Tanh(),
            nn.Linear(300, 200),
            nn.Tanh(),
            nn.Linear(200, 97),
            nn.Tanh(),
            nn.Linear(97, 97)
        )


    def forward(self, x):
        return self.network(x)