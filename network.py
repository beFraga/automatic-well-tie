import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from geophysics import ricker_wavelet
from utils import plot

class DualTaskAE(nn.Module):
    def __init__(self):
        super(DualTaskAE, self).__init__()

        filters = 32
        latent_filters = 8
        wavelet_filters = 8
        kernel_size = 3

        self.encoder = nn.Sequential(
            nn.Conv1d(1, filters, kernel_size=kernel_size, stride=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(filters, filters, kernel_size=kernel_size, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(filters, latent_filters, kernel_size=kernel_size, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.seismic_decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_filters, filters, kernel_size=kernel_size, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(filters, 1, kernel_size=kernel_size, stride=3, padding=1, output_padding=2)
        )

        self.wavelet_branch = nn.Sequential(
            # nn.Conv1d(latent_filters, wavelet_filters, kernel_size=5, stride=1, padding=2),
            # nn.ReLU(inplace=True),
            # nn.Conv1d(wavelet_filters, wavelet_filters, kernel_size=5, stride=1, padding=2),
            # nn.ReLU(inplace=True),
            nn.Conv1d(latent_filters, 1, kernel_size=64, stride=1, padding=10)
        )

        last = self.wavelet_branch[-1]
        wavelet = ricker_wavelet(random.uniform(5, 125), 0.004, 64)
        plot(wavelet)
        wavelet = wavelet.view(1, 1, 64)
        wavelet = wavelet.repeat(1, last.in_channels, 1)
        with torch.no_grad():
            last.weight.copy_(wavelet)
            # last.bias.zero_()
        # for param in last.parameters():
        #     param.requires_grad = False        

    def forward(self, s_noyse):
        latent = self.encoder(s_noyse)
        s_syn = self.seismic_decoder(latent)
        w = self.wavelet_branch(latent)
        w = w / torch.sqrt(torch.sum(w ** 2, dim=-1, keepdim=True) + 1e-8)
        return s_syn, w
    

class SeisAE(nn.Module):
    def __init__(self):
        super(SeisAE, self).__init__()

        filters = 32
        latent_filters = 8
        wavelet_filters = 8
        kernel_size = 3

        self.encoder = nn.Sequential(
            nn.Conv1d(1, filters, kernel_size=kernel_size, stride=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(filters, filters, kernel_size=kernel_size, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(filters, latent_filters, kernel_size=kernel_size, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_filters, filters, kernel_size=kernel_size, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(filters, 1, kernel_size=kernel_size, stride=3, padding=1, output_padding=2)
        )

    def forward(self, s_noyse):
        latent = self.encoder(s_noyse)
        s_syn = self.decoder(latent)
        return s_syn
    
    def encode(self, s):
        return self.encoder(s)
    

class WaveletDecoder(nn.Module):
    def __init__(self):
        super(WaveletDecoder, self).__init__()

        filters = 8
        wavelet_filters = 8

        self.decoder = nn.Sequential(
            nn.Conv1d(filters, wavelet_filters, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(wavelet_filters, wavelet_filters, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(wavelet_filters, 1, kernel_size=5, stride=1, padding=3)
        )

    def forward(self, latent):
        w = self.decoder(latent)
        w = w / torch.sqrt(torch.sum(w ** 2, dim=-1, keepdim=True) + 1e-8)
        return w
    




class TimeShiftPredictor(nn.Module):
    def __init__(self):
        super(TimeShiftPredictor, self).__init__()

        kernel_size=3
        filters = 32
        concat_filter = 16


        stride = 1
        padding = 1 # TODO CALCULAR NOVAMENTE OS DOIS

        self.synthetic_network = nn.Sequential(
            nn.Conv1d(1, filters, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv1d(filters, filters, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True)
        )

        self.truth_network = nn.Sequential(
            nn.Conv1d(1, filters, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv1d(filters, filters, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True)
        )

        self.concat_network = nn.Sequential(
            nn.Conv1d(2 * filters, concat_filter, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv1d(concat_filter, 1, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True)
        )

    def forward(self, s, s_syn):
        conv_s = self.truth_network(s)
        conv_s_synth = self.synthetic_network(s_syn)

        x = torch.cat([s, s_syn], dim=1)

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