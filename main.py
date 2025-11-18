from dataset import timeshift_dataset_generator, MLPDataset, WaveletExtractorDataset, SeismicDataset
from model import DualModel, TimeShiftModel, MLPWaveletModel, SeisAEModel, WaveletDecoderModel
from utils import plotar_amostras_como_curvas, plot, plot_2, plot_2j

import torch
import sys, time, yaml
import numpy as np
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WORKDIR = Path().absolute()
DATASET_PATH = WORKDIR / "datasets" / "F3_Demo_2023" / "Rawdata"
WELL_PATH = DATASET_PATH / "Well_data" / "All_wells_RawData"
LOGS_PATH = WELL_PATH / "Lasfiles"
DTR_PATH = WELL_PATH / "DT_model"
SEGY_FILE = DATASET_PATH / "Seismic_data.sgy"

print("Work directory: %s" % WORKDIR)
print("Dataset directory: %s" % DATASET_PATH)

with open('./parameters.yaml', 'r') as yf:
    parameters = yaml.load(yf, Loader=yaml.SafeLoader)

dual_task_params    = parameters["dual_task"]
mlp_params          = parameters["mlp_wavelet"]
sae_params          = parameters["seis_ae"]
wd_params           = parameters["wavelet_decoder"]

SAVE_DIR = WORKDIR / "training"

def main():
    start_time = time.time()
    print("----- Starting DualTaskAE Train -----")
    seismicDataset = SeismicDataset({"syfile": SEGY_FILE, "train_size": dual_task_params["train_sample"]}) # TODO: passar argumentos corretos
    train_seis_data, val_seis_data, test_seis_data = seismicDataset.get_loaders()
    dualTask = DualModel(SAVE_DIR,
                         train_seis_data,
                         val_seis_data,
                         parameters,
                         device=device)

    dualTask.train()

    print("Total training time (DualTask):")
    print(time.time() - start_time)

    start_time = time.time()
    print("----- Starting TimeShiftPredictor Train -----")
    
    _, w = dualTask.net() #TODO: passar args corretos
    tsDataset = timeshift_dataset_generator() #TODO: passar args corretos
    train_ts_data, val_ts_data, test_ts_data = tsDataset.get_loaders()
    tsPredictor = TimeShiftModel(SAVE_DIR,
                                 train_seis_data,
                                 val_seis_data,
                                 parameters,
                                 device=device)

    tsPredictor.train()
    print("Total training time (TimeShiftPredictor):")
    print(time.time() - start_time)


def trainDT():
    start_time = time.time()
    print("----- Starting DualTaskAE Train -----")
    print("----- Generating Seismic Noise Dataset -----")
    seismicDataset = SeismicDataset({"syfile": SEGY_FILE, "train_size": dual_task_params["train_sample"]})
    print(f"Generated {len(seismicDataset)} samples")
    dualTask = DualModel(SAVE_DIR,
                         seismicDataset,
                         dual_task_params,
                         device=device)
    dualTask.train()

    print("Total training time (DualTask):")
    print(time.time() - start_time)
    print("Loss total (DualTask):")
    print(dualTask.history["train_loss_total"])

def runDT():
    start_time = time.time()
    print("----- Starting DualTaskAE Test -----")
    print("----- Generating Seismic Noise Dataset -----")
    seismicDataset = SeismicDataset({"syfile": SEGY_FILE, "train_size": dual_task_params["train_sample"]})
    print(f"Generated {len(seismicDataset)} samples")
    dualTask = DualModel(SAVE_DIR, seismicDataset, dual_task_params)
    dualTask.load_network(dualTask.save_dir / dualTask.state_dict)
    plot(dualTask.net.wavelet_branch[-1].weight.detach().cpu().numpy()[0,0,:])
    results = dualTask.run_test()
    plotar_amostras_como_curvas(results["s"], results["s_syn"], results["w"])


def trainMLP():
    start_time = time.time()
    print("----- Starting MLPWaveletExtractor Train -----")
    print("----- Generating Seismic Random Dataset -----")
    seismicDataset = MLPDataset(mlp_params['train_sample'],
                                {"train": True})
    print(f"Generated {len(seismicDataset)} samples")
    mlpwave = MLPWaveletModel(SAVE_DIR,
                         seismicDataset,
                         mlp_params,
                         device=device)
    mlpwave.train()

    print("Total training time (DualTask):")
    print(time.time() - start_time)
    print("Loss total (DualTask):")
    print(mlpwave.history["train_loss_total"])

def runMLP():
    print("----- Starting MLPWaveletExtractor Test -----")
    print("----- Generating Seismic Random Dataset -----")
    dataset_args = {
        "lasdir": LOGS_PATH,
        "syfile": SEGY_FILE,
        "train": False
    }
    seismicDataset = MLPDataset(mlp_params['train_sample'],
                                dataset_args)
    print(f"Generated {len(seismicDataset)} samples")
    mlpwave = MLPWaveletModel(SAVE_DIR,
                         seismicDataset,
                         mlp_params,
                         device=device)
    mlpwave.load_network(mlpwave.save_dir / mlpwave.state_dict)
    results = mlpwave.run_test()
    plotar_amostras_como_curvas(results["s_"], results["s"], results["w"])


def trainSAE():
    start_time = time.time()
    print("----- Starting SeisAE Train -----")
    print("----- Generating Seismic Noise Dataset -----")
    seismicDataset = SeismicDataset({"syfile": SEGY_FILE, "train_size": sae_params["train_sample"]})
    print(f"Generated {len(seismicDataset)} samples")
    seisae = SeisAEModel(SAVE_DIR,
                         seismicDataset,
                         sae_params,
                         device=device)
    seisae.train()

    print("Total training time (SeisAE):")
    print(time.time() - start_time)
    print("Loss total (SeisAE):")
    print(seisae.history["train_loss_total"])

def trainWD():
    start_time = time.time()
    print("----- Starting WaveletDecoder Train -----")
    print("----- Generating Seismic Noise Dataset -----")
    seismicDataset = SeismicDataset({"syfile": SEGY_FILE, "train_size": sae_params["train_sample"]})
    print(f"Generated {len(seismicDataset)} samples")
    seisae = SeisAEModel(SAVE_DIR,
                         seismicDataset,
                         sae_params,
                         device=device)
    seisae.load_network(seisae.save_dir / seisae.state_dict)
    seismicDataset = WaveletExtractorDataset({"train_size": wd_params['train_sample'], "syfile": SEGY_FILE, "model": seisae})
    print(f"Generated {len(seismicDataset)} samples")
    wdec = WaveletDecoderModel(SAVE_DIR,
                         seismicDataset,
                         wd_params,
                         device=device)
    wdec.train()

    print("Total training time (SeisAE):")
    print(time.time() - start_time)
    print("Loss total (SeisAE):")
    print(wdec.history["train_loss_total"])

def runWD():
    print("----- Starting WaveletDecoder Test -----")
    print("----- Generating Seismic Noise Dataset -----")
    seismicDataset = SeismicDataset({"syfile": SEGY_FILE, "train_size": sae_params["train_sample"]})
    print(f"Generated {len(seismicDataset)} samples")
    seisae = SeisAEModel(SAVE_DIR,
                         seismicDataset,
                         sae_params,
                         device=device)
    seisae.load_network(seisae.save_dir / seisae.state_dict)
    seismicDataset = WaveletExtractorDataset({"train_size": wd_params['train_sample'], "syfile": SEGY_FILE, "model": seisae})
    print(f"Generated {len(seismicDataset)} samples")
    wdec = WaveletDecoderModel(SAVE_DIR,
                         seismicDataset,
                         wd_params,
                         device=device)
    wdec.load_network(wdec.save_dir / wdec.state_dict)
    r = wdec.run_test()
    plot_2(r['s'], r['w'])


def runSAE():
    print("----- Starting SeismicAE Test -----")
    print("----- Generating Seismic Noise Dataset -----")
    seismicDataset = SeismicDataset({"syfile": SEGY_FILE, "train_size": sae_params["train_sample"]})
    print(f"Generated {len(seismicDataset)} samples")
    seisae = SeisAEModel(SAVE_DIR,
                         seismicDataset,
                         sae_params,
                         device=device)
    seisae.load_network(seisae.save_dir / seisae.state_dict)
    r = seisae.run_test()
    plot_2j(r['s'], r['s_syn'])

    

if __name__ == "__main__":
    print("""
    1 - Treinar tudo
    2 - Carregar tudo
    3 - Treinar dt
    4 - Treinar ts
    5 - Carregar dt
    6 - Carregar ts
    7 - Treinar mlp
    8 - Carregar mlp
    9 - Treinar SAE
    10 - Treinar WD
    11 - Carregar WD
    12 - Carregar SAE
    """)
    i = int(input())
    if i == 1:
        sys.exit(main())
    elif i == 2:
        pass
    elif i == 3:
        sys.exit(trainDT())
    elif i == 5:
        sys.exit(runDT())
    elif i == 7:
        sys.exit(trainMLP())
    elif i == 8:
        sys.exit(runMLP())
    elif i == 9:
        sys.exit(trainSAE())
    elif i == 10:
        sys.exit(trainWD())
    elif i == 11:
        sys.exit(runWD())
    elif i == 12:
        sys.exit(runSAE())