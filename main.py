from dataset import seismic_dataset_generator, timeshift_dataset_generator
from model import DualModel, TimeShiftModel
from utils import plot_seismic_and_wavelet, plot_seismic_comparison, plotar_amostras_como_curvas

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
dual_task_params = parameters["dual_task"]

SAVE_DIR = WORKDIR / "training"

def main():
    start_time = time.time()
    print("----- Starting DualTaskAE Train -----")
    seismicDataset = seismic_dataset_generator(LOGS_PATH, SEGY_FILE, dual_task_params['train_sample']) # TODO: passar argumentos corretos: lasdir, syfile, size_train
    train_seis_data, val_seis_data, test_seis_data = seismicDataset.get_loaders()
    dualTask = DualModel(SAVE_DIR,
                         seismicDataset,
                         dual_task_params,
                         device=device)

    dualTask.train()

    print("Total training time (DualTask):")
    print(time.time() - start_time)

    start_time = time.time()
    print("----- Starting TimeShiftPredictor Train -----")
    
    # timeshit ainda não funciona como deveria
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

def runBoth():
    print("NOT IMPLEMENTED YET")
    pass

def trainDT():
    start_time = time.time()
    print("----- Starting DualTaskAE Train -----")
    print("----- Generating Seismic Noise Dataset -----")
    seismicDataset = seismic_dataset_generator(LOGS_PATH, SEGY_FILE, dual_task_params['train_sample'])
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

def trainTS():
    print("NOT IMPLEMENTED YET")
    pass

def runDT():
    start_time = time.time()
    print("----- Starting DualTaskAE Test -----")
    print("----- Generating Seismic Noise Dataset -----")
    seismicDataset = seismic_dataset_generator(LOGS_PATH, SEGY_FILE, dual_task_params['train_sample'])
    print(f"Generated {len(seismicDataset)} samples")
    dualTask = DualModel(SAVE_DIR, seismicDataset, dual_task_params)
    dualTask.load_network(dualTask.save_dir / dualTask.state_dict)
    results = dualTask.run_test()
    plotar_amostras_como_curvas(results["s"], results["s_syn"], results["w"])

def runTS():
    print("NOT IMPLEMENTED YET")
    pass
    

    

if __name__ == "__main__":
    print("""
    1 - Treinar tudo
    2 - Carregar tudo
    3 - Treinar dt
    4 - Treinar ts
    5 - Carregar dt
    6 - Carregar ts
    """)
    i = int(input())
    if i == 1:
        sys.exit(main())
    elif i == 2:
        sys.exit(runBoth)
    elif i == 3:
        sys.exit(trainDT())
    elif i ==4:
        sys.exit(trainTS())
    elif i == 5:
        sys.exit(runDT())
    elif i == 6:
        sys.exit(runTS())