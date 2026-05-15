from welltie.dataset import MLPDataset
from welltie.model import MLPWaveletModel
from utils import plotar_amostras_como_curvas

import torch
import sys
import time
import yaml
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

with open("./parameters.yaml", "r") as yf:
    parameters = yaml.load(yf, Loader=yaml.SafeLoader)

params = parameters["mlp_wavelet"]

SAVE_DIR = WORKDIR / "training"




def train():
    start_time = time.time()
    print("----- Starting MLPWaveletExtractor Train -----")
    print("----- Generating Seismic Random Dataset -----")
    seismicDataset = MLPDataset(params['train_sample'],
                                {"train": True})
    print(f"Generated {len(seismicDataset)} samples")
    mlpwave = MLPWaveletModel(SAVE_DIR,
                         seismicDataset,
                         params,
                         device=device)
    mlpwave.train()

    print("Total training time (DualTask):")
    print(time.time() - start_time)
    print("Loss total (DualTask):")
    print(mlpwave.history["train_loss_total"])
    run()

def run():
    print("----- Starting MLPWaveletExtractor Test -----")
    print("----- Generating Seismic Random Dataset -----")
    dataset_args = {
        "lasdir": LOGS_PATH,
        "syfile": SEGY_FILE,
        "train": False
    }
    seismicDataset = MLPDataset(params['train_sample'],
                                dataset_args)
    print(f"Generated {len(seismicDataset)} samples")
    mlpwave = MLPWaveletModel(SAVE_DIR,
                         seismicDataset,
                         params,
                         device=device)
    mlpwave.load_network(mlpwave.save_dir / mlpwave.state_dict)
    results = mlpwave.run_test()
    plotar_amostras_como_curvas(results["s_"], results["s"], results["w"])



switch = {
    "train": train,
    "run": run
}

if __name__ == "__main__":
    sys.exit(switch[sys.argv[1]]())