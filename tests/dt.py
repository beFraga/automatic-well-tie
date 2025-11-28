from welltie.dataset import SeismicDataset
from welltie.model import DualModel
from utils import plotar_amostras_como_curvas, plot

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

params = parameters["dual_task"]

SAVE_DIR = WORKDIR / "training"



def train():
    start_time = time.time()
    print("----- Starting DualTaskAE Train -----")
    print("----- Generating Seismic Noise Dataset -----")
    seismicDataset = SeismicDataset({"syfile": SEGY_FILE, "train_size": params["train_sample"]})

    print(f"Generated {len(seismicDataset)} samples")
    dualTask = DualModel(SAVE_DIR, seismicDataset, params, device=device)
    dualTask.train()

    print("Total training time (DualTask):")
    print(time.time() - start_time)
    print("Loss total (DualTask):")
    print(dualTask.history["train_loss_total"])
    run()



def run():
    start_time = time.time()
    print("----- Starting DualTaskAE Test -----")
    print("----- Generating Seismic Noise Dataset -----")
    seismicDataset = SeismicDataset({"syfile": SEGY_FILE, "train_size": params["train_sample"]})

    print(f"Generated {len(seismicDataset)} samples")
    dualTask = DualModel(SAVE_DIR, seismicDataset, params)
    dualTask.load_network(dualTask.save_dir / dualTask.state_dict)
    plot(dualTask.net.wavelet_branch[-1].weight.detach().cpu().numpy()[0,0,:])
    results = dualTask.run_test()
    plotar_amostras_como_curvas(results["s"], results["s_syn"], results["w"])


switch = {
    "train": train,
    "run": run
}

if __name__ == "__main__":
    sys.exit(switch[sys.argv[0]]())