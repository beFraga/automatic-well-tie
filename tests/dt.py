from welltie.dataset import SeismicDataset
from welltie.model import DualModel
from utils import plotar_amostras_como_curvas, plot, plot_axis

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
    dataset = SeismicDataset({"syfile": SEGY_FILE, "lasdir": LOGS_PATH, "train_size": params["train_sample"]}, batch_size=params['batch_size'])

    print(f"Generated {len(dataset)} samples")
    model = DualModel(SAVE_DIR, dataset, params, device=device)
    model.train()

    print("Total training time (DualTask):")
    print(time.time() - start_time)
    print("Loss total (DualTask):")
    print(model.history["train_loss_total"])
    run(dataset=dataset)



def run(dataset=None):
        
    start_time = time.time()
    print("----- Starting DualTaskAE Test -----")
    if dataset is None:
        dataset = SeismicDataset({"syfile": SEGY_FILE, "lasdir": LOGS_PATH, "train_size": params["train_sample"]})
    print("----- Generating Seismic Noise Dataset -----")

    print(f"Generated {len(dataset)} samples")
    model = DualModel(SAVE_DIR, dataset, params)
    model.load_network(model.save_dir / model.state_dict)
    results = model.run_test()
    plotar_amostras_como_curvas(results["s"], results["s_syn"], results["w"])
    print(results["x"].shape)
    print(results["w_spec"].shape)
    plot_axis(results["x"], results["w_spec"])


switch = {
    "train": train,
    "run": run
}

if __name__ == "__main__":
    sys.exit(switch[sys.argv[1]]())