from welltie.dataset import SeismicDataset, TimeShiftDataset
from welltie.model import DualModel, TimeShiftModel
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

params = parameters["time_shift"]
_params = parameters["dual_task"]

SAVE_DIR = WORKDIR / "training"



def train():
    start_time = time.time()
    print("----- Starting TimeShift Test -----")
    _dataset = SeismicDataset({"syfile": SEGY_FILE, "lasdir": LOGS_PATH, "train_size": _params["train_sample"]})
    _model = DualModel(SAVE_DIR, _dataset, _params)
    _model.load_network(_model.save_dir / _model.state_dict)

    args = {
        "syfile": SEGY_FILE,
        "lasdir": LOGS_PATH,
        "_model": _model
    }

    print("----- Generating TDR dataset -----")
    dataset = TimeShiftDataset(args, batch_size=params["batch_size"], train_distortions=params["train_distortions"])
    print(f"Generated {len(dataset)} samples")

    model = TimeShiftModel(SAVE_DIR, dataset, params)
    model.train()
    
    print("Total training time (DualTask):")
    print(time.time() - start_time)
    print("Loss total (DualTask):")
    print(model.history["train_loss_total"])
    run(dataset=dataset)



def run(dataset=None):
    pass

switch = {
    "train": train,
    "run": run
}

if __name__ == "__main__":
    sys.exit(switch[sys.argv[1]]())