from welltie.dataset import SeismicDataset, TimeShiftDataset
from welltie.model import DualModel, TimeShiftModel

import torch
import sys
import time
import yaml
from pathlib import Path

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WORKDIR = Path().absolute()
DATASET_PATH = WORKDIR / "datasets" / "angola"
LOGS_PATH = DATASET_PATH / "lasfiles"
SEGY_FILE = DATASET_PATH / "full_stack.sgy"
EXT_PATH = DATASET_PATH / "extracted_data"
# DATASET_PATH = WORKDIR / "datasets" / "F3_Demo_2023" / "Rawdata"
# WELL_PATH = DATASET_PATH / "Well_data" / "All_wells_RawData"
# LOGS_PATH = WELL_PATH / "Lasfiles"
# DTR_PATH = WELL_PATH / "DT_model"
# SEGY_FILE = DATASET_PATH / "Seismic_data.sgy"

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
    _dataset = SeismicDataset({"syfile": SEGY_FILE, "lasdir": LOGS_PATH, "train_size": _params["train_sample"]}, angola=True)
    _model = DualModel(SAVE_DIR, _dataset, _params, device=device)
    _model.load_network()

    print("----- Generating TDR dataset -----")
    dataset = TimeShiftDataset(SEGY_FILE, LOGS_PATH, EXT_PATH, _model, batch_size=params["batch_size"], train_distortions=params["train_distortions"], test_real=True)
    print(f"Generated {len(dataset)} samples")

    model = TimeShiftModel(SAVE_DIR, dataset, params, device=device)
    model.train()
    
    print("Total training time (DualTask):")
    print(time.time() - start_time)
    print("Loss total (DualTask):")
    print(model.history["train_loss_total"])
    run(dataset=dataset)



def run(dataset=None):
    print("----- Starting TimeShift Test -----")
    if dataset is None:
        print("----- Generating TimeShift Dataset -----")
        _dataset = SeismicDataset({"syfile": SEGY_FILE, "lasdir": LOGS_PATH, "train_size": _params["train_sample"]}, angola=True)
        _model = DualModel(SAVE_DIR, _dataset, _params, device=device)
        _model.load_network()
        dataset = TimeShiftDataset(SEGY_FILE, LOGS_PATH, EXT_PATH, _model, batch_size=params["batch_size"], train_distortions=params["train_distortions"], test_real=True)
        print(f"Generated {len(dataset)} samples")
    model = TimeShiftModel(SAVE_DIR, dataset, params, device=device)
    model.load_network()

    results = model.run_test()
    for k in results.keys():
        print(f"{k}: {results[k].shape}")
    for i in range(results["s"].shape[0]):
        mask = results["mask"][i]
        t_ = results["t"][mask]
        s_ = results["s"][i][mask]
        sw_ = results["s_syn"][i][mask]
        sb_ = results["sb"][i][mask]
        # ts_ = results["ts"][i][mask]
        ts_syn_ = results["ts_syn"][i][mask]
        plt.plot(s_, t_, linewidth=1.5, label='Original')
        plt.fill_betweenx(t_, 0, s_, where=(s_ > 0), alpha=0.5)
        plt.plot(sw_, t_, linewidth=1.5, label='Warped')
        plt.fill_betweenx(t_, 0, sw_, where=(sw_ > 0), alpha=0.5)
        plt.plot(sb_, t_, '--', linewidth=1.5, label='Back')
        plt.fill_betweenx(t_, 0, sb_, where=(sb_ > 0), alpha=0.5)
        plt.gca().invert_yaxis()
        plt.title('Traces')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

        # plt.plot(t_, ts_, linewidth=1.5, label="Original")
        plt.plot(t_, ts_syn_, linewidth=1.5, label="Synth")
        plt.title("Time Shift")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

switch = {
    "train": train,
    "run": run
}

if __name__ == "__main__":
    sys.exit(switch[sys.argv[1]]())