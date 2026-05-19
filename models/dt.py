from welltie.dataset import SeismicDataset
from welltie.model import DualModel

import torch
import sys
import time
import yaml
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WORKDIR = Path().absolute()

#"""
DATASET_PATH = WORKDIR / "datasets" / "angola"
LOGS_PATH = DATASET_PATH / "lasfiles"
SEGY_FILE = DATASET_PATH / "full_stack.sgy"
#"""

"""
DATASET_PATH = WORKDIR / "datasets" / "F3_Demo_2023" / "Rawdata"
WELL_PATH = DATASET_PATH / "Well_data" / "All_wells_RawData"
LOGS_PATH_F = WELL_PATH / "Lasfiles"
DTR_PATH = WELL_PATH / "DT_model"
SEGY_FILE = DATASET_PATH / "Seismic_data.sgy"
#"""

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
    dataset = SeismicDataset({"syfile": SEGY_FILE, "lasdir": LOGS_PATH, "train_size": params["train_sample"]}, batch_size=params['batch_size'], angola=True)

    print(f"Generated {len(dataset)} samples")
    model = DualModel(SAVE_DIR, dataset, params, device=device, state_dict="dualmodel_state_dict_angola.pt")
    model.train()

    print("Total training time (DualTask):")
    print(time.time() - start_time)
    print("Loss total (DualTask):")
    print(model.history["train_loss_total"])

    plt.plot(model.history["train_loss_total"])
    plt.plot(model.history["validation_loss_total"])
    plt.title("Loss")
    plt.legend(["Train", "Validation"], loc="upper right")
    plt.xlabel("Epoch")
    plt.show()

    run(dataset=dataset)



def run(dataset=None):
    print("----- Starting DualTaskAE Test -----")
    if dataset is None:
        print("----- Generating Seismic Noise Dataset -----")
        dataset = SeismicDataset({"syfile": SEGY_FILE, "lasdir": LOGS_PATH, "train_size": params["train_sample"]}, angola=True)

    print(f"Generated {len(dataset)} samples")
    model = DualModel(SAVE_DIR, dataset, params, state_dict="dualmodel_state_dict_angola.pt")
    model.load_network()
    results = model.run_test()
    
    print([f"{k}: {v.shape}" for k,v in results.items()])
    print(results["w"].shape, results["s"].shape)
    plt.plot(results["x"], results["s"][0])
    plt.plot(results["x"], results["s_syn"][0], '--')
    plt.ylabel("Normalized Amplitude")
    plt.xlabel("Time (s)")
    plt.title("Reconstruction")
    plt.legend(["Ground Truth", "Generated"], loc="upper right")
    plt.grid(alpha=0.3)
    plt.show()

    plt.plot(results["x_f"], np.log(results["s_spec"][0]))
    plt.plot(results["x_f"], np.log(results['w_spec'][0]), '--')
    plt.title("Spectre Well-1")
    plt.ylabel("Log-Scale Amplitude")
    plt.xlabel("Frequency (Hz)")
    plt.legend(["Seismic", "Wavelet"], loc="upper right")
    plt.grid(alpha=0.3)
    plt.show()

    t = (np.arange(results["w"].shape[-1]) - results["w"].shape[-1]//2) * results["dt"]
    plt.plot(t, results["w"][0], linewidth=1)
    plt.plot(t, results["w"][1], linewidth=1)
    plt.plot(t, results["w"][2], linewidth=1)
    plt.plot(t, results["w"][3], linewidth=1)
    plt.legend(["Well-1", "Well-2", "Well-3", "Well-4"], loc="upper right")
    plt.ylabel("Amplitude")
    plt.xlabel("Time (s)")
    plt.title("Wavelet")
    plt.grid(alpha=0.3)
    plt.show()

    plt.plot(results["x_f"], np.log(results["w_spec"][0]), linewidth=1)
    plt.plot(results["x_f"], np.log(results["w_spec"][1]), linewidth=1)
    plt.plot(results["x_f"], np.log(results["w_spec"][2]), linewidth=1)
    plt.plot(results["x_f"], np.log(results["w_spec"][3]), linewidth=1)
    plt.legend(["Well-1", "Well-2", "Well-3", "Well-4"], loc="upper right")
    plt.ylabel("Log-Scale Amplitude")
    plt.xlabel("Time (s)")
    plt.title("Spectre")
    plt.grid(alpha=0.3)
    plt.show()


    # t = (np.arange(results["w"].shape[-1]) - results["w"].shape[-1]//2) * results["dt"]
    # t2 = (np.arange(results2["w"].shape[-1]) - results2["w"].shape[-1]//2) * results2["dt"]
    # plt.plot(t, results["w"][0], linewidth=1)
    # plt.plot(t, results["w"][1], linewidth=1)
    # plt.plot(t, results["w"][2], linewidth=1)
    # plt.plot(t, results["w"][3], linewidth=1)
    # plt.plot(t2, results2["w"][0], linewidth=1)
    # plt.plot(t2, results2["w"][1], linewidth=1)
    # plt.plot(t2, results2["w"][2], linewidth=1)
    # plt.plot(t2, results2["w"][3], linewidth=1)
    # plt.legend(["Well-1", "Well-2", "Well-3", "Well-4", "F02-1", "F03-2", "F03-4", "F06-1"], loc="upper right")
    # plt.ylabel("Amplitude")
    # plt.xlabel("Time (s)")
    # plt.title("Wavelet")
    # plt.grid(alpha=0.3)
    # plt.show()


    # plt.plot(results["x_f"], np.log(results["w_spec"][0]), linewidth=1)
    # plt.plot(results["x_f"], np.log(results["w_spec"][1]), linewidth=1)
    # plt.plot(results["x_f"], np.log(results["w_spec"][2]), linewidth=1)
    # plt.plot(results["x_f"], np.log(results["w_spec"][3]), linewidth=1)
    # plt.plot(results2["x_f"], np.log(results2["w_spec"][0]), linewidth=1)
    # plt.plot(results2["x_f"], np.log(results2["w_spec"][1]), linewidth=1)
    # plt.plot(results2["x_f"], np.log(results2["w_spec"][2]), linewidth=1)
    # plt.plot(results2["x_f"], np.log(results2["w_spec"][3]), linewidth=1)
    # plt.legend(["Well-1", "Well-2", "Well-3", "Well-4", "F02-1", "F03-2", "F03-4", "F06-1"], loc="upper right")
    # plt.ylabel("Log-Scale Amplitude")
    # plt.xlabel("Time (s)")
    # plt.title("Spectre")
    # plt.grid(alpha=0.3)
    # plt.show()


switch = {
    "train": train,
    "run": run
}

if __name__ == "__main__":
    sys.exit(switch[sys.argv[1]]())