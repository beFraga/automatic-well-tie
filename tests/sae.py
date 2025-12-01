from welltie.dataset import SeismicDataset
from welltie.model import SeisAEModel
from utils import plot_2j

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

params = parameters["seis_ae"]

SAVE_DIR = WORKDIR / "training"



def run():
    print("----- Starting SeismicAE Test -----")
    print("----- Generating Seismic Noise Dataset -----")
    seismicDataset = SeismicDataset({"syfile": SEGY_FILE, "train_size": params["train_sample"]})
    print(f"Generated {len(seismicDataset)} samples")
    seisae = SeisAEModel(SAVE_DIR,
                         seismicDataset,
                         params,
                         device=device)
    seisae.load_network(seisae.save_dir / seisae.state_dict)
    r = seisae.run_test()
    plot_2j(r['s'], r['s_syn'])

def train():
    start_time = time.time()
    print("----- Starting SeisAE Train -----")
    print("----- Generating Seismic Noise Dataset -----")
    seismicDataset = SeismicDataset({"syfile": SEGY_FILE, "train_size": params["train_sample"]})
    print(f"Generated {len(seismicDataset)} samples")
    seisae = SeisAEModel(SAVE_DIR,
                         seismicDataset,
                         params,
                         device=device)
    seisae.train()

    print("Total training time (SeisAE):")
    print(time.time() - start_time)
    print("Loss total (SeisAE):")
    print(seisae.history["train_loss_total"])
    run()



switch = {
    "train": train,
    "run": run
}

if __name__ == "__main__":
    sys.exit(switch[sys.argv[1]]())