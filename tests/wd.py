from welltie.dataset import WaveletExtractorDataset, SeismicDataset
from welltie.model import SeisAEModel, WaveletDecoderModel
from welltie.geophysics import power_spectrum
from utils import plot_4

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

sae_params  = parameters["seis_ae"]
wd_params   = parameters["wavelet_decoder"]

SAVE_DIR = WORKDIR / "training"



def train():
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
    run()

def run():
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
    plot_4(r['s'], r['w'], power_spectrum(r['s']), power_spectrum(r['w']))



switch = {
    "train": train,
    "run": run
}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Use: python -m tests.wd (train|run)")
        sys.exit(1)

    action = sys.argv[1]
    if action not in switch.keys():
        print("Use: python -m tests.wd (train|run)")
        sys.exit(1)
    else:
        sys.exit(switch[action]())