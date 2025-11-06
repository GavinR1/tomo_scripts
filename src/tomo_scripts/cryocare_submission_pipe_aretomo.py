# Author: Gavin Rice
# Date: 14.11.24
# Description: Script to prepare and submit batch cryocare jobs for tomograms processed with AreTomo. 

import argparse
import numpy as np
import json
import subprocess
import pandas as pd
import pytz
import dateutil

parser = argparse.ArgumentParser(description='Batch cryocare script')
parser.add_argument('--tomo_list', default='tomo_list.txt', type=str, help='Text file containing list of tomograms to denoise.')
parser.add_argument('--path', type=str, help='Directory containing output files from aretomo')
args = parser.parse_args()
tomo_list = args.tomo_list
path = args.path

def make_tomo_list():
    tomo_list = 'for f in *.mdoc; do basename "$f" .mdoc; done > tomo_list.txt'
    subprocess.call(tomo_list, shell=True)

def read_tomo_list(tomo_list):
    print("Reading tomo list")
    with open(tomo_list) as f:
        tomo_list = f.read().splitlines()
        print("Found", len(tomo_list), "tomos")
        print("Tomos found:", tomo_list)
    return tomo_list

def create_dir_structure(tomo, path):
    print("Organizing your data nicely.")
    dir = 'mkdir %s/%s_cryocare' % (path, tomo)
    subprocess.call(dir, shell=True)

def write_configs(tomo, path):
    print('Writing cryocare config files')
    prep_config = {
        "even": [
          "%s/%s_EVN_Vol.mrc" % (path, tomo)
        ],
        "odd": [
          "%s/%s_ODD_Vol.mrc" % (path, tomo)
        ],
        "patch_shape": [
          72,
          72,
          72
        ],
        "num_slices": 1200,
        "split": 0.9,
        "tilt_axis": "Y",
        "n_normalization_samples": 500,
        "path": "cryocare_run/"
        }
    json_prep = json.dumps(prep_config, indent=4)
    with open(f"{path}/{tomo}_cryocare/train_data_config.json", "w") as outfile:
        outfile.write(json_prep)

    train_config = {
        "train_data": "cryocare_run/",
        "epochs": 100,
        "steps_per_epoch": 200,
        "batch_size": 16,
        "unet_kern_size": 3,
        "unet_n_depth": 3,
        "unet_n_first": 16,
        "learning_rate": 0.0004,
        "model_name": "cryocare_model",
        "path": "cryocare_run/"
        }
    json_train = json.dumps(train_config, indent=4)
    with open(f"{path}/{tomo}_cryocare/train_config.json", "w") as outfile:
        outfile.write(json_train)

    predict_config = {
      "path": "cryocare_run/cryocare_model.tar.gz",
      "even":  "%s/%s_EVN_Vol.mrc" % (path, tomo),
      "odd":  "%s/%s_ODD_Vol.mrc" % (path, tomo),
      "n_tiles": [1, 1, 1],
      "output": "%s/%s_cryocare/predict/" % (path, tomo)
        }
    json_predict = json.dumps(predict_config, indent=4)
    with open(f"{path}/{tomo}_cryocare/predict_config.json", "w") as outfile:
        outfile.write(json_predict)
def submit_cryocare(tomo, path):
    print('submitting cryocare jobs to CLEM')
    subprocess.call([
        "sbatch",
        "--chdir",
        f"{path}/{tomo}_cryocare/", "submit_cryocare.sh", # change path if you need to 
        "train_data_config.json",
        "train_config.json",
        "predict_config.json"
    ])


def _main_():
    print("Welcome to autocryocare: aretomo edition. First we will prepare the data then run cryocare on everything")
    #make_tomo_list() # optional to create tomo list from mdoc files in directory
    tomos = read_tomo_list(tomo_list=tomo_list)
    for i in tomos:
        create_dir_structure(tomo=i, path=path)
        write_configs(tomo=i, path=path)
        submit_cryocare(tomo=i, path=path)



if __name__ == "__main__":
    _main_()