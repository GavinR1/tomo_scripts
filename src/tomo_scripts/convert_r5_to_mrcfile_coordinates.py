## Author: Gavin Rice
## Date: 10.03.25
## Description: Converts coords from relion5 format (origin in center of tomo, coords in angstroms) to bottom left of box origin and pixels. Useful for manual extraction of subvolumes.

import pandas as pd
import starfile
import argparse
from pathlib import Path
import os

def create_parser() -> argparse.ArgumentParser:
    """Creates parser"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--star", type=str, required=True, help="Particles star file from relion5 (eg. run_data.star)")
    parser.add_argument("--out", type=str, required=True, help="Directory for output coords files to be written")
    parser.add_argument("--Xdim", type=int, required=True, help="Tomogram dimension in X for extraction")
    parser.add_argument("--Ydim", type=int, required=True, help="Tomogram dimension in Y for extraction")
    parser.add_argument("--Zdim", type=int, required=True, help="Tomogram dimension in Z for extraction")
    parser.add_argument("--apx", type=float, required=True, help="Pixel size of tomogram to extract from")

    return parser

def convert_coords(star, Xdim, Ydim, Zdim, apx, outdir):
    df = starfile.read(star)
    outdir.mkdir(exist_ok=True, parents=True)
    seen_tomos = set()
    output_files = {}

    for index, row in df.iterrows():
        tomo_name = row['rlnTomoName']
        if tomo_name not in seen_tomos:
            seen_tomos.add(tomo_name)
            output_files[tomo_name] = []
        output_files[tomo_name].append(
            f"{(row['rlnCenteredCoordinateXAngst'] / apx) + (Xdim / 2)} {(row['rlnCenteredCoordinateYAngst'] / apx) + (Ydim / 2)} {(row['rlnCenteredCoordinateZAngst'] / apx) + (Zdim / 2)}\n")

    for tomo_name, lines in output_files.items():
        with open(f"{outdir}/{tomo_name}.coords", "w") as f:
            f.writelines(lines)


def run(args) -> None:
    """Runs the script"""
    star = args.star
    xdim = args.Xdim
    ydim = args.Ydim
    zdim = args.Zdim
    apx = args.apx
    output_dir = Path(args.out)
    convert_coords(star=star, Xdim=xdim, Ydim=ydim, Zdim=zdim, apx=apx, outdir=output_dir)

def _main_() -> None:
    parser = create_parser()
    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    _main_()