# Author: Gavin Rice
# Date: 25.08.23
# Description: Unstack a tilt series into individual tilt image micrographs.

import argparse
import numpy as np
import mrcfile
import os

def create_parser():

    argparser = argparse.ArgumentParser(
        description="Unstack tilt series",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    argparser.add_argument("-i", "--input", required=True, help="Path to tiltseries to unstack")
    argparser.add_argument("-o", "--output", required=True, help="Output folder")

    return argparser

def unstack(input, outpth):
    print('Unstacking tilt series: {}'.format(input))
    mrc = mrcfile.open(input, mode="r", permissive=True)
    tilts = mrc.data.shape[0]
    print('Number of tilts found: {}'.format(tilts))
    i = 1
    for tilt in mrc.data:
        mrcfile.write(os.path.join(outpth, f'tilt_{i}.mrc'), tilt, overwrite=True)
        i += 1

def _main_():
    parser = create_parser()
    args = parser.parse_args()
    input = args.input
    outpth = args.output
    os.makedirs(outpth, exist_ok=True)
    unstack(input=input, outpth=outpth)
    print("Success *happy tomo noises*")


if __name__ == "__main__":
    _main_()