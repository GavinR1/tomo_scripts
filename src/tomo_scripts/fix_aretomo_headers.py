import argparse
import mrcfile
import numpy as np
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fix headers of AreTomo tomograms using CryoFlows reference."
    )
    parser.add_argument(
        "--in_cryoflows",
        type=str,
        required=True,
        help="Path to input CryoFlows tomogram (.mrc) used as header reference",
    )
    parser.add_argument(
        "--in_aretomo",
        type=str,
        required=True,
        help="Path to input AreTomo tomogram (.mrc) whose data will be preserved",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=False,
        help="Output file path for header-fixed tomogram (default: adds _headerfix to AreTomo name)",
    )
    return parser.parse_args()


def headerfix(cryoflows_tomo, aretomo_tomo, output_tomo):
    with mrcfile.open(cryoflows_tomo, permissive=True) as cf, \
         mrcfile.open(aretomo_tomo, permissive=True) as ar:

        data = ar.data.copy()

        with mrcfile.new(output_tomo, overwrite=True) as mrc_out:
            if ar.header.mode == 1:
                data = data.astype(np.int16)
            elif ar.header.mode == 2:
                data = data.astype(np.float32)
            elif ar.header.mode == 0:
                data = data.astype(np.int8)

            mrc_out.set_data(data)
            np.copyto(mrc_out.header, ar.header)

            mrc_out.header.mapc = cf.header.mapc
            mrc_out.header.mapr = cf.header.mapr
            mrc_out.header.maps = cf.header.maps

            if hasattr(mrc_out.header, "cellb"):
                mrc_out.header.cellb.alpha = cf.header.cellb.alpha
                mrc_out.header.cellb.beta = cf.header.cellb.beta
                mrc_out.header.cellb.gamma = cf.header.cellb.gamma

            if hasattr(mrc_out.header, "origin"):
                mrc_out.header.origin.x = cf.header.origin.x
                mrc_out.header.origin.y = cf.header.origin.y
                mrc_out.header.origin.z = cf.header.origin.z

            mrc_out.header.nx = data.shape[2]
            mrc_out.header.ny = data.shape[1]
            mrc_out.header.nz = data.shape[0]
            mrc_out.header.mx = data.shape[2]
            mrc_out.header.my = data.shape[1]
            mrc_out.header.mz = data.shape[0]

            mrc_out.update_header_stats()

    print(f"Header-fixed tomogram written to: {output_tomo}")


def main():
    args = parse_args()
    cryoflows_tomo = args.in_cryoflows
    aretomo_tomo = args.in_aretomo

    if args.out:
        output_tomo = args.out
    else:
        base, ext = os.path.splitext(aretomo_tomo)
        output_tomo = f"{base}_headerfix{ext}"

    headerfix(cryoflows_tomo, aretomo_tomo, output_tomo)


if __name__ == "__main__":
    main()
