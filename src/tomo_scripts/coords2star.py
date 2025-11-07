"""
Combine all *.coords files in a directory into a single STAR file.

Each input file is assumed to contain:
    x y z

The filename (without ".coords") is used as rlnTomoName.

Usage:
  python coords2star.py -i ./coords -o all_picks.star
"""

import argparse
from pathlib import Path
import pandas as pd
import starfile


def load_coords_file(path: Path) -> pd.DataFrame:
    """Load a single .coords file into a standardized dataframe."""
    tomo_name = path.name.replace(".coords", "")
    df = pd.read_csv(path, header=None, sep=" ")
    df.columns = ["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"]
    df["rlnTomoName"] = tomo_name
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Combine *.coords files into a single STAR file."
    )
    parser.add_argument(
        "-i", "--input-dir", type=Path, required=True, help="Directory with coords files"
    )
    parser.add_argument(
        "-o", "--output-file", type=Path, required=True, help="Output .star file path"
    )

    args = parser.parse_args()
    input_dir: Path = args.input_dir
    output_file: Path = args.output_file

    coords_files = sorted(input_dir.glob("*.coords"))
    if not coords_files:
        raise FileNotFoundError(f"No .coords files found in {input_dir}")

    all_dfs = []
    for path in coords_files:
        print(f"[INFO] Reading {path.name}")
        df = load_coords_file(path)
        all_dfs.append(df)

    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"[OK] Combined {len(coords_files)} files, total {len(combined_df)} entries.")

    starfile.write(combined_df, output_file, overwrite=True)
    print(f"[DONE] Wrote {output_file}")


if __name__ == "__main__":
    main()
