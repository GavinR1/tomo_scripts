## Author: Gavin Rice
## Date Created: 05.03.25

import argparse
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
from pathlib import Path
from enum import Enum, auto
import mrcfile
from dataclasses import dataclass
import os
import starfile

'''
Extracts subvolumes and write them to disk with optional 2D projection.
If --in_star is provided, reads already-extracted subvolumes from rlnImageName
and projects them directly to 2D using --n central Z-slices.
'''

class OutOfVolumeException(Exception):
    ''' out of volume exception'''
    pass

def create_parser() -> argparse.ArgumentParser:
    """Creates parser"""
    parser = argparse.ArgumentParser()

    # Two input modes:
    # (A): tomogram list + directories (extraction path)
    # (B): in_star with rlnImageName paths (projection-only path)
    parser.add_argument("--in_star", type=str, required=False,
                        help="STAR file containing rlnImageName of already-extracted 3D subvolumes. "
                             "If provided, the script will project these to 2D (ignores extraction inputs).")

    parser.add_argument("--tomograms", type=str, required=False,
                        help="Path to text file containing list of tomograms (extraction mode)")
    parser.add_argument("--vol_dir", type=str, required=False,
                        help="Directory containing tomograms (extraction mode)")
    parser.add_argument("--coord_dir", type=str, required=False,
                        help="Directory containing coordinates (extraction mode)")

    parser.add_argument("--out", type=str, required=True, help="Output directory")
    parser.add_argument("--boxsize", type=int, required=False, help="Boxsize in px (extraction mode)")
    parser.add_argument("--id", type=str, required=False, help="Name of protein to extract (extraction mode)")

    parser.add_argument("--project2D", action="store_true",
                        help="Project subvolumes to 2D images (always done if --in_star is used)")
    parser.add_argument("--n", type=int, default=1,
                        help="Number of Z-slices (from the center) to use for projection")
    return parser

def project_subvolumes_to_2d(subvolume: np.ndarray, n: int) -> np.ndarray:
    """Projects a 3D subvolume to a 2D image along the Z-axis using n central slices."""
    if subvolume.ndim != 3:
        raise ValueError(f"Expected 3D subvolume, got shape {subvolume.shape}")
    z_size = subvolume.shape[0]
    if n <= 0:
        raise ValueError("--n must be >= 1")
    n_eff = min(n, z_size)
    middle_z = z_size // 2
    start_z = max(0, middle_z - (n_eff // 2))
    end_z = min(z_size, start_z + n_eff)
    start_z = max(0, end_z - n_eff)
    projected_image = np.sum(subvolume[start_z:end_z, :, :], axis=0)
    return projected_image.astype(np.float32)

def write_projected_images(output_dir: Path,
                           subvolumes: List[np.ndarray],
                           n: int,
                           coords: List[Tuple[float, float, float]],
                           written_files: List[str],
                           tomonames: List[str]) -> None:
    """Writes the projected 2D images to disk and creates a star file."""
    path_2d_projections = output_dir / '2D_projections'
    path_2d_projections.mkdir(exist_ok=True, parents=True)
    data = {
        'rlnCoordinateX': [],
        'rlnCoordinateY': [],
        'rlnCoordinateZ': [],
        'rlnImageName': [],
        'rlnMicrographName': []
    }
    for i, (subvolume, (x, y, z), file) in enumerate(zip(subvolumes, coords, written_files)):
        projected_image = project_subvolumes_to_2d(subvolume, n)
        filename = f"2D_{Path(file).name}"
        with mrcfile.new(path_2d_projections / filename, overwrite=True) as mrc:
            mrc.set_data(projected_image)
        data['rlnCoordinateX'].append(x)
        data['rlnCoordinateY'].append(y)
        data['rlnCoordinateZ'].append(z)
        data['rlnImageName'].append(f"2D_projections/{filename}")
        tomo = tomonames[i] if i < len(tomonames) else ""
        data['rlnMicrographName'].append(tomo)
    df = pd.DataFrame(data)
    starfile.write(df, output_dir / 'extracted_subvolumes_2D.star')

def extract_and_write(coords, volume: np.array, output_dir: Path, apix=None, tomoname=None, pid=None, box_size=None) -> List[Tuple[str, Tuple]]:
    '''Writes subvolumes to disk.'''
    pnum = 1
    written_files = []
    for line in coords:
        nx1 = (int(line[0]) - (box_size) // 2)
        nx2 = (int(line[0]) + (box_size) // 2)
        ny1 = (int(line[1]) - (box_size) // 2)
        ny2 = (int(line[1]) + (box_size) // 2)
        nz1 = (int(line[2]) - (box_size) // 2)
        nz2 = (int(line[2]) + (box_size) // 2)

        subvol = volume[nz1:nz2, ny1:ny2, nx1:nx2]
        if subvol.shape != (box_size, box_size, box_size):
            continue
        subvol = subvol * -1
        num_str = str(pnum).zfill(4)
        filename = tomoname + '_' + pid + '_' + num_str + '.mrc'
        with mrcfile.new(output_dir.joinpath(filename), overwrite=True) as newmrc:
            newmrc.set_data(subvol.astype(np.float32))
            if apix is not None:
                newmrc.voxel_size = apix
        written_files.append((filename, line))
        pnum += 1
    return written_files

def read_apix(path_vol: Path):
    if path_vol.exists() and path_vol.is_file():
        with mrcfile.mmap(path_vol) as mrc:
            return mrc.voxel_size
    else:
        raise ValueError("Your input volume does not exist.")

def read_volume(path_vol: Path) -> np.array:
    if path_vol.exists() and path_vol.is_file():
        with mrcfile.open(path_vol) as mrc:
            return mrc.data
    else:
        raise ValueError("Your input volume does not exist.")

def get_coordinates(path_coord: Path, pixel_size: float) -> List[Tuple]:
    with open(path_coord, 'r') as file1:
        lines = file1.readlines()
    coords = []
    for l in lines:
        lsplitted = l.split()
        x = float(lsplitted[0])
        y = float(lsplitted[1])
        z = float(lsplitted[2])
        coords.append((x, y, z))
    return coords

def _load_star_df(star_path: Path) -> pd.DataFrame:
    """Load a STAR file as a DataFrame. Supports single or multi-data blocks."""
    table = starfile.read(star_path)
    if isinstance(table, dict):
        for key in ("data_particles", "particles", "data_"):
            if key in table:
                return table[key]
        return next(iter(table.values()))
    return table

def load_subvolumes_from_star(star_path: Path) -> Tuple[List[np.ndarray],
                                                        List[Tuple[float, float, float]],
                                                        List[str],
                                                        List[str]]:
    """
    Reads subvolume file paths from rlnImageName in the STAR, opens each volume,
    and returns (subvolumes, coords, image_names, tomo_names).
    """
    df = _load_star_df(star_path)

    if 'rlnImageName' not in df.columns:
        raise ValueError("STAR file must contain 'rlnImageName'.")

    base_dir = star_path.parent
    image_names: List[str] = df['rlnImageName'].astype(str).tolist()

    xs = df['rlnCoordinateX'].tolist() if 'rlnCoordinateX' in df.columns else [0.0]*len(df)
    ys = df['rlnCoordinateY'].tolist() if 'rlnCoordinateY' in df.columns else [0.0]*len(df)
    zs = df['rlnCoordinateZ'].tolist() if 'rlnCoordinateZ' in df.columns else [0.0]*len(df)
    coords = list(zip(xs, ys, zs))

    if 'rlnTomoName' in df.columns:
        tomo_names = df['rlnTomoName'].astype(str).tolist()
    elif 'rlnMicrographName' in df.columns:
        tomo_names = df['rlnMicrographName'].astype(str).tolist()
    else:
        tomo_names = [""] * len(df)

    subvolumes: List[np.ndarray] = []
    for relpath in image_names:
        vol_path = (base_dir / relpath).resolve()
        if not vol_path.exists():
            vol_path = Path(relpath)
        if not vol_path.exists():
            raise FileNotFoundError(f"Subvolume file not found: {relpath}")
        with mrcfile.open(vol_path) as mrc:
            arr = mrc.data
            if arr.ndim == 3:
                subvol = arr
            elif arr.ndim == 4 and arr.shape[0] == 1:
                subvol = arr[0]
            else:
                raise ValueError(f"File {vol_path} is not a 3D subvolume (got shape {arr.shape}).")
            subvolumes.append(subvol)
    return subvolumes, coords, image_names, tomo_names

def run_extraction_mode(args) -> None:
    """Extract subvolumes from tomograms, optionally project, write star file."""
    path_tomograms = Path(args.tomograms)
    path_out = Path(args.out)
    path_vol_dir = Path(args.vol_dir)
    path_coord_dir = Path(args.coord_dir)
    box_size = args.boxsize
    pid = args.id

    if not all([args.tomograms, args.vol_dir, args.coord_dir, args.boxsize, args.id]):
        raise ValueError("Extraction mode requires --tomograms, --vol_dir, --coord_dir, --boxsize, and --id.")

    with open(path_tomograms, 'r') as file1:
        lines = file1.readlines()
    filenames = [line.strip() for line in lines]

    all_written_files = []
    all_coords = []
    all_tomonames = []

    path_out.mkdir(exist_ok=True, parents=True)
    path_3d_subvolumes = path_out / '3D_subvolumes'
    path_3d_subvolumes.mkdir(exist_ok=True, parents=True)

    all_subvolumes = []
    all_coords_2d = []
    all_tomonames_2d = []

    for filename in filenames:
        path_vol = path_vol_dir / f"{filename}.mrc"
        path_coord = path_coord_dir / f"{filename}.coords"
        volume = read_volume(path_vol)
        apix = read_apix(path_vol)
        coords = get_coordinates(path_coord, apix)

        print(f"Extracting and writing 3D subvolumes to disk for {filename}.")
        written_files = extract_and_write(coords, volume, output_dir=path_3d_subvolumes, apix=apix, tomoname=filename,
                                          pid=pid, box_size=box_size)
        all_written_files.extend([f"3D_subvolumes/{file[0]}" for file in written_files])
        all_coords.extend([file[1] for file in written_files])
        all_tomonames.extend([filename] * len(written_files))

        if args.project2D:
            print(f"Projecting subvolumes to 2D for {filename}.")
            subvolumes = []
            for file in written_files:
                with mrcfile.open(path_3d_subvolumes / file[0]) as mrc:
                    subvolumes.append(mrc.data)
            all_subvolumes.extend(subvolumes)
            all_coords_2d.extend([file[1] for file in written_files])
            all_tomonames_2d.extend([filename] * len(written_files))

    print("Writing star file for 3D subvolumes.")
    data = {
        'rlnCoordinateX': [x for x, y, z in all_coords],
        'rlnCoordinateY': [y for x, y, z in all_coords],
        'rlnCoordinateZ': [z for x, y, z in all_coords],
        'rlnImageName': all_written_files,
        'rlnTomoName': all_tomonames
    }
    df = pd.DataFrame(data)
    starfile.write(df, path_out / 'extracted_subvolumes.star')

    if args.project2D and len(all_subvolumes) > 0:
        print("Writing star file for 2D Projections.")
        write_projected_images(path_out, all_subvolumes, args.n, all_coords_2d, all_written_files, all_tomonames_2d)

def run_star_mode(args) -> None:
    """
    Take a STAR with rlnImageName pointing to already-extracted 3D subvolumes,
    project each to 2D using --n, and write a 2D STAR.
    """
    star_path = Path(args.in_star)
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True, parents=True)

    print(f"Loading subvolumes from STAR: {star_path}")
    subvolumes, coords, image_names, tomo_names = load_subvolumes_from_star(star_path)

    print(f"Projecting {len(subvolumes)} subvolumes to 2D using n={args.n} central Z-slices.")
    write_projected_images(out_dir, subvolumes, args.n, coords, image_names, tomo_names)
    print("Done writing 2D projections and STAR.")

def run(args) -> None:
    """Dispatch based on input mode."""
    if args.in_star:
        run_star_mode(args)
    else:
        run_extraction_mode(args)

def _main_() -> None:
    parser = create_parser()
    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    _main_()
