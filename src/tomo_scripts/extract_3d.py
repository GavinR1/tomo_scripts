## Author: Gavin Rice
## Date Created: 05.03.25

import argparse
from typing import Iterable, List, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import mrcfile
import starfile
from tqdm import tqdm

COORD_COLUMNS = ["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"]
TOMO_COLUMN = "rlnTomoName"
PARTICLE_COLUMNS = [TOMO_COLUMN, *COORD_COLUMNS, "rlnImageName", "rlnOpticsGroup", "rlnGroupNumber"]

def create_parser() -> argparse.ArgumentParser:
    """Creates parser"""
    parser = argparse.ArgumentParser(description="Extract 3D subvolumes from STAR coordinates.")
    parser.add_argument("--star", type=Path, required=True,
                        help="STAR file containing tomogram names and XYZ pixel coordinates")
    parser.add_argument("--vol_dir", type=Path, required=True, help="Directory containing tomograms")
    parser.add_argument("--tomo_column", "--tomo-column", default=TOMO_COLUMN,
                        help="STAR column containing tomogram names (default: %(default)s)")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument("--boxsize", type=int, required=True, help="Boxsize in px")
    parser.add_argument("--coords_scale", type=float,
                        help="Input coordinate pixel size in angstroms; converts coordinates to tomogram pixels (default: coordinates already use tomogram pixels)")
    parser.add_argument("--voxel_spacing", type=float,
                        help="Tomogram voxel size in angstroms (default: MRC header); also used for extracted MRC and STAR pixel sizes")
    parser.add_argument("--apix", type=float, help="Tilt-series pixel size in angstroms for STAR output (defaults to the MRC voxel size)")
    parser.add_argument("--voltage", type=float, default=300.0, help="Microscope voltage in kV")
    parser.add_argument("--cs", type=float, default=2.5, help="Spherical aberration in mm")
    parser.add_argument("--amplitude-contrast", type=float, default=0.1, help="Amplitude contrast")
    parser.add_argument("--ctf-precomputed", action="store_true", help="Mark particles as CTF-premultiplied in the STAR output")
    return parser

def extract_and_write(coords: Iterable[Tuple], volume: np.ndarray, output_dir: Path,
                      *, apix, tomoname: str, box_size: int) -> List[Tuple[str, Tuple]]:
    '''Writes subvolumes to disk.'''
    written_files = []
    for line in coords:
        x, y, z = (int(value) - box_size // 2 for value in line)
        nz, ny, nx = volume.shape
        if min(x, y, z) < 0 or x + box_size > nx or y + box_size > ny or z + box_size > nz:
            continue
        subvol = volume[z:z + box_size, y:y + box_size, x:x + box_size]
        filename = f"{tomoname}_particle_{len(written_files):03d}.mrc"
        with mrcfile.new(output_dir.joinpath(filename), overwrite=True) as newmrc:
            newmrc.set_data(-subvol.astype(np.float32))
            newmrc.voxel_size = apix
        written_files.append((filename, line))
    return written_files


def read_star_particles(path: Path, tomo_column: str = TOMO_COLUMN) -> pd.DataFrame:
    """Read coordinate rows from a particles block or a single coordinate loop."""
    blocks = starfile.read(path, always_dict=True, parse_as_string=[tomo_column])
    required = {tomo_column, *COORD_COLUMNS}
    if "particles" in blocks:
        candidates = [blocks["particles"]]
    else:
        candidates = list(blocks.values())
    tables = [table for table in candidates if isinstance(table, pd.DataFrame)
              and required.issubset(table.columns)]
    if len(tables) != 1:
        raise ValueError(f"{path} must contain one particle table with {', '.join([tomo_column, *COORD_COLUMNS])}")
    particles = tables[0].copy()
    if particles[tomo_column].isna().any() or particles[tomo_column].str.strip().isin(["", ".", "?"]).any():
        raise ValueError(f"{path} contains a missing {tomo_column}")
    particles[COORD_COLUMNS] = particles[COORD_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(particles[COORD_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError(f"{path} contains non-numeric or non-finite particle coordinates")
    return particles


def find_tomogram(vol_dir: Path, name: str) -> Path:
    path = vol_dir / f"{name}.mrc"
    if path.is_file():
        return path
    raise FileNotFoundError(f"No tomogram found at {path}")


def run(args: argparse.Namespace) -> None:
    particles = read_star_particles(args.star, args.tomo_column)
    groups = particles.groupby(args.tomo_column, sort=False)
    tomograms = {}
    prefixes = set()
    for name in groups.groups:
        path = find_tomogram(args.vol_dir, name)
        prefix = Path(name).name
        if prefix.lower().endswith(".mrc"):
            prefix = prefix[:-4]
        if prefix in prefixes:
            raise ValueError(f"Multiple tomograms would write particles with the same filename prefix {prefix!r}")
        prefixes.add(prefix)
        tomograms[name] = (path, prefix)

    output_dir = args.out
    particle_dir = output_dir / "particles"
    particle_dir.mkdir(parents=True, exist_ok=True)
    optics_by_apix = {}
    particle_rows = []
    skipped = 0
    for number, (name, rows) in enumerate(groups, start=1):
        path, prefix = tomograms[name]
        label = f"{number}/{groups.ngroups} {name}"
        tqdm.write(f"{label}: loading tomogram into RAM")
        with mrcfile.open(path, mode="r") as mrc:
            if mrc.data.ndim != 3:
                raise ValueError(f"{path} must contain a 3D tomogram")
            spacing = np.array([float(mrc.voxel_size[axis]) for axis in ("x", "y", "z")])
            if args.voxel_spacing is not None:
                spacing[:] = args.voxel_spacing
            if not np.isfinite(spacing).all() or (spacing <= 0).any():
                raise ValueError(f"{path} has invalid voxel sizes; provide --voxel_spacing")
            apix = float(spacing[0])
            coords = rows[COORD_COLUMNS].to_numpy(dtype=float)
            if args.coords_scale is not None:
                coords = coords * args.coords_scale / spacing
            with tqdm(coords, desc=f"{label}: extracting", unit="pick") as progress:
                written = extract_and_write(progress, mrc.data, particle_dir, apix=tuple(spacing),
                                            tomoname=prefix, box_size=args.boxsize)
        skipped += len(rows) - len(written)
        if written:
            group = optics_by_apix.setdefault(apix, len(optics_by_apix) + 1)
            particle_rows.extend((name, *coord, f"particles/{filename}", group, 1)
                                 for filename, coord in written)
        print(f"{name}: wrote {len(written)} of {len(rows)} particles.")

    optics = pd.DataFrame({
        "rlnOpticsGroup": list(optics_by_apix.values()),
        "rlnOpticsGroupName": [f"optics{group}" for group in optics_by_apix.values()],
        "rlnVoltage": args.voltage,
        "rlnSphericalAberration": args.cs,
        "rlnAmplitudeContrast": args.amplitude_contrast,
        "rlnTomoTiltSeriesPixelSize": list(optics_by_apix) if args.apix is None else args.apix,
        "rlnCtfDataAreCtfPremultiplied": int(args.ctf_precomputed),
        "rlnImageDimensionality": 3,
        "rlnImagePixelSize": list(optics_by_apix),
        "rlnImageSize": args.boxsize,
    })
    extracted = pd.DataFrame.from_records(particle_rows, columns=PARTICLE_COLUMNS)
    out_star = output_dir / "extracted_subvolumes.star"
    starfile.write({"optics": optics, "particles": extracted}, out_star, overwrite=True)
    print(f"Wrote {len(extracted)} particles to {out_star}. Skipped {skipped} crops outside the tomogram bounds.")


def _main_() -> None:
    parser = create_parser()
    args = parser.parse_args()
    if args.boxsize < 1:
        parser.error("--boxsize must be positive")
    if args.apix is not None and (not np.isfinite(args.apix) or args.apix <= 0):
        parser.error("--apix must be a positive pixel size")
    for option in ("coords_scale", "voxel_spacing"):
        value = getattr(args, option)
        if value is not None and (not np.isfinite(value) or value <= 0):
            parser.error(f"--{option} must be positive and finite")
    run(args)


if __name__ == "__main__":
    _main_()
