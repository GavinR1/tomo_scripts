#!/usr/bin/env python3
"""Project particles from tomograms into a stack for CryoSPARC."""

from argparse import ArgumentParser, SUPPRESS
from pathlib import Path
from tempfile import TemporaryDirectory

import mrcfile
import numpy as np
import pandas as pd
import starfile
from scipy.ndimage import gaussian_filter, rotate
from tqdm import tqdm


COORD_COLUMNS = ["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"]


def read_picks(path, tomo_column, coords_scale, voxel_spacing):
    blocks = starfile.read(path, always_dict=True, parse_as_string=[tomo_column])
    required = [tomo_column, *COORD_COLUMNS]
    candidates = [blocks["particles"]] if "particles" in blocks else blocks.values()
    tables = [
        table for table in candidates
        if isinstance(table, pd.DataFrame) and set(required).issubset(table.columns)
    ]
    if len(tables) != 1 or tables[0].empty:
        raise ValueError(f"{path} must contain one nonempty pick table with {', '.join(required)}")

    picks = tables[0][required].copy()
    names = picks[tomo_column]
    if names.isna().any() or names.str.strip().isin(["", ".", "?"]).any():
        raise ValueError(f"{path} contains missing tomogram names")
    coords = picks[COORD_COLUMNS].apply(pd.to_numeric, errors="coerce")
    coords = coords * coords_scale / voxel_spacing
    if not np.isfinite(coords.to_numpy(dtype=float)).all():
        raise ValueError(f"{path} contains invalid XYZ coordinates")
    picks[COORD_COLUMNS] = coords
    return picks


def crop_volume(volume, center_zyx, shape_zyx):
    start = center_zyx - shape_zyx // 2
    stop = start + shape_zyx
    lo = np.maximum(start, 0)
    hi = np.minimum(stop, volume.shape)
    if np.any(lo >= hi):
        raise ValueError(f"Crop at XYZ pixel coordinates {tuple(center_zyx[::-1].tolist())} misses the tomogram")

    source = tuple(slice(lower, upper) for lower, upper in zip(lo, hi))
    crop = volume[source].astype(np.float32, copy=False)
    if crop.shape == tuple(shape_zyx):
        return crop

    # Preserve the crop and use smoothed noise only for the part beyond the edge.
    target = tuple(slice(lower, upper) for lower, upper in zip(lo - start, hi - start))
    padded = np.random.normal(abs(crop.mean()), 10 * crop.std(), size=shape_zyx)
    padded[target] = crop
    padded = gaussian_filter(padded, sigma=2)
    padded[target] = crop
    return padded.astype(np.float32)


def project(crop, angle, box_size):
    if angle:
        crop = rotate(crop, angle, axes=(0, 2), reshape=True, order=1,
                      mode="constant", cval=0)
    width = crop.shape[2]
    if width < box_size:
        padding = box_size - width
        left = padding // 2
        return np.pad(crop.sum(axis=0), ((0, 0), (left, padding - left)))
    x_start = (width - box_size) // 2
    return crop[:, :, x_start:x_start + box_size].sum(axis=0)


def write_stack_star(path, stack_name, num_images, box_size, voxel_spacing):
    optics = pd.DataFrame({
        "rlnOpticsGroup": [1],
        "rlnOpticsGroupName": ["optics1"],
        "rlnVoltage": [300.0],
        "rlnSphericalAberration": [2.7],
        "rlnAmplitudeContrast": [0.1],
        "rlnImageDimensionality": [2],
        "rlnImagePixelSize": [voxel_spacing],
        "rlnImageSize": [box_size],
    })
    particles = pd.DataFrame({
        "rlnImageName": [f"{i}@{stack_name}" for i in range(1, num_images + 1)],
        "rlnOpticsGroup": 1,
        "rlnGroupNumber": 1,
    })
    starfile.write({"optics": optics, "particles": particles}, path)


def extract_and_project(args, shape_xyz):
    picks = read_picks(args.in_coords, args.col_name, args.coords_scale, args.voxel_spacing)
    groups = picks.groupby(args.col_name, sort=True)
    tomograms = {name: args.in_vol / f"{name}.mrc" for name in groups.groups}
    for path in tomograms.values():
        if not path.is_file():
            raise FileNotFoundError(f"Tomogram not found: {path}")

    box_size = int(shape_xyz[0])
    crop_shape_zyx = shape_xyz[::-1].copy()
    if any(args.angles):
        # Match the existing tilt projections by extending the crop along X.
        crop_shape_zyx[2] = (box_size * 3 // 2) // 2 * 2

    num_angles = len(args.angles)
    num_images = len(picks) * num_angles
    stack_name = "particles.mrcs"
    star_name = "particles_stack.star"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out_dir) as tmp:
        staging = Path(tmp)
        # Mode 2 stores float32. Write projections directly to their stack indices.
        with mrcfile.new_mmap(staging / stack_name, (num_images, box_size, box_size),
                              mrc_mode=2) as stack:
            stack.set_image_stack()
            stack.voxel_size = args.voxel_spacing
            tomogram_offset = 0
            for name, rows in tqdm(groups, desc="Projecting tomograms"):
                with mrcfile.mmap(tomograms[name], permissive=True) as tomo:
                    if tomo.data is None or tomo.data.ndim != 3:
                        raise ValueError(f"{tomograms[name]} must contain a 3D tomogram")
                    centers_zyx = rows[COORD_COLUMNS[::-1]].to_numpy().astype(int)
                    # Visit nearby Z slices together while preserving the original stack order.
                    for pick_index in np.argsort(centers_zyx[:, 0], kind="stable"):
                        crop = crop_volume(tomo.data, centers_zyx[pick_index], crop_shape_zyx)
                        first_image = tomogram_offset + pick_index * num_angles
                        for image_index, angle in enumerate(args.angles, start=first_image):
                            stack.data[image_index] = -project(crop, angle, box_size)
                tomogram_offset += len(rows) * num_angles

        write_stack_star(staging / star_name, stack_name, num_images, box_size, args.voxel_spacing)
        for name in (stack_name, star_name):
            (staging / name).replace(args.out_dir / name)

    print(f"Wrote {num_images} projections to {args.out_dir / stack_name} and {args.out_dir / star_name}")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--in_coords", type=Path, required=True, help="STAR file containing particle picks")
    parser.add_argument("--in_vol", type=Path, required=True, help="Directory containing <tomogram name>.mrc files")
    parser.add_argument("--out_dir", type=Path, required=True, help="Directory for particles.mrcs and particles_stack.star")
    parser.add_argument("--extract_shape", type=int, nargs=3, required=True, metavar=("X", "Y", "Z"),
                        help="Crop dimensions in angstroms, rounded down to even voxel dimensions")
    parser.add_argument("--voxel_spacing", type=float, required=True, help="Tomogram voxel size in angstroms")
    parser.add_argument("--coords_scale", type=float, default=1,
                        help="Multiply input coordinates by this factor to convert to angstroms")
    parser.add_argument("--col_name", default="rlnTomoName", help="STAR column containing tomogram names")
    parser.add_argument("--angles", type=float, nargs="+", default=[0.0], help=SUPPRESS)
    args = parser.parse_args()

    if not np.isfinite(args.voxel_spacing) or args.voxel_spacing <= 0:
        parser.error("--voxel_spacing must be positive and finite")
    if not np.isfinite(args.coords_scale) or args.coords_scale <= 0:
        parser.error("--coords_scale must be positive and finite")
    if not np.isfinite(args.angles).all():
        parser.error("--angles must be finite")

    shape_xyz = (np.asarray(args.extract_shape) / args.voxel_spacing / 2).astype(int) * 2
    if np.any(shape_xyz < 2):
        parser.error("--extract_shape must span at least two voxels along each axis")
    if shape_xyz[0] != shape_xyz[1]:
        parser.error("--extract_shape must give square XY projections for CryoSPARC")
    extract_and_project(args, shape_xyz)


if __name__ == "__main__":
    main()
