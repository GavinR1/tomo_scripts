#!/usr/bin/env python3
"""
Standalone minislab generator.

Supports:
  - Relion .star coordinate input + tomogram directory (mrc or zarr)
  - copick .json coordinate input + copick tomograms (via copick)

Does NOT include live mode.

Dependencies:
  numpy, scipy, pandas, tqdm, mrcfile, starfile, zarr
  plus copick ONLY IF you use copick json input.
"""

from __future__ import annotations

import json
import os
import warnings
from argparse import ArgumentParser
from typing import Optional

import numpy as np
import pandas as pd
import scipy.ndimage
import zarr
from tqdm import tqdm

warnings.simplefilter(action="ignore", category=FutureWarning)


def load_mrc(filename: str) -> np.ndarray:
    import mrcfile
    with mrcfile.open(filename, "r", permissive=True) as mrc:
        return mrc.data


def save_mrc(
    data: np.ndarray,
    filename: str,
    overwrite: bool = True,
    apix: float | None = None,
):
    import mrcfile
    if data.dtype != np.dtype("float32"):
        data = data.astype(np.float32)
    with mrcfile.new(filename, overwrite=overwrite) as mrc:
        mrc.set_data(data)
        if apix:
            mrc.voxel_size = apix


def get_voxel_size(filename: str, isotropic: bool = True) -> float:
    import mrcfile
    apix = mrcfile.open(filename).voxel_size.tolist()
    if isotropic:
        return apix[0]
    return apix


def make_starfile(d_coords: dict, out_file: str, coords_scale: float = 1):
    import starfile

    rln = {}
    rln["rlnTomoName"] = np.concatenate(
        [d_coords[tomo].shape[0] * [tomo] for tomo in d_coords],
    ).ravel()
    rln["rlnCoordinateX"] = (
        np.concatenate([d_coords[tomo][:, 0] for tomo in d_coords]).ravel()
        * coords_scale
    )
    rln["rlnCoordinateY"] = (
        np.concatenate([d_coords[tomo][:, 1] for tomo in d_coords]).ravel()
        * coords_scale
    )
    rln["rlnCoordinateZ"] = (
        np.concatenate([d_coords[tomo][:, 2] for tomo in d_coords]).ravel()
        * coords_scale
    )
    if np.all(np.array([d_coords[tomo].shape[1] for tomo in d_coords]) == 4):
        rln["rlnScore"] = np.concatenate([d_coords[tomo][:, 3] for tomo in d_coords]).ravel()
    for key in ["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]:
        rln[key] = np.zeros(len(rln["rlnCoordinateX"]))
    rln["rlnTomoManifoldIndex"] = np.ones(len(rln["rlnCoordinateX"])).astype(int)
    rln["rlnTomoParticleId"] = np.arange(len(rln["rlnCoordinateX"])).astype(int)

    rln_df = pd.DataFrame.from_dict(rln)
    starfile.write(rln_df, out_file)


def make_stack_starfile(
    in_stack: str,
    out_star: str,
    apix: float,
    ctf_precomputed: bool = True,
    voltage: float = 300.0,
    cs: float = 2.7,
    amplitude_contrast: float = 0.1,
):
    import starfile

    fname = os.path.basename(in_stack)
    apix_tomo = get_voxel_size(in_stack)
    n_particles, stack_dim, stack_dim1 = load_mrc(in_stack).shape
    assert stack_dim == stack_dim1

    grp_optics = {}
    grp_optics["rlnVoltage"] = [voltage]
    grp_optics["rlnSphericalAberration"] = [cs]
    grp_optics["rlnAmplitudeContrast"] = [amplitude_contrast]
    grp_optics["rlnTomoTiltSeriesPixelSize"] = [apix]
    grp_optics["rlnOpticsGroup"] = [1]
    grp_optics["rlnOpticsGroupName"] = ["optics1"]
    grp_optics["rlnCtfDataAreCtfPremultiplied"] = [1 if ctf_precomputed else 0]
    grp_optics["rlnImageDimensionality"] = [2]
    grp_optics["rlnImagePixelSize"] = [apix_tomo]
    grp_optics["rlnImageSize"] = [stack_dim]

    grp_particles = {}
    grp_particles["rlnImageName"] = [f"{num}@{fname}" for num in range(n_particles)]
    grp_particles["rlnOpticsGroup"] = n_particles * [1]
    grp_particles["rlnGroupNumber"] = n_particles * [1]

    dstack = {}
    dstack["optics"] = pd.DataFrame.from_dict(grp_optics)
    dstack["particles"] = pd.DataFrame.from_dict(grp_particles)

    starfile.write(dstack, out_star)


def read_starfile(
    in_star: str,
    col_name: str = "rlnTomoName",
    coords_scale: float = 1,
    extra_col_name: str | None = None,
) -> dict:
    import starfile

    particles = starfile.read(in_star)
    if len(particles) == 0:
        print(f"Warning: {in_star} appears to be an empty starfile")
        return {}
    if isinstance(particles, dict):
        particles = particles["particles"]

    tomo_names = np.unique(particles[col_name].values)
    d_coords = {}
    for tomo in tomo_names:
        tomo_indices = np.where(particles[col_name].values == tomo)[0]
        d_coords[tomo] = (
            np.array(
                [
                    particles.rlnCoordinateX.iloc[tomo_indices],
                    particles.rlnCoordinateY.iloc[tomo_indices],
                    particles.rlnCoordinateZ.iloc[tomo_indices],
                ],
            ).T
            * coords_scale
        )
        if extra_col_name is not None:
            d_coords[tomo] = np.hstack(
                (
                    d_coords[tomo],
                    particles[extra_col_name].iloc[tomo_indices].values[:, np.newaxis],
                ),
            )
    return d_coords


def combine_star_files(
    in_star: list,
    col_name: str = "rlnTomoName",
    coords_scale: float = 1,
) -> dict:
    d_coords_list = [
        read_starfile(star_path, col_name=col_name, coords_scale=coords_scale)
        for star_path in in_star
    ]
    d_coords = {}
    for d in d_coords_list:
        for k, v in d.items():
            d_coords.setdefault(k, []).append(v)

    d_coords = {key: d_coords[key] for key in d_coords}
    for key in d_coords:
        if len(d_coords[key]) > 1:
            print(f"Warning! {key} spanned multiple star files")
        d_coords[key] = np.vstack(d_coords[key])

    return d_coords


def read_copick_json(fname: str) -> np.ndarray:
    with open(fname) as f:
        points = json.load(f)["points"]
    locs = [points[i]["location"] for i in range(len(points))]
    return np.array([(locs[i]["x"], locs[i]["y"], locs[i]["z"]) for i in range(len(locs))])


class CopickInterface:
    def __init__(self, config: str):
        import copick
        self._copick = copick
        self.root = copick.from_file(config)

    def get_run_names(self) -> list[str]:
        return [run.name for run in self.root.runs]

    def get_run_tomogram(self, run_name: str, voxel_spacing: float, tomo_type: str) -> np.ndarray:
        run = self.root.get_run(run_name)
        tomogram = run.get_voxel_spacing(voxel_spacing).get_tomogram(tomo_type=tomo_type)
        return tomogram.numpy()

    def get_run_coords(
        self,
        run_name: str,
        particle_name: str,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> np.ndarray:
        picks = self.root.get_run(run_name).get_picks(
            particle_name,
            session_id=session_id,
            user_id=user_id,
        )

        if len(picks) == 0:
            return np.empty(0)

        if (user_id is not None) and len(picks) > 1:
            print(f"Warning! Multiple CopickPicks found for run {run_name} and user_id {user_id}")

        coords = []
        for pickset in picks:
            coords.extend([(p.location.x, p.location.y, p.location.z) for p in pickset.points])

        return np.array(coords)

    def get_all_coords(
        self,
        particle_name: str,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> dict:
        run_names = self.get_run_names()
        d_coords = {}
        for run_name in run_names:
            coords = self.get_run_coords(run_name, particle_name, user_id=user_id, session_id=session_id)
            if len(coords) > 0:
                d_coords[run_name] = coords
        return d_coords


def get_subvolume(
    coord: np.ndarray,
    volume: np.ndarray | zarr.core.Array,
    shape: np.ndarray | tuple[int, int, int],
) -> np.ndarray:
    c = coord.astype(int)
    xstart, xend = c[2] - int(shape[2] / 2), c[2] + int(shape[2] / 2)
    ystart, yend = c[1] - int(shape[1] / 2), c[1] + int(shape[1] / 2)
    zstart, zend = c[0] - int(shape[0] / 2), c[0] + int(shape[0] / 2)

    xstart = max(0, xstart)
    ystart = max(0, ystart)
    zstart = max(0, zstart)

    xend = min(xend, volume.shape[0])
    yend = min(yend, volume.shape[1])
    zend = min(zend, volume.shape[2])

    subvolume = volume[xstart:xend, ystart:yend, zstart:zend]

    delta = np.array(shape)[::-1] - np.array(subvolume.shape)
    if np.sum(np.abs(delta)) == 0:
        return subvolume

    fill_volume = np.random.normal(
        loc=np.abs(subvolume.mean()),
        scale=10 * subvolume.std(),
        size=np.array(shape)[::-1],
    )

    xs, ys, zs = 0, 0, 0
    ze, ye, xe = shape

    if delta[0] != 0:
        if xstart == 0:
            xs = delta[0]
        else:
            xe -= delta[0]

    if delta[1] != 0:
        if ystart == 0:
            ys = delta[1]
        else:
            ye -= delta[1]

    if delta[2] != 0:
        if zstart == 0:
            zs = delta[2]
        else:
            ze -= delta[2]

    fill_volume[xs:xe, ys:ye, zs:ze] = subvolume
    fill_volume = scipy.ndimage.gaussian_filter(fill_volume, sigma=2)
    fill_volume[xs:xe, ys:ye, zs:ze] = subvolume

    return fill_volume


def tilt_subvolume(
    subvolume: np.ndarray,
    angle: float,
    extract_shape: np.ndarray | tuple[int, int, int],
) -> np.ndarray:
    subvolume_rot = scipy.ndimage.rotate(
        subvolume,
        angle,
        axes=(0, 2),
        reshape=True,
        order=1,
        mode="constant",
        cval=0,
    )

    mdpt = int(subvolume_rot.shape[2] / 2)
    hdim = int(extract_shape[0] / 2)
    return subvolume_rot[:, :, mdpt - hdim : mdpt + hdim]


def render_even(shape: np.ndarray) -> np.ndarray:
    odd_dims = np.where(shape.astype(int) % 2)[0]
    if len(odd_dims) > 0:
        shape[odd_dims] -= 1
    return shape


def generate_minislabs(
    coords: np.ndarray,
    volume: np.ndarray,
    extract_shape: np.ndarray | tuple,
    angles: list | None = None,
    buffered_shape: np.ndarray | tuple | None = None,
) -> dict:
    if angles is None:
        angles = [0]
    counter = 0
    projs = {}

    if angles == [0]:
        buffered_shape = extract_shape
    else:
        if buffered_shape is None:
            buffered_shape = np.array([1.5, 1, 1]) * np.array(extract_shape)
            buffered_shape = render_even(buffered_shape.astype(int))

    for c in coords:
        subvolume = get_subvolume(c, volume, buffered_shape)
        if subvolume.shape != tuple(buffered_shape)[::-1]:
            raise ValueError("Shapes do not match")

        if angles == [0]:
            projs[counter] = np.sum(subvolume, axis=0)
            counter += 1
        else:
            for angle in angles:
                tilted_subvolume = tilt_subvolume(subvolume, angle, extract_shape)
                projs[counter] = np.sum(tilted_subvolume, axis=0)
                counter += 1

    return projs


class Minislab:
    def __init__(self, extract_shape: tuple[int, int, int], angles: list | None = None):
        if angles is None:
            angles = [0]
        self.minislabs = {}
        self.angles = angles
        self.shape = extract_shape
        buffered_shape = np.array([1.5, 1, 1]) * np.array(extract_shape)
        self.buffered_shape = render_even(buffered_shape.astype(int))

        self.num_particles = 0
        self.num_galleries = 0
        self.particle_index = []
        self.particle_tilt = []
        self.tomogram_id = []
        self.row_idx, self.col_idx, self.gallery_idx = [], [], []

    def make_minislabs(self, coords: np.ndarray, volume: np.ndarray, tomo_id: str) -> None:
        projs = generate_minislabs(coords, volume, self.shape, self.angles, self.buffered_shape)

        for i in range(len(projs)):
            self.minislabs[self.num_particles] = projs[i]
            self.num_particles += 1

        self.particle_tilt.extend(coords.shape[0] * list(self.angles))
        self.particle_index.extend(list(np.repeat(np.arange(coords.shape[0]), len(self.angles))))
        self.tomogram_id.extend(len(projs) * [tomo_id])

    def make_one_gallery(
        self,
        gshape: tuple[int, int],
        key_list: list,
        contrast: float = 1.0,
        border_factor: float = 1.1,
        border_width: int = 2,
    ) -> np.ndarray:
        if len(key_list) > np.prod(gshape):
            raise IndexError("Number of minislabs exceeds number of gallery tiles.")

        pshape = self.minislabs[key_list[0]].shape
        gallery = np.zeros((gshape[0] * pshape[0], gshape[1] * pshape[1]), dtype=np.float32)
        stacked = np.array([contrast * self.minislabs[i] for i in key_list])

        counter = 0
        for i in range(gshape[0]):
            for j in range(gshape[1]):
                if counter > len(key_list) - 1:
                    fill_volume = np.random.choice(stacked.flatten(), size=pshape)
                    gallery[i*pshape[0]:(i+1)*pshape[0], j*pshape[1]:(j+1)*pshape[1]] = fill_volume
                else:
                    gallery[i*pshape[0]:(i+1)*pshape[0], j*pshape[1]:(j+1)*pshape[1]] = stacked[counter]
                    self.row_idx.append(i)
                    self.col_idx.append(j)
                    self.gallery_idx.append(self.num_galleries)
                counter += 1

        self.num_galleries += 1

        if border_width != 0:
            border_value = border_factor * np.mean(stacked)
            for i in range(gshape[0]):
                gallery[i*pshape[0]-border_width:i*pshape[0]+border_width] = border_value
            for j in range(gshape[1]):
                gallery[:, j*pshape[1]-border_width:j*pshape[1]+border_width] = border_value

        return gallery

    def make_gallery_bookkeeper(self, out_dir: str) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "tomogram": self.tomogram_id,
                "particle": self.particle_index,
                "tilt": self.particle_tilt,
                "gallery": self.gallery_idx,
                "row": self.row_idx,
                "col": self.col_idx,
            },
        )
        df.to_csv(os.path.join(out_dir, "particle_map.csv"), index=False)
        return df

    def make_stack(self, out_dir: str, apix: float, contrast: float = 1.0) -> np.ndarray:
        os.makedirs(out_dir, exist_ok=True)
        stack = np.array([self.minislabs[i] for i in self.minislabs])
        stack *= contrast
        save_mrc(stack, os.path.join(out_dir, "particles.mrcs"), apix=apix)

        df = pd.DataFrame(
            {"tomogram": self.tomogram_id, "particle": self.particle_index, "tilt": self.particle_tilt},
        )
        df.to_csv(os.path.join(out_dir, "particle_map.csv"), index=False)
        return stack


def make_minislabs_multi_entry(
    in_coords: str,
    in_vol: Optional[str],
    out_dir: str,
    extract_shape: tuple[int, int, int],
    voxel_spacing: float,
    extension: str = "mrc",
    tomo_type: str | None = None,
    particle_name: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    coords_scale: float = 1,
    col_name: str = "rlnMicrographName",
    angles: list = [0],
    gshape: tuple[int, int] = (16, 15),
    make_stack: bool = False,
    invert_contrast: bool = False,
) -> None:
    # load coordinates -- starfile entry
    if os.path.splitext(in_coords)[-1] == ".star":
        cp_interface = None
        d_coords = read_starfile(in_coords, coords_scale=coords_scale, col_name=col_name)

    # load coordinates -- copick entry
    elif os.path.splitext(in_coords)[-1] == ".json":
        cp_interface = CopickInterface(in_coords)
        if particle_name is None:
            raise ValueError("--particle_name is required when --in_coords is a copick .json")
        d_coords = cp_interface.get_all_coords(particle_name, user_id=user_id, session_id=session_id)

    else:
        raise ValueError("in_coords argument not recognized (expected .star or .json)")

    # allow copick volumes via in_vol too
    if cp_interface is None and in_vol is not None and os.path.splitext(in_vol)[-1] == ".json":
        cp_interface = CopickInterface(in_vol)

    os.makedirs(out_dir, exist_ok=True)

    extract_shape_px = (np.array(extract_shape) / voxel_spacing).astype(int)
    extract_shape_px = render_even(extract_shape_px)

    contrast = -1.0 if invert_contrast else 1.0

    n_tiles = int(np.prod(np.array(gshape)))
    montage = Minislab(tuple(extract_shape_px.tolist()), angles)

    for run_name in tqdm(d_coords):
        print(f"Processing volume {run_name}")

        # load volume -- directory entry
        if in_vol is not None and os.path.isdir(in_vol):
            vol_name = os.path.join(in_vol, f"{run_name}.{extension}")
            if extension == "mrc":
                volume = load_mrc(vol_name)
            elif extension == "zarr":
                volume = np.array(zarr.open(vol_name, "r"))
            else:
                raise ValueError("extension must be 'mrc' or 'zarr'")
        # load volume -- copick entry
        else:
            if cp_interface is None:
                raise ValueError("Could not load volumes: provide --in_vol directory or copick config")
            if tomo_type is None:
                raise ValueError("--tomo_type is required when loading tomograms from copick")
            volume = cp_interface.get_run_tomogram(run_name, voxel_spacing, tomo_type)

        coords_pixels = d_coords[run_name] / voxel_spacing
        montage.make_minislabs(coords_pixels, volume, run_name)

        # write out galleries as sufficient minislabs accumulate
        if len(montage.minislabs) > n_tiles and not make_stack:
            goffset = montage.num_galleries
            n_galleries = len(montage.minislabs) // n_tiles
            for ng in range(n_galleries):
                gstart = ng + goffset
                key_list = np.arange(gstart * n_tiles, gstart * n_tiles + n_tiles).astype(int)
                gallery = montage.make_one_gallery(gshape, list(key_list), contrast=contrast)
                save_mrc(
                    gallery,
                    os.path.join(out_dir, f"particles_{montage.num_galleries-1:03d}.mrc"),
                    apix=voxel_spacing,
                )
                for k in key_list:
                    montage.minislabs.pop(int(k), None)

    if not make_stack:
        key_list = list(montage.minislabs.keys())
        if len(key_list) > 0:
            gallery = montage.make_one_gallery(gshape, key_list, contrast=contrast)
            save_mrc(
                gallery,
                os.path.join(out_dir, f"particles_{montage.num_galleries-1:03d}.mrc"),
                apix=voxel_spacing,
            )
        montage.make_gallery_bookkeeper(out_dir)

    if make_stack:
        montage.make_stack(out_dir, voxel_spacing, contrast=contrast)



def parse_args():
    parser = ArgumentParser(description="Generate minislabs (per-particle 2D projections).")

    parser.add_argument("--in_coords", type=str, required=True,
                        help="Coordinate file: a Relion .star OR a copick config .json")
    parser.add_argument("--in_vol", type=str, required=False,
                        help="Directory containing volumes OR a copick config .json (if volumes from copick)")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output directory for galleries and/or particle stack")

    parser.add_argument("--extract_shape", type=int, nargs=3, required=True,
                        help="Subvolume extraction shape (x y z) in Angstrom")
    parser.add_argument("--voxel_spacing", type=float, required=True,
                        help="Tomogram voxel size in Angstrom")

    parser.add_argument("--coords_scale", type=float, default=1,
                        help="Multiplicative factor to convert input coords to Angstrom (starfile case)")

    parser.add_argument("--col_name", type=str, default="rlnTomoName",
                        help="Tomogram column name in starfile(s)")

    parser.add_argument("--extension", type=str, default="mrc",
                        help="Tomogram extension if reading from directory: mrc or zarr")

    # copick options
    parser.add_argument("--tomo_type", type=str, required=False,
                        help="Tomogram type if extracting from copick")
    parser.add_argument("--user_id", type=str, required=False,
                        help="User ID if coordinates from copick")
    parser.add_argument("--session_id", type=str, required=False,
                        help="Session ID if coordinates from copick")
    parser.add_argument("--particle_name", type=str, required=False,
                        help="Particle name (required if coords from copick)")

    parser.add_argument("--angles", type=float, nargs="+", default=[0],
                        help="Tilt angles to apply to each particle")

    parser.add_argument("--gallery_shape", type=int, nargs=2, default=[16, 15],
                        help="Gallery shape (rows cols)")

    parser.add_argument("--make_stack", action="store_true",
                        help="Make particle stack instead of galleries")

    parser.add_argument("--invert_contrast", action="store_true",
                        help="Invert default contrast")

    return parser.parse_args()


def write_config_json(args) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    d = vars(args)

    cfg = {
        "software": {"name": "slabpick-standalone", "version": "1.0.0"},
        "input": {"in_coords": d.get("in_coords"), "in_vol": d.get("in_vol")},
        "output": {"out_dir": d.get("out_dir")},
        "parameters": {k: v for k, v in d.items() if k not in {"in_coords", "in_vol", "out_dir"}},
    }

    with open(os.path.join(args.out_dir, "make_minislabs.json"), "w") as f:
        json.dump(cfg, f, indent=4)


def main():
    args = parse_args()
    write_config_json(args)

    make_minislabs_multi_entry(
        in_coords=args.in_coords,
        in_vol=args.in_vol,
        out_dir=args.out_dir,
        extract_shape=tuple(args.extract_shape),
        voxel_spacing=args.voxel_spacing,
        extension=args.extension,
        tomo_type=args.tomo_type,
        particle_name=args.particle_name,
        user_id=args.user_id,
        session_id=args.session_id,
        coords_scale=args.coords_scale,
        col_name=args.col_name,
        angles=args.angles,
        gshape=tuple(args.gallery_shape),
        make_stack=args.make_stack,
        invert_contrast=args.invert_contrast,
    )


if __name__ == "__main__":
    main()