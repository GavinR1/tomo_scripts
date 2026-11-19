import argparse
from pathlib import Path

import numpy as np


def _import_mrcfile():
    try:
        import mrcfile
    except ImportError as exc:
        raise ImportError(
            "notch_filter_z_tomogram requires the 'mrcfile' package. Install it with "
            "'pip install mrcfile'."
        ) from exc
    return mrcfile


def _read_volume(path_in: Path) -> tuple[np.ndarray, float]:
    mrcfile = _import_mrcfile()
    with mrcfile.open(path_in, permissive=True) as mrc:
        volume = np.asarray(mrc.data, dtype=np.float32)
        voxel_size = float(mrc.voxel_size.x)

    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D MRC tomogram, got shape {volume.shape}.")
    if not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError(
            f"Invalid voxel size in MRC header: {voxel_size}. "
            "Pass --voxel-size manually."
        )

    return volume, voxel_size


def notch_filter_z_band(
    volume: np.ndarray,
    voxel_size: float,
    f_low: float,
    f_high: float,
    taper: float,
    depth: float,
) -> np.ndarray:
    if f_low <= 0 or f_high <= 0:
        raise ValueError("Frequency bounds must be positive.")
    if f_low >= f_high:
        raise ValueError("--f-low must be less than --f-high.")
    if taper <= 0:
        raise ValueError("--taper must be positive.")
    if not 0 <= depth <= 1:
        raise ValueError("--depth must be between 0 and 1.")

    v = volume.astype(np.float32, copy=False)
    mean = float(np.mean(v))
    v0 = v - mean

    F = np.fft.fftn(v0)

    nz = v.shape[0]
    fz = np.fft.fftfreq(nz, d=voxel_size)
    af = np.abs(fz)

    left = 1.0 / (1.0 + np.exp(-(af - f_low) / taper))
    right = 1.0 / (1.0 + np.exp((af - f_high) / taper))
    band = left * right

    notch_1d = 1.0 - depth * band
    filtered = np.fft.ifftn(F * notch_1d[:, None, None]).real + mean

    return filtered.astype(np.float32)


def _write_volume(path_out: Path, volume: np.ndarray, path_template: Path, voxel_size: float) -> None:
    mrcfile = _import_mrcfile()
    path_out.parent.mkdir(parents=True, exist_ok=True)

    with mrcfile.open(path_template, permissive=True) as src:
        with mrcfile.new(path_out, overwrite=True) as dst:
            dst.set_data(volume.astype(np.float32, copy=False))
            np.copyto(dst.header, src.header)
            dst.header.nx = volume.shape[2]
            dst.header.ny = volume.shape[1]
            dst.header.nz = volume.shape[0]
            dst.header.mx = volume.shape[2]
            dst.header.my = volume.shape[1]
            dst.header.mz = volume.shape[0]
            dst.header.mode = 2
            dst.voxel_size = voxel_size
            dst.update_header_stats()


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Apply a smooth notch filter along tomogram Z/beam direction to suppress "
            "periodic reconstruction stripes before subvolume extraction."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--in",
        dest="path_in",
        required=True,
        help="Input 3D tomogram MRC file.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output filtered tomogram MRC file.",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=None,
        help="Voxel size in Angstrom/pixel. If omitted, uses the input MRC header.",
    )
    parser.add_argument(
        "--f-low",
        type=float,
        default=0.052,
        help="Lower notch frequency in cycles/Angstrom.",
    )
    parser.add_argument(
        "--f-high",
        type=float,
        default=0.064,
        help="Upper notch frequency in cycles/Angstrom.",
    )
    parser.add_argument(
        "--taper",
        type=float,
        default=0.006,
        help="Smooth logistic edge width in cycles/Angstrom.",
    )
    parser.add_argument(
        "--depth",
        type=float,
        default=0.6,
        help="Notch strength. 0 leaves the tomogram unchanged; 1 fully removes the band.",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    path_in = Path(args.path_in)
    path_out = Path(args.out)

    volume, header_voxel_size = _read_volume(path_in)
    voxel_size = args.voxel_size if args.voxel_size is not None else header_voxel_size

    filtered = notch_filter_z_band(
        volume=volume,
        voxel_size=voxel_size,
        f_low=args.f_low,
        f_high=args.f_high,
        taper=args.taper,
        depth=args.depth,
    )
    _write_volume(path_out, filtered, path_template=path_in, voxel_size=voxel_size)

    period_low = 1.0 / args.f_high
    period_high = 1.0 / args.f_low
    print(
        "Wrote filtered tomogram: "
        f"{path_out}\n"
        f"Voxel size: {voxel_size:.6g} A/px\n"
        f"Notch band: {args.f_low:.6g}-{args.f_high:.6g} cycles/A "
        f"({period_low:.2f}-{period_high:.2f} A periodicity, "
        f"{period_low / voxel_size:.2f}-{period_high / voxel_size:.2f} px)\n"
        f"Taper: {args.taper:.6g} cycles/A, depth: {args.depth:.3g}"
    )


def _main_() -> None:
    parser = create_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    _main_()
