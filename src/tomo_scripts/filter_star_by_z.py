"""
Filter STAR particles by rlnCoordinateZ range.

Example:
  python src/tomo_scripts/filter_star_by_z.py \
      --star data/particles.star \
      --min-z 50 \
      --max-z 300 \
      --out data/particles_z50_300.star
"""

import argparse
from pathlib import Path

import pandas as pd


Z_COLUMN = "rlnCoordinateZ"


def normalize_block_name(block_name: str) -> str:
    return block_name.removeprefix("data_")


def read_star_dataframe(path: Path, block_name: str | None = None) -> pd.DataFrame:
    import starfile

    data = starfile.read(path)
    if isinstance(data, pd.DataFrame):
        return data

    if not isinstance(data, dict):
        raise TypeError(f"Unexpected object from starfile.read({path}): {type(data)!r}")

    if block_name is None:
        if len(data) == 1:
            return next(iter(data.values()))
        if "particles" in data:
            return data["particles"]
        raise ValueError(
            f"{path} contains multiple STAR blocks {list(data)}. "
            "Specify one with --block."
        )

    normalized = normalize_block_name(block_name)
    candidates = [block_name, normalized, f"data_{normalized}"]
    for candidate in candidates:
        if candidate in data:
            return data[candidate]
    raise ValueError(f"STAR block {block_name!r} not found in {path}. Blocks: {list(data)}")


def filter_by_z(df: pd.DataFrame, min_z: float, max_z: float) -> pd.DataFrame:
    if Z_COLUMN not in df.columns:
        raise ValueError(f"STAR file is missing required column {Z_COLUMN!r}.")
    if min_z > max_z:
        raise ValueError(f"--min-z must be <= --max-z, got {min_z} > {max_z}.")

    z = pd.to_numeric(df[Z_COLUMN])
    keep = z.between(min_z, max_z, inclusive="both")
    return df.loc[keep].copy()


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Filter a STAR file to particles with rlnCoordinateZ within a range.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--star",
        required=True,
        help="Input STAR file.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output filtered STAR file.",
    )
    parser.add_argument(
        "--min-z",
        required=True,
        type=float,
        help="Minimum rlnCoordinateZ to keep, inclusive.",
    )
    parser.add_argument(
        "--max-z",
        required=True,
        type=float,
        help="Maximum rlnCoordinateZ to keep, inclusive.",
    )
    parser.add_argument(
        "--block",
        default=None,
        help=(
            "Optional STAR block name to read. For data_particles, use particles "
            "or data_particles."
        ),
    )
    return parser


def run(args: argparse.Namespace) -> None:
    import starfile

    path_star = Path(args.star)
    path_out = Path(args.out)

    df = read_star_dataframe(path_star, args.block)
    filtered = filter_by_z(df, args.min_z, args.max_z)

    path_out.parent.mkdir(parents=True, exist_ok=True)
    starfile.write(filtered, path_out, overwrite=True)

    removed = len(df) - len(filtered)
    print(
        f"[DONE] Kept {len(filtered)} / {len(df)} particles with "
        f"{args.min_z} <= {Z_COLUMN} <= {args.max_z}."
    )
    print(f"[DONE] Removed {removed} particles.")
    print(f"[DONE] Wrote STAR: {path_out}")


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
