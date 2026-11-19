"""
Map CryoSPARC-exported STAR particles back to original tomogram picks.

This script takes selected/refined CryoSPARC particles from a RELION-style STAR
file whose particle rows still contain minislab/micrograph rlnCoordinateX/Y and
rlnAngleRot/Tilt/Psi. It maps each row back through particle_map.csv to recover
the original tomogram rlnCoordinateX/Y/Z from all_picks.star, then writes a STAR
file with tomogram coordinates plus the CryoSPARC orientations.

particle_map.csv is matched back to the minislabs input STAR by tomogram name
and particle order within that tomogram. This handles input STAR files whose
global row order differs from the order used by particle_map.csv.

Example:
  python src/tomo_scripts/map_cs_particles_to_tomo.py \
      --particles-star-cs data/cs-files/Pos_6_2/cs_p117_j283.star \
      --particle-map data/particle_map.csv \
      --minislabs-input-star data/all_picks.star \
      --minislab-config data/make_minislabs.json \
      --out-star data/good_particles_mapped_to_tomo_with_angles.star
"""

import argparse
import json
from pathlib import Path
from typing import Any
import starfile
import numpy as np
import pandas as pd


ORIENTED_STAR_COLS = [
    "rlnCoordinateX",
    "rlnCoordinateY",
    "rlnCoordinateZ",
    "rlnAngleRot",
    "rlnAngleTilt",
    "rlnAnglePsi",
    "rlnTomoName",
]


def load_minislab_config(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        config = json.load(f)
    return config.get("parameters", config)


def gallery_shape_from_config(config: dict[str, Any]) -> tuple[int, int] | None:
    gallery_shape = config.get("gallery_shape")
    if gallery_shape is None:
        return None
    if len(gallery_shape) != 2:
        raise ValueError("make_minislabs config gallery_shape must contain two values.")
    return int(gallery_shape[0]), int(gallery_shape[1])


def infer_gallery_shape_from_particle_map(
    particle_map: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[int, int]:
    required = ["row", "col"]
    missing = [col for col in required if col not in particle_map.columns]
    if missing:
        raise ValueError(f"particle_map.csv is missing required columns: {missing}")

    cols = int(particle_map["col"].max()) + 1
    rows = int(particle_map["row"].max()) + 1

    config_shape = gallery_shape_from_config(config)
    if config_shape is None:
        return cols, rows
    if config_shape not in {(cols, rows), (rows, cols)}:
        raise ValueError(
            f"particle_map implies gallery shape (cols, rows)=({cols}, {rows}), "
            f"but the minislab config has gallery_shape={config_shape}."
        )
    if config_shape == (rows, cols):
        print("[INFO] Interpreting minislab config gallery_shape as [rows, cols].")
    return cols, rows


def minislab_tile_pixels_from_config(config: dict[str, Any]) -> tuple[float, float]:
    extract_shape = config.get("extract_shape")
    voxel_spacing = config.get("voxel_spacing")
    if extract_shape is None or voxel_spacing is None:
        raise ValueError(
            "Need extract_shape and voxel_spacing in make_minislabs config to "
            "convert STAR pixel coordinates."
        )
    return float(extract_shape[0]) / float(voxel_spacing), float(extract_shape[1]) / float(
        voxel_spacing
    )


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
            "Specify one with --particles-block."
        )

    normalized = normalize_block_name(block_name)
    candidates = [block_name, normalized, f"data_{normalized}"]
    for candidate in candidates:
        if candidate in data:
            return data[candidate]
    raise ValueError(f"STAR block {block_name!r} not found in {path}. Blocks: {list(data)}")


def read_all_picks_star(path: Path) -> pd.DataFrame:
    all_picks = read_star_dataframe(path)
    required = ["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ", "rlnTomoName"]
    missing = [col for col in required if col not in all_picks.columns]
    if missing:
        raise ValueError(f"all_picks.star is missing required columns: {missing}")

    all_picks = all_picks.copy()
    for col in ["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"]:
        all_picks[col] = pd.to_numeric(all_picks[col])
    all_picks["minislabs_input_row"] = np.arange(len(all_picks), dtype=np.int64)
    all_picks["tomogram_particle_order"] = all_picks.groupby(
        all_picks["rlnTomoName"].astype(str),
        sort=False,
    ).cumcount()
    return all_picks


def decode_gallery_from_micrograph_path(paths: pd.Series | np.ndarray) -> np.ndarray:
    paths = np.asarray(paths)
    if paths.dtype.kind == "S":
        paths = np.char.decode(paths, encoding="utf-8")

    galleries = pd.Series(paths.astype(str)).str.extract(
        r"particles_(\d+)\.mrc$",
        expand=False,
    )
    if galleries.isna().any():
        bad_paths = pd.Series(paths.astype(str))[galleries.isna()].head().tolist()
        raise ValueError(
            f"Could not decode gallery number from these micrograph paths: {bad_paths}"
        )
    return galleries.astype(int).to_numpy(dtype=np.int64)


def particles_star_to_gallery_df(
    star_particles: pd.DataFrame,
    gallery_shape: tuple[int, int],
    config: dict[str, Any],
) -> pd.DataFrame:
    required = [
        "rlnMicrographName",
        "rlnCoordinateX",
        "rlnCoordinateY",
        "rlnAngleRot",
        "rlnAngleTilt",
        "rlnAnglePsi",
    ]
    missing = [col for col in required if col not in star_particles.columns]
    if missing:
        raise ValueError(f"Particles STAR is missing required columns: {missing}")

    cols, rows = gallery_shape
    tile_x, tile_y = minislab_tile_pixels_from_config(config)

    out = star_particles.copy().reset_index(drop=True)
    out["cs_star_index"] = np.arange(len(out), dtype=np.int64)
    out["cs_index"] = out["cs_star_index"]
    out["location/micrograph_path"] = out["rlnMicrographName"].astype(str)
    out["gallery"] = decode_gallery_from_micrograph_path(
        out["location/micrograph_path"].to_numpy()
    )
    out["location/center_x_frac"] = out["rlnCoordinateX"].astype(float) / (tile_x * cols)
    out["location/center_y_frac"] = out["rlnCoordinateY"].astype(float) / (tile_y * rows)
    return out


def add_expected_gallery_centers(
    particle_map: pd.DataFrame,
    gallery_shape: tuple[int, int],
) -> pd.DataFrame:
    cols, rows = gallery_shape
    out = particle_map.copy().reset_index(drop=True)
    out["expected_center_x_frac"] = (out["col"].astype(float) + 0.5) / cols
    out["expected_center_y_frac"] = (out["row"].astype(float) + 0.5) / rows
    return out


def match_star_to_particle_map(
    star_df: pd.DataFrame,
    particle_map: pd.DataFrame,
    gallery_shape: tuple[int, int],
) -> pd.DataFrame:
    required = ["gallery", "particle", "tomogram", "row", "col"]
    missing = [col for col in required if col not in particle_map.columns]
    if missing:
        raise ValueError(f"particle_map.csv is missing required columns: {missing}")

    particle_map = particle_map.copy().reset_index(drop=True)
    particle_map["particle_map_row"] = np.arange(len(particle_map), dtype=np.int64)
    particle_map["tomogram_particle_order"] = particle_map.groupby(
        particle_map["tomogram"].astype(str),
        sort=False,
    ).cumcount()
    particle_map = add_expected_gallery_centers(particle_map, gallery_shape)
    matched_groups = []

    for gallery, picks in star_df.groupby("gallery", sort=True):
        candidates = particle_map[particle_map["gallery"] == gallery].copy()
        if candidates.empty:
            print(
                f"[WARN] No particle_map entries for gallery {gallery}; "
                f"skipping {len(picks)} STAR rows."
            )
            continue

        picks = picks.reset_index(drop=True)
        candidates = candidates.reset_index(drop=True)
        pick_xy = picks[
            ["location/center_x_frac", "location/center_y_frac"]
        ].to_numpy(dtype=float)
        candidate_xy = candidates[
            ["expected_center_x_frac", "expected_center_y_frac"]
        ].to_numpy(dtype=float)

        distances = np.linalg.norm(pick_xy[:, None, :] - candidate_xy[None, :, :], axis=2)
        pick_order, candidate_order = np.unravel_index(
            np.argsort(distances, axis=None),
            distances.shape,
        )

        assigned_candidate = np.full(len(picks), -1, dtype=np.int64)
        assigned_distance = np.full(len(picks), np.nan, dtype=float)
        used_candidates = np.zeros(len(candidates), dtype=bool)

        for pick_idx, candidate_idx in zip(pick_order, candidate_order):
            if assigned_candidate[pick_idx] != -1 or used_candidates[candidate_idx]:
                continue
            assigned_candidate[pick_idx] = candidate_idx
            assigned_distance[pick_idx] = distances[pick_idx, candidate_idx]
            used_candidates[candidate_idx] = True
            if (assigned_candidate >= 0).all():
                break

        keep = assigned_candidate >= 0
        matched = picks.loc[keep].copy()
        candidate_rows = candidates.iloc[assigned_candidate[keep]].reset_index(drop=True)
        matched = pd.concat(
            [matched.reset_index(drop=True), candidate_rows.add_suffix("_map")],
            axis=1,
        )
        matched["match_distance_frac"] = assigned_distance[keep]
        matched_groups.append(matched)

    if not matched_groups:
        return pd.DataFrame()

    matched = pd.concat(matched_groups, ignore_index=True)
    rename = {
        "tomogram_map": "tomogram",
        "particle_map": "particle",
        "tilt_map": "tilt",
        "gallery_map": "gallery_from_particle_map",
        "row_map": "row",
        "col_map": "col",
        "particle_map_row_map": "particle_map_row",
        "tomogram_particle_order_map": "tomogram_particle_order",
    }
    return matched.rename(columns={k: v for k, v in rename.items() if k in matched.columns})


def map_particle_map_to_all_picks(
    matched: pd.DataFrame,
    all_picks: pd.DataFrame,
) -> pd.DataFrame:
    required = ["particle", "tomogram"]
    missing = [col for col in required if col not in matched.columns]
    if missing:
        raise ValueError(f"Matched table is missing columns required for STAR join: {missing}")
    if "tomogram_particle_order" not in matched.columns:
        raise ValueError(
            "Matched table is missing tomogram_particle_order from particle_map.csv."
        )

    star_cols = [
        "rlnTomoName",
        "tomogram_particle_order",
        "minislabs_input_row",
        "rlnCoordinateX",
        "rlnCoordinateY",
        "rlnCoordinateZ",
    ]
    missing_star_cols = [col for col in star_cols if col not in all_picks.columns]
    if missing_star_cols:
        raise ValueError(f"minislabs input STAR is missing required columns: {missing_star_cols}")

    matched_for_tomo = matched.drop(columns=["rlnCoordinateX", "rlnCoordinateY"], errors="ignore")
    all_picks_lookup = all_picks[star_cols].rename(columns={"rlnTomoName": "tomogram"})
    matched_tomograms = set(matched_for_tomo["tomogram"].astype(str).unique())
    input_tomograms = set(all_picks_lookup["tomogram"].astype(str).unique())
    missing_tomograms = sorted(matched_tomograms - input_tomograms)
    if missing_tomograms:
        available_preview = sorted(input_tomograms)[:10]
        raise ValueError(
            "particle_map.csv contains tomograms that are not present in the "
            "minislabs input STAR. This usually means --minislabs-input-star "
            "does not match the particle_map.csv.\n"
            f"Missing tomograms, first 10: {missing_tomograms[:10]}\n"
            f"Input STAR tomograms, first 10: {available_preview}"
        )

    mapped = matched_for_tomo.merge(
        all_picks_lookup,
        on=["tomogram", "tomogram_particle_order"],
        how="left",
        validate="many_to_one",
    )

    missing_star = mapped["rlnCoordinateX"].isna()
    if missing_star.any():
        bad = mapped.loc[
            missing_star,
            ["tomogram", "tomogram_particle_order", "particle"],
        ].head()
        raise ValueError(
            f"{missing_star.sum()} matched particles did not exist in minislabs input STAR "
            "when matching by tomogram and order-within-tomogram. Examples:\n"
            f"{bad}"
        )

    mapped["rlnTomoName"] = mapped["tomogram"]

    ordered_cols = [
        "cs_star_index",
        "particle",
        "rlnTomoName",
        "rlnCoordinateX",
        "rlnCoordinateY",
        "rlnCoordinateZ",
        "rlnAngleRot",
        "rlnAngleTilt",
        "rlnAnglePsi",
        "gallery",
        "gallery_from_particle_map",
        "row",
        "col",
        "match_distance_frac",
        "location/center_x_frac",
        "location/center_y_frac",
        "location/micrograph_path",
    ]
    ordered_cols = [col for col in ordered_cols if col in mapped.columns]
    return mapped[ordered_cols + [col for col in mapped.columns if col not in ordered_cols]]


def report_large_match_distances(
    matched: pd.DataFrame,
    gallery_shape: tuple[int, int],
    threshold: float | None,
) -> None:
    if threshold is None:
        threshold = 0.5 / min(gallery_shape)
    large_distance = matched["match_distance_frac"] > threshold
    if not large_distance.any():
        return

    print(
        f"[WARN] {large_distance.sum()} STAR rows matched farther than "
        f"{threshold:.4f} fraction units."
    )
    preview_cols = [
        "cs_star_index",
        "gallery",
        "rlnCoordinateX",
        "rlnCoordinateY",
        "particle",
        "row",
        "col",
        "match_distance_frac",
    ]
    preview_cols = [col for col in preview_cols if col in matched.columns]
    preview = (
        matched.loc[large_distance, preview_cols]
        .sort_values("match_distance_frac", ascending=False)
        .head(20)
    )
    print(preview.to_string(index=False))


def default_output_path(particles_star: Path) -> Path:
    output_dir = particles_star.parent
    stem = particles_star.stem
    return output_dir / f"{stem}_mapped_to_tomo_with_angles.star"


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Map a CryoSPARC-exported particles STAR from minislab gallery images "
            "back to original tomogram coordinates, preserving Rot/Tilt/Psi."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--particles-star-cs",
        required=True,
        dest="particles_star_cs",
        help="Input CryoSPARC particles STAR file.",
    )
    parser.add_argument(
        "--particles-block",
        default="particles",
        help=(
            "STAR block containing particle rows. Use particles for data_particles; "
            "data_particles is also accepted."
        ),
    )
    parser.add_argument(
        "--particle-map",
        required=True,
        help="Input particle_map.csv generated when making minislab galleries.",
    )
    parser.add_argument(
        "--minislabs-input-star",
        required=True,
        dest="minislabs_input_star",
        help="Input minislabs source STAR with original tomogram coordinates.",
    )
    parser.add_argument(
        "--minislab-config",
        required=True,
        help=(
            "Input make_minislabs.json. Requires extract_shape, voxel_spacing, "
            "and preferably gallery_shape."
        ),
    )
    parser.add_argument(
        "--out-star",
        default=None,
        help=(
            "Output STAR path. If omitted, writes "
            "<particles-star-cs-stem>_mapped_to_tomo_with_angles.star beside "
            "--particles-star-cs."
        ),
    )
    parser.add_argument(
        "--max-match-distance-frac",
        type=float,
        default=None,
        help=(
            "Warn when a STAR row is matched farther than this fraction-unit distance. "
            "If omitted, uses half a tile in the smaller gallery dimension."
        ),
    )
    return parser


def run(args: argparse.Namespace) -> None:

    particles_star = Path(args.particles_star_cs)
    particle_map_file = Path(args.particle_map)
    all_picks_star = Path(args.minislabs_input_star)
    minislab_config = Path(args.minislab_config)
    default_star = default_output_path(particles_star)
    out_star = Path(args.out_star) if args.out_star else default_star

    config = load_minislab_config(minislab_config)
    star_particles = read_star_dataframe(particles_star, args.particles_block)
    particle_map = pd.read_csv(particle_map_file)
    all_picks = read_all_picks_star(all_picks_star)
    gallery_shape = infer_gallery_shape_from_particle_map(particle_map, config)
    star_df = particles_star_to_gallery_df(star_particles, gallery_shape, config)

    print(f"[INFO] Gallery shape used for matching (cols, rows): {gallery_shape}")
    print(f"[INFO] CryoSPARC STAR particles: {len(star_df)}")
    print(f"[INFO] Particle map rows: {len(particle_map)}")
    print(f"[INFO] All picks rows: {len(all_picks)}")
    print(f"[INFO] CryoSPARC galleries: {sorted(star_df['gallery'].unique().tolist())}")

    matched_to_map = match_star_to_particle_map(star_df, particle_map, gallery_shape)
    if matched_to_map.empty:
        raise ValueError("No CryoSPARC STAR rows could be matched to particle_map rows.")

    print(
        "[INFO] Matched STAR particles to particle_map rows: "
        f"{len(matched_to_map)} / {len(star_df)}"
    )
    print(f"[INFO] Matched galleries: {matched_to_map['gallery'].nunique()}")
    print(
        "[INFO] Largest tile-center distance in fraction units: "
        f"{matched_to_map['match_distance_frac'].max():.6g}"
    )
    report_large_match_distances(
        matched_to_map,
        gallery_shape,
        args.max_match_distance_frac,
    )

    good_particles = map_particle_map_to_all_picks(matched_to_map, all_picks)
    missing_output_cols = [
        col for col in ORIENTED_STAR_COLS if col not in good_particles.columns
    ]
    if missing_output_cols:
        raise ValueError(f"Cannot write output STAR; missing columns: {missing_output_cols}")

    out_star.parent.mkdir(parents=True, exist_ok=True)
    starfile.write(good_particles[ORIENTED_STAR_COLS], out_star, overwrite=True)

    print(f"[OK] Particles mapped to original tomogram picks: {len(good_particles)}")
    print(f"[OK] Unique original tomograms: {good_particles['rlnTomoName'].nunique()}")
    print(f"[DONE] Wrote STAR: {out_star}")


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()