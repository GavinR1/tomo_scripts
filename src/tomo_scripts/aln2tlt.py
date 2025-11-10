"""
Convert AreTomo .aln (global alignment section) to IMOD-style .tlt transform file
(a11 a12 a21 a22 dx dy per section).

Example:
    python aln2tlt.py --in_aln Position_001_EVN_ali.aln --out_tlt Position_001_EVN.tlt
    or
    for f in *.aln; do python aln2tlt.py --in_aln "$f" --out_xf "$(basename "$f" | sed -E 's/^.*(24mar.*)\.aln$/\1.tlt/')"; done

Batch:
    for f in *.aln; do \
      python aln2tlt.py --in_aln "$f" \
        --out_tlt "$(basename "$f" .aln).tlt"; \
    done

Notes:
- The 2×2 block is built from ROT (degrees): [[cos, sin], [-sin, cos]].
- dx,dy are computed as -(R * [TX, TY]).
- If GMAG (AreTomo magnification) and/or a binning --scale are present, they
  multiply only the 2×2 block (a11..a22), not the translations.
"""

import argparse
import math
import re
from pathlib import Path
from typing import Optional


def parse_aln(aln_path: Path):
    """Parse AreTomo .aln file and extract rows from the global alignment table."""
    rows = []
    in_global = False
    with open(aln_path, "r") as f:
        for line in f:
            if re.match(r"^\s*#\s*SEC", line):
                in_global = True
                continue
            if in_global:
                if not line.strip() or line.lstrip().startswith("#"):
                    break
                parts = line.split()
                if len(parts) < 5:
                    continue
                sec = int(parts[0])
                rot = float(parts[1])
                gmag = float(parts[2])
                tx = float(parts[3])
                ty = float(parts[4])
                rows.append((sec, rot, gmag, tx, ty))
    rows.sort(key=lambda r: r[0])
    return rows


def to_tlt_row(rot_deg: float, gmag: float, tx: float, ty: float, scale: Optional[float]):
    """
    Convert AreTomo ROT/GMAG/TX/TY to IMOD .tlt row (a11 a12 a21 a22 dx dy).

    Matrix = s * [[cos, sin], [-sin, cos]], where s = gmag * scale (if provided).
    Translations are -(R * [tx, ty]) (not multiplied by s).
    """
    s = gmag * (scale if scale is not None else 1.0)
    th = math.radians(rot_deg)
    c, si = math.cos(th), math.sin(th)

    a11 = s * c
    a12 = s * si
    a21 = -s * si
    a22 = s * c

    dx = -(c * tx + si * ty)
    dy = -(-si * tx + c * ty)
    return a11, a12, a21, a22, dx, dy


def main():
    parser = argparse.ArgumentParser(
        description="Convert AreTomo .aln (global table) to IMOD-style .tlt (a11 a12 a21 a22 dx dy)."
    )
    parser.add_argument("--in_aln", type=Path, required=True, help="Input AreTomo .aln file")
    parser.add_argument("--out_tlt", type=Path, required=True, help="Output IMOD .tlt file")
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Optional extra multiplicative scale (e.g., binning) applied to the 2×2 matrix only.",
    )

    args = parser.parse_args()

    rows = parse_aln(args.in_aln)
    if not rows:
        raise SystemExit("❌ No global alignment rows found in input .aln file.")

    with open(args.out_tlt, "w") as f:
        for _, rot, gmag, tx, ty in rows:
            a11, a12, a21, a22, dx, dy = to_tlt_row(rot, gmag, tx, ty, args.scale)
            f.write(f"{a11:9.3f} {a12:9.3f} {a21:9.3f} {a22:9.3f} {dx:9.2f} {dy:9.2f}\n")

    print(f"Wrote {len(rows)} transforms to {args.out_tlt}")


if __name__ == "__main__":
    main()
