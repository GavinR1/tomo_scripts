"""
Convert AreTomo .aln (global alignment section) to IMOD .xf transform file.

Example:
    python aln_to_xf.py --in_aln Position_001_EVN_ali.aln --out_xf Position_001_EVN.xf
    or
    for f in *.aln; do python aln2xf.py --in_aln "$f" --out_xf "$(basename "$f" | sed -E 's/^.*(24mar.*)\.aln$/\1.xf/')"; done
"""

import argparse
import math
import re
from pathlib import Path


def parse_aln(aln_path):
    """Parse AreTomo .aln file and extract global alignment table rows."""
    rows = []
    grab = False
    with open(aln_path, "r") as f:
        for line in f:
            if re.match(r'^\s*#\s*SEC', line):
                grab = True
                continue
            if grab:
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


def to_xf_row(rot_deg, gmag, tx, ty, scale=None):
    """Convert AreTomo ROT/GMAG/TX/TY to IMOD a,b,c,d,dx,dy row."""
    s = gmag * (scale if scale is not None else 1.0)
    theta = math.radians(rot_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    a = s * cos_t
    b = s * sin_t
    c = -s * sin_t
    d = s * cos_t

    dx = -(tx * cos_t + ty * sin_t)
    dy = -(-tx * sin_t + ty * cos_t)
    return a, b, c, d, dx, dy


def main():
    parser = argparse.ArgumentParser(
        description="Convert AreTomo .aln file (global table) to IMOD .xf format."
    )
    parser.add_argument(
        "--in_aln", type=Path, required=True,
        help="Input AreTomo .aln file"
    )
    parser.add_argument(
        "--out_xf", type=Path, required=True,
        help="Output IMOD .xf file"
    )
    parser.add_argument(
        "--scale", type=float, default=None,
        help="Binning multiplier if applicable"
    )

    args = parser.parse_args()

    rows = parse_aln(args.in_aln)
    if not rows:
        raise SystemExit("No global alignment rows found in input .aln file.")

    with open(args.out_xf, "w") as f:
        for _, rot, gmag, tx, ty in rows:
            a, b, c, d, dx, dy = to_xf_row(rot, gmag, tx, ty, args.scale)
            f.write(f"{a:9.5f} {b:9.5f} {c:9.5f} {d:9.5f} {dx:10.3f} {dy:10.3f}\n")

    print(f"Wrote {len(rows)} transforms to {args.out_xf}")


if __name__ == "__main__":
    main()
