"""Extract the TILT column from an AreTomo .aln file and write an IMOD .tlt file.

This script reads the first (global) alignment table in an AreTomo .aln file,
extracts the 'TILT' column, and writes one tilt angle per line to the output
.tlt file, formatted with two decimal places and a field width that matches
existing IMOD .tlt examples.

Example:
        python aln2tlt.py --in_aln Position_001_EVN_ali.aln --out_tlt Position_001_EVN.tlt
            or
        for f in *.aln; do python aln2tlt.py --in_aln "$f" --out_tlt "$(basename "$f" | sed -E 's/^.*(24mar.*)\.aln$/\1.tlt/')"; done
"""

import argparse
import re
from pathlib import Path


def parse_aln_tilts(aln_path: Path):
    """Parse AreTomo .aln file and extract the TILT column from the first numeric table.

    This is a simple and robust approach: find the first contiguous block of
    non-comment lines where the first token is an integer (the global table),
    then take the last token of each row as the tilt angle.
    """
    rows = []
    table_started = False
    with open(aln_path, "r") as fh:
        for raw in fh:
            line = raw.strip()
            # skip empty/comment lines before the table
            if not line or line.startswith('#'):
                if table_started:
                    break
                continue
            first_tok = line.split()[0]
            try:
                int(first_tok)
            except Exception:
                # not a data row
                if table_started:
                    break
                continue
            table_started = True
            rows.append(line.split())

    if not rows:
        raise ValueError(f"No numeric table found in aln file: {aln_path}")

    tilts = []
    for parts in rows:
        # assume tilt is the last column in the data rows
        if not parts:
            continue
        tilt_tok = parts[-1]
        try:
            tilts.append(float(tilt_tok))
        except Exception:
            continue

    if not tilts:
        raise ValueError(f"No tilt angles parsed from aln file: {aln_path}")
    return tilts




def main():
    parser = argparse.ArgumentParser(
        description="Extract the TILT column from an AreTomo .aln and write an IMOD .tlt file."
    )
    parser.add_argument("--in_aln", type=Path, required=True, help="Input AreTomo .aln file")
    parser.add_argument("--out_tlt", type=Path, required=True, help="Output IMOD .tlt file")

    args = parser.parse_args()

    tilts = parse_aln_tilts(args.in_aln)

    # write one tilt angle per line
    with open(args.out_tlt, "w") as f:
        for t in tilts:
            f.write(f"{t:7.2f}\n")

    print(f"Wrote {len(tilts)} tilt angles to {args.out_tlt}")


if __name__ == "__main__":
    main()
