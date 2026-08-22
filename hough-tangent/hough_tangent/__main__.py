"""CLI: detect the discs, lock a lattice onto them, draw the tangent outline."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .pipeline import Config, run
from .render import Style


def _grid(text: str) -> tuple[int, int]:
    try:
        rows, cols = (int(v) for v in text.lower().split("x"))
    except ValueError as exc:  # pragma: no cover - argparse reports it
        raise argparse.ArgumentTypeError("grid must look like 4x4") from exc
    return rows, cols


def _cells(text: str) -> np.ndarray:
    rows = [r.strip() for r in text.split(",") if r.strip()]
    if not rows or len({len(r) for r in rows}) != 1:
        raise argparse.ArgumentTypeError("cells must be equal-length rows of 0/1")
    return np.array([[ch != "0" for ch in row] for row in rows], dtype=bool).reshape(-1)


def _colour(text: str) -> tuple[int, int, int]:
    parts = text.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("colour must be B,G,R")
    return tuple(int(v) for v in parts)  # type: ignore[return-value]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m hough_tangent",
        description="Hough circle detection with a lattice prior, outlined by the "
        "exact convex hull of the discs (outer common tangents + arcs).",
    )
    p.add_argument("src", type=Path, help="input video")
    p.add_argument("-o", "--out", type=Path, default=Path("out.mp4"), help="output video")
    p.add_argument(
        "--grid", type=_grid, default=(4, 4), metavar="RxC", help="lattice size, default 4x4"
    )
    p.add_argument(
        "--no-grid",
        action="store_true",
        help="skip the lattice prior and outline the raw Hough hits",
    )
    p.add_argument(
        "--no-refine",
        action="store_true",
        help="skip the guided second look at cells the global pass missed",
    )
    p.add_argument(
        "--cells",
        type=_cells,
        default=None,
        metavar="1111,1011,...",
        help="state the occupancy yourself, one digit per cell, rows "
        "comma separated; overrides what the analysis pass concluded",
    )

    g = p.add_argument_group("resolution")
    g.add_argument(
        "--work-width", type=int, default=1080, help="width the detector runs at, default 1080"
    )
    g.add_argument(
        "--out-width", type=int, default=1080, help="width of the rendered video, default 1080"
    )

    g = p.add_argument_group("range")
    g.add_argument("--start", type=int, default=0, help="first frame")
    g.add_argument("--end", type=int, default=None, help="stop before this frame")
    g.add_argument("--stride", type=int, default=1, help="keep every Nth frame")

    g = p.add_argument_group("hough")
    g.add_argument(
        "--r-min-frac",
        type=float,
        default=0.030,
        help="min disc radius as a fraction of the working width",
    )
    g.add_argument(
        "--r-max-frac",
        type=float,
        default=0.105,
        help="max disc radius as a fraction of the working width",
    )
    g.add_argument(
        "--no-adaptive",
        action="store_true",
        help="do not retune the radius window once the lattice locks",
    )

    g = p.add_argument_group("lattice")
    g.add_argument(
        "--snap",
        type=float,
        default=0.0,
        help="0 keeps the measured centres, 1 snaps them onto the fitted "
        "lattice; the array physically deforms, so 0 is the honest default",
    )
    g.add_argument(
        "--gate", type=float, default=0.45, help="match radius as a fraction of the cell spacing"
    )
    g.add_argument("--min-observed", type=int, default=6, help="detections needed to keep the lock")
    g.add_argument(
        "--max-coast", type=int, default=20, help="frames to hold the last lattice before giving up"
    )

    g = p.add_argument_group("smoothing")
    g.add_argument("--no-smooth", action="store_true", help="disable the One Euro filter")
    g.add_argument(
        "--min-cutoff",
        type=float,
        default=0.9,
        help="lower = steadier when still, more lag when moving",
    )
    g.add_argument(
        "--beta", type=float, default=0.015, help="higher = less lag when moving, more jitter"
    )

    g = p.add_argument_group("drawing")
    g.add_argument(
        "--margin-frac",
        type=float,
        default=0.0,
        help="pad the outline outwards by this fraction of a disc radius",
    )
    g.add_argument(
        "--outline",
        choices=("ring", "hull"),
        default="ring",
        help="ring runs tangent to every disc on the outer ring, following the "
        "lattice; hull is the convex hull, which only touches the discs that "
        "reach it",
    )
    g.add_argument(
        "--corners",
        choices=("sharp", "round"),
        default="sharp",
        help="sharp meets each pair of tangents at their crossing point; round "
        "follows the disc between them",
    )
    g.add_argument(
        "--max-corner",
        type=float,
        default=6.0,
        help="a corner further than this many radii from its disc stays rounded",
    )
    g.add_argument(
        "--tangent-extend",
        type=float,
        default=0.0,
        help="draw each tangent line this many px past its two touch points",
    )
    g.add_argument("--hull-colour", type=_colour, default=None, metavar="B,G,R")
    g.add_argument("--fill-alpha", type=float, default=0.10)
    g.add_argument("--no-circles", action="store_true", help="outline only")
    g.add_argument("--no-tangent-points", action="store_true")
    g.add_argument("--no-hud", action="store_true")
    g.add_argument(
        "--debug", action="store_true", help="also draw the raw Hough candidates in grey"
    )

    p.add_argument("--fourcc", default="avc1", help="H.264 by default; try mp4v if it fails")
    p.add_argument("-q", "--quiet", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows, cols = args.grid
    style = Style(
        fill_alpha=args.fill_alpha,
        tangent_extend=args.tangent_extend,
        show_circles=not args.no_circles,
        show_tangent_points=not args.no_tangent_points,
        show_hud=not args.no_hud,
        sharp_corners=args.corners == "sharp",
        max_corner_reach=args.max_corner,
    )
    if args.hull_colour:
        style.hull = args.hull_colour

    cfg = Config(
        src=args.src,
        dst=args.out,
        rows=rows,
        cols=cols,
        work_width=args.work_width,
        out_width=args.out_width,
        start=args.start,
        end=args.end,
        stride=max(1, args.stride),
        r_min_frac=args.r_min_frac,
        r_max_frac=args.r_max_frac,
        adaptive=not args.no_adaptive,
        snap=args.snap,
        gate_frac=args.gate,
        min_observed=args.min_observed,
        max_coast=args.max_coast,
        cells=args.cells,
        refine=not args.no_refine,
        smooth=not args.no_smooth,
        min_cutoff=args.min_cutoff,
        beta=args.beta,
        outline=args.outline,
        margin_frac=args.margin_frac,
        fourcc=args.fourcc,
        debug=args.debug,
        quiet=args.quiet,
        style=style,
        no_grid=args.no_grid,
    )
    stats = run(cfg)
    if not args.quiet:
        locked = stats.locked
        mean_obs = stats.observed_total / locked if locked else 0.0
        mean_present = stats.present_total / locked if locked else 0.0
        print(
            f"{stats.frames} frames in {stats.seconds:.1f}s "
            f"({stats.frames / max(stats.seconds, 1e-6):.1f} fps)"
        )
        print(
            f"lattice locked on {locked}/{stats.frames} frames "
            f"({100.0 * locked / max(stats.frames, 1):.1f}%), "
            f"coasted {stats.coasted}, mean {mean_obs:.1f}/{mean_present:.0f} "
            f"discs detected per frame"
        )
        if stats.ring_frames:
            print(
                f"outline ran tangent to every ring disc on {stats.ring_frames} frames; "
                f"worst stand-off {stats.max_standoff:.2f} px "
                f"(only where a disc sits recessed between its neighbours)"
            )
        print(f"wrote {cfg.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
