"""Video in, annotated video out."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .detect import DetectParams, detect_circles, refine_in_roi
from .grid import GridModel, GridTracker, ring_order
from .hull import hull_of_disks, tangent_ring
from .render import Style, draw_overlay
from .smooth import OneEuro


@dataclass
class Config:
    src: Path
    dst: Path
    rows: int = 4
    cols: int = 4
    work_width: int = 1080
    out_width: int = 1080
    start: int = 0
    end: int | None = None
    stride: int = 1
    # Radius search window, as a fraction of the working-resolution width.
    r_min_frac: float = 0.030
    r_max_frac: float = 0.105
    adaptive: bool = True  # narrow the Hough window once the lattice locks
    snap: float = 0.0  # 0 = trust the detections, 1 = snap to the lattice
    gate_frac: float = 0.45
    min_observed: int = 6
    max_coast: int = 20
    refine: bool = True  # hunt for missed cells in a window around their prediction
    prescan: int = 60  # frames sampled up front to learn which cells are empty
    cells: np.ndarray | None = None  # explicit occupancy mask, skips the prescan
    smooth: bool = True
    min_cutoff: float = 0.9
    beta: float = 0.015
    outline: str = "ring"  # "ring" touches every outer disc, "hull" is the convex hull
    margin_frac: float = 0.0  # pad the outline outwards by this fraction of a radius
    fourcc: str = "avc1"
    debug: bool = False
    quiet: bool = False
    style: Style = field(default_factory=Style)
    no_grid: bool = False  # detect only, no lattice prior (diagnostic)


@dataclass
class Stats:
    frames: int = 0
    locked: int = 0
    coasted: int = 0
    observed_total: int = 0
    present_total: int = 0
    ring_frames: int = 0
    max_standoff: float = 0.0
    seconds: float = 0.0
    occupancy: np.ndarray | None = None


def _tracker(cfg: Config) -> GridTracker:
    return GridTracker(
        cfg.rows,
        cfg.cols,
        snap=cfg.snap,
        gate_frac=cfg.gate_frac,
        min_observed=cfg.min_observed,
        max_coast=cfg.max_coast,
    )


def _adapt(params: DetectParams, model: GridModel, radii: np.ndarray) -> None:
    med = float(np.median(radii))
    params.r_min = max(6.0, 0.65 * med)
    params.r_max = 1.45 * med
    params.min_dist = max(0.5 * model.spacing, 1.4 * params.r_min)


def _open(src: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {src}")
    return cap


def learn_occupancy(
    cfg: Config, work_size: tuple[int, int], params: DetectParams, total: int
) -> np.ndarray:
    """Sample the clip to find which lattice cells actually hold a disc.

    A 4x4 lattice does not mean 16 discs.  Without this the tracker draws a
    phantom circle in every empty cell for the first frames, until the running
    occupancy estimate catches up.
    """
    lo = cfg.start
    hi = cfg.end if cfg.end is not None else total
    n = int(min(cfg.prescan, max(hi - lo, 1)))
    cap = _open(cfg.src)
    tracker = _tracker(cfg)
    try:
        for i in np.linspace(lo, max(hi - 1, lo), n).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            if not ok:
                continue
            work = cv2.resize(frame, work_size, interpolation=cv2.INTER_AREA)
            det = detect_circles(work, params)
            if cfg.refine:
                tracker.refiner = lambda c, r, w=work, g=det.gray: refine_in_roi(w, g, c, r)
            model = tracker.update(det.circles)
            if model is not None and cfg.adaptive:
                _adapt(params, model, model.radii)
    finally:
        cap.release()
    if tracker.locked_frames < max(4, n // 4):
        return np.ones(cfg.rows * cfg.cols, dtype=bool)
    return tracker.occupancy >= tracker.vacancy_threshold


def _make_writer(path: Path, fourcc: str, fps: float, size: tuple[int, int]):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"could not open {path} for writing with fourcc {fourcc!r}")
    return writer


def run(cfg: Config) -> Stats:
    cap = _open(cfg.src)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    work_w = min(cfg.work_width, src_w)
    work_size = (work_w, max(1, round(src_h * work_w / src_w)))
    out_w = min(cfg.out_width, src_w)
    out_size = (out_w, max(1, round(src_h * out_w / src_w)))
    to_out = out_w / work_w
    px = out_w / 1080.0

    params = DetectParams(r_min=cfg.r_min_frac * work_w, r_max=cfg.r_max_frac * work_w)
    params.min_dist = 1.6 * params.r_min

    occupied: np.ndarray | None = None
    if not cfg.no_grid:
        if cfg.cells is not None:
            occupied = cfg.cells
        elif cfg.prescan > 0:
            if not cfg.quiet:
                print(f"  prescanning {cfg.prescan} frames for cell occupancy...")
            occupied = learn_occupancy(cfg, work_size, params, total)
            if not cfg.quiet:
                grid = occupied.reshape(cfg.rows, cfg.cols)
                for row in grid:
                    print("    " + " ".join("#" if v else "." for v in row))

    outline_cells = (
        ring_order(cfg.rows, cfg.cols) if cfg.outline == "ring" and not cfg.no_grid else []
    )

    tracker = _tracker(cfg)
    if occupied is not None:
        tracker.forced_occupied = occupied
    pos_filter = OneEuro(fps / cfg.stride, cfg.min_cutoff, cfg.beta)
    rad_filter = OneEuro(fps / cfg.stride, cfg.min_cutoff, cfg.beta * 0.5)
    generation = -1

    cfg.dst.parent.mkdir(parents=True, exist_ok=True)
    writer = _make_writer(cfg.dst, cfg.fourcc, fps / cfg.stride, out_size)

    if cfg.start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, cfg.start)
    last = cfg.end if cfg.end is not None else total
    stats = Stats(occupancy=occupied)
    t0 = time.time()
    idx = cfg.start
    try:
        while idx < last:
            ok, frame = cap.read()
            if not ok:
                break
            if (idx - cfg.start) % cfg.stride:
                idx += 1
                continue

            work = cv2.resize(frame, work_size, interpolation=cv2.INTER_AREA)
            det = detect_circles(work, params)

            centres = np.zeros((0, 2))
            radii = np.zeros(0)
            observed = None
            present_cells = None

            if cfg.no_grid:
                centres, radii = det.circles[:, :2], det.circles[:, 2]
                status = f"hough only: {len(det.circles)}/{len(det.raw)}"
            else:
                if cfg.refine:
                    tracker.refiner = lambda c, r, w=work, g=det.gray: refine_in_roi(w, g, c, r)
                model = tracker.update(det.circles)
                if model is None:
                    status = "searching"
                    pos_filter.reset()
                    rad_filter.reset()
                else:
                    if tracker.generation != generation:
                        generation = tracker.generation
                        pos_filter.reset()
                        rad_filter.reset()
                    nodes, node_r = model.nodes, model.radii
                    if cfg.smooth:
                        nodes = pos_filter(nodes)
                        node_r = rad_filter(node_r)
                    present = model.present
                    present_cells = np.flatnonzero(present)
                    centres, radii, observed = (
                        nodes[present],
                        node_r[present],
                        model.observed[present],
                    )
                    n_obs, n_present = int(observed.sum()), int(present.sum())
                    stats.locked += 1
                    stats.observed_total += n_obs
                    stats.present_total += n_present
                    status = (
                        f"coasting {tracker.coasting}"
                        if tracker.coasting
                        else f"lock {n_obs}/{n_present}"
                    )
                    if cfg.adaptive:
                        _adapt(params, model, node_r)

            ring: list[int] = []
            if outline_cells and present_cells is not None:
                pos = {int(cell): i for i, cell in enumerate(present_cells)}
                ring = [pos[cell] for cell in outline_cells if cell in pos]

            canvas = cv2.resize(
                frame, out_size, interpolation=cv2.INTER_AREA if out_w < src_w else cv2.INTER_CUBIC
            )
            hull = None
            oc, orr = centres * to_out, radii * to_out
            if len(oc):
                margin = cfg.margin_frac * float(np.median(orr))
                if len(ring) >= 3:
                    hull = tangent_ring(oc, orr, ring, margin=margin)
                    stats.ring_frames += 1
                    stats.max_standoff = max(stats.max_standoff, hull.max_standoff())
                else:
                    hull = hull_of_disks(oc, orr, margin=margin)

            draw_overlay(
                canvas,
                oc,
                orr,
                observed,
                hull,
                style=cfg.style,
                px=px,
                raw=det.raw * to_out if cfg.debug and len(det.raw) else None,
                hud_lines=[f"frame {idx}  |  {status}"] if cfg.style.show_hud else None,
            )
            writer.write(canvas)
            stats.frames += 1

            if not cfg.quiet and stats.frames % 30 == 0:
                done = idx - cfg.start + 1
                span = max(1, last - cfg.start)
                rate = stats.frames / max(time.time() - t0, 1e-6)
                print(
                    f"\r  {done}/{span} frames  {rate:5.1f} fps  {status:<22}", end="", flush=True
                )
            idx += 1
    finally:
        cap.release()
        writer.release()
        if not cfg.quiet:
            print()

    stats.seconds = time.time() - t0
    return stats
