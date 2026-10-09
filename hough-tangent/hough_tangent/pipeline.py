"""Video in, annotated video out.

Two passes over the clip.  The first tracks the lattice and keeps the geometry
it measured; only then, knowing the whole clip, does it decide which cells hold
a disc.  The second pass decodes again and draws from the stored geometry.

The split matters: an earlier version learned occupancy with one tracker and
drew with a second, and the two placed the lattice differently often enough to
apply a cell's verdict to its neighbour -- a phantom circle over bare frame on
one side, a real disc dropped out of the outline on the other.  One tracker,
one labelling, no way for them to disagree.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np

from .detect import DetectParams, detect_circles, refine_in_roi
from .grid import GridModel, GridTracker, boundary_order, hole_orders
from .hull import DiskHull, hull_of_disks, tangent_ring
from .render import Style, draw_overlay
from .smooth import OneEuro
from .validate import disc_reference


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
    cells: np.ndarray | None = None  # explicit occupancy mask, skips the analysis verdict
    smooth: bool = True
    min_cutoff: float = 0.9
    beta: float = 0.015
    outline: str = "ring"  # "ring" follows the occupied boundary, "hull" is the convex hull
    margin_frac: float = 0.0  # pad the outline outwards by this fraction of a radius
    holes: bool = True  # outline empty cells the board surrounds, not just the outside
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
    holes: int = 0
    hole_frames: int = 0
    relocks: int = 0
    seconds: float = 0.0
    occupancy: np.ndarray | None = None


@dataclass
class FrameState:
    """What the analysis pass measured on one frame, in working coordinates."""

    index: int
    generation: int = 0
    nodes: np.ndarray | None = None
    radii: np.ndarray | None = None
    observed: np.ndarray | None = None
    coasting: int = 0
    raw: np.ndarray | None = None
    loose: np.ndarray | None = None  # detections when running without a lattice


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


def _frames(cap: cv2.VideoCapture, cfg: Config, last: int):
    """Yield (index, frame) for the configured range and stride."""
    if cfg.start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, cfg.start)
    idx = cfg.start
    while idx < last:
        ok, frame = cap.read()
        if not ok:
            return
        if (idx - cfg.start) % cfg.stride == 0:
            yield idx, frame
        idx += 1


def analyse(
    cfg: Config, work_size: tuple[int, int], params: DetectParams, last: int, progress=None
) -> list[FrameState]:
    """Track the lattice through the clip, keeping the geometry of every frame."""
    cap = _open(cfg.src)
    tracker = _tracker(cfg)
    states: list[FrameState] = []
    try:
        for idx, frame in _frames(cap, cfg, last):
            work = cv2.resize(frame, work_size, interpolation=cv2.INTER_AREA)
            det = detect_circles(work, params)
            if cfg.no_grid:
                states.append(FrameState(index=idx, loose=det.circles, raw=det.raw))
                if progress:
                    progress(len(states), f"hough only: {len(det.circles)}/{len(det.raw)}")
                continue
            if cfg.refine:
                ref = disc_reference(work, det.gray, det.circles)
                tracker.refiner = lambda c, r, w=work, g=det.gray, q=ref: refine_in_roi(
                    w, g, c, r, ref=q
                )
            model = tracker.update(det.circles)
            if model is None:
                states.append(FrameState(index=idx, raw=det.raw))
                if progress:
                    progress(len(states), "searching")
                continue
            states.append(
                FrameState(
                    index=idx,
                    generation=tracker.generation,
                    nodes=model.nodes.copy(),
                    radii=model.radii.copy(),
                    observed=model.observed.copy(),
                    coasting=tracker.coasting,
                    raw=det.raw,
                )
            )
            if cfg.adaptive:
                _adapt(params, model, model.radii)
            if progress:
                progress(len(states), f"lock {int(model.observed.sum())}/{model.count}")
    finally:
        cap.release()
    return states


def occupancy_from(states: list[FrameState], count: int, threshold: float) -> np.ndarray:
    """Decide which cells hold a disc, from the whole clip rather than a sample.

    Cell labels are only comparable inside one unbroken track, so score the
    longest one.  Measured over six clips an occupied cell is seen on >=97% of
    frames and an empty one on <=20%, which is a gap the threshold sits well
    inside.
    """
    locked = [s for s in states if s.observed is not None]
    if not locked:
        return np.ones(count, dtype=bool)
    runs: dict[int, list[FrameState]] = {}
    for s in locked:
        runs.setdefault(s.generation, []).append(s)
    best = max(runs.values(), key=len)
    rate = np.mean([s.observed for s in best], axis=0)
    occupied = rate >= threshold
    return occupied if occupied.any() else np.ones(count, dtype=bool)


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
    cap.release()

    work_w = min(cfg.work_width, src_w)
    work_size = (work_w, max(1, round(src_h * work_w / src_w)))
    out_w = min(cfg.out_width, src_w)
    out_size = (out_w, max(1, round(src_h * out_w / src_w)))
    to_out = out_w / work_w
    px = out_w / 1080.0
    last = cfg.end if cfg.end is not None else total
    span = max(1, last - cfg.start)

    params = DetectParams(r_min=cfg.r_min_frac * work_w, r_max=cfg.r_max_frac * work_w)
    params.min_dist = 1.6 * params.r_min

    t0 = time.time()

    def note(stage: str):
        if cfg.quiet:
            return None

        def cb(done: int, status: str):
            if done % 30 == 0:
                rate = done / max(time.time() - t0, 1e-6)
                print(
                    f"\r  {stage} {done * cfg.stride}/{span}  {rate:5.1f} fps  {status:<22}",
                    end="",
                    flush=True,
                )

        return cb

    states = analyse(cfg, work_size, replace(params), last, note("analysing"))
    if not cfg.quiet:
        print()

    count = cfg.rows * cfg.cols
    occupied: np.ndarray | None = None
    if not cfg.no_grid:
        occupied = cfg.cells if cfg.cells is not None else occupancy_from(states, count, 0.5)
        if not cfg.quiet:
            for row in occupied.reshape(cfg.rows, cfg.cols):
                print("    " + " ".join("#" if v else "." for v in row))

    outline_cells = (
        boundary_order(occupied, cfg.rows, cfg.cols)
        if cfg.outline == "ring" and not cfg.no_grid and occupied is not None
        else []
    )
    pos = (
        {int(c): i for i, c in enumerate(np.flatnonzero(occupied))} if occupied is not None else {}
    )
    ring = [pos[c] for c in outline_cells if c in pos]
    # Empty cells the board surrounds get an outline of their own, tangent to
    # the discs round them from the inside.
    holes = (
        [[pos[c] for c in walk if c in pos] for walk in hole_orders(occupied, cfg.rows, cfg.cols)]
        if cfg.holes and not cfg.no_grid and occupied is not None
        else []
    )
    holes = [h for h in holes if len(h) >= 3]

    pos_filter = OneEuro(fps / cfg.stride, cfg.min_cutoff, cfg.beta)
    rad_filter = OneEuro(fps / cfg.stride, cfg.min_cutoff, cfg.beta * 0.5)
    generation = -1

    cfg.dst.parent.mkdir(parents=True, exist_ok=True)
    writer = _make_writer(cfg.dst, cfg.fourcc, fps / cfg.stride, out_size)
    cap = _open(cfg.src)
    stats = Stats(occupancy=occupied, holes=len(holes))
    stats.relocks = len({s.generation for s in states if s.observed is not None})
    by_index = {s.index: s for s in states}
    try:
        for idx, frame in _frames(cap, cfg, last):
            state = by_index.get(idx)
            centres = np.zeros((0, 2))
            radii = np.zeros(0)
            observed: np.ndarray | None = None
            status = "searching"

            if state is not None and state.loose is not None:
                centres, radii = state.loose[:, :2], state.loose[:, 2]
                status = f"hough only: {len(state.loose)}"
            elif state is not None and state.nodes is not None:
                if state.generation != generation:
                    generation = state.generation
                    pos_filter.reset()
                    rad_filter.reset()
                nodes, node_r = state.nodes, state.radii
                if cfg.smooth:
                    nodes = pos_filter(nodes)
                    node_r = rad_filter(node_r)
                keep = occupied if occupied is not None else np.ones(len(nodes), bool)
                centres, radii, observed = nodes[keep], node_r[keep], state.observed[keep]
                n_obs, n_present = int(observed.sum()), int(keep.sum())
                stats.locked += 1
                stats.observed_total += n_obs
                stats.present_total += n_present
                status = (
                    f"coasting {state.coasting}" if state.coasting else f"lock {n_obs}/{n_present}"
                )
                if state.coasting:
                    stats.coasted += 1

            canvas = cv2.resize(
                frame, out_size, interpolation=cv2.INTER_AREA if out_w < src_w else cv2.INTER_CUBIC
            )
            hull = None
            inner: list[DiskHull] = []
            oc, orr = centres * to_out, radii * to_out
            if len(oc):
                margin = cfg.margin_frac * float(np.median(orr))
                if len(ring) >= 3 and len(oc) == len(pos):
                    hull = tangent_ring(oc, orr, ring, margin=margin)
                    stats.ring_frames += 1
                    stats.max_standoff = max(stats.max_standoff, hull.max_standoff())
                else:
                    hull = hull_of_disks(oc, orr, margin=margin)
                if holes and len(oc) == len(pos):
                    inner = [tangent_ring(oc, orr, h, margin=margin, fallback=False) for h in holes]
                    stats.hole_frames += 1

            draw_overlay(
                canvas,
                oc,
                orr,
                observed,
                hull,
                style=cfg.style,
                px=px,
                raw=state.raw * to_out
                if (cfg.debug and state is not None and state.raw is not None and len(state.raw))
                else None,
                hud_lines=[f"frame {idx}  |  {status}"] if cfg.style.show_hud else None,
                holes=inner,
            )
            writer.write(canvas)
            stats.frames += 1
            if not cfg.quiet and stats.frames % 30 == 0:
                rate = stats.frames / max(time.time() - t0, 1e-6)
                print(
                    f"\r  drawing  {stats.frames * cfg.stride}/{span}  {rate:5.1f} fps  "
                    f"{status:<22}",
                    end="",
                    flush=True,
                )
    finally:
        cap.release()
        writer.release()
        if not cfg.quiet:
            print()

    stats.seconds = time.time() - t0
    return stats
