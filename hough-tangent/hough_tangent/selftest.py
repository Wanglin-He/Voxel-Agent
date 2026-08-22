"""Self-checks that need no video: `python -m hough_tangent.selftest`.

Deliberately not pytest files -- this package is not part of the repository's
test suite and pulling OpenCV into it would break `uv run pytest`.
"""

from __future__ import annotations

import numpy as np

from .grid import GridTracker, initialise, ring_order
from .hull import hull_of_disks, tangent_ring

TWO_PI = 2.0 * np.pi
_failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        _failures.append(name)


def test_hull_is_tangent() -> None:
    """Every straight edge must touch its two disks and clear all the others."""
    rng = np.random.default_rng(7)
    worst_touch, worst_side = 0.0, -np.inf
    for _ in range(200):
        n = int(rng.integers(3, 18))
        c = rng.uniform(-500, 500, size=(n, 2))
        r = rng.uniform(10, 90, size=n)
        h = hull_of_disks(c, r)
        for t in h.tangents:
            nx, ny, off = t.line
            signed = c @ np.array([nx, ny]) - off  # <= -r for every disk
            worst_side = max(worst_side, float((signed + r).max()))
            for k in (t.disk_a, t.disk_b):
                worst_touch = max(worst_touch, abs(float(signed[k] + r[k])))
    check("tangent lines touch both disks", worst_touch < 1e-6, f"max error {worst_touch:.2e}")
    check("no disk crosses a tangent line", worst_side < 1e-6, f"max overshoot {worst_side:.2e}")


def test_hull_contains_every_disk() -> None:
    """The sampled outline must enclose all the disks it was built from."""
    rng = np.random.default_rng(11)
    worst = -np.inf
    for _ in range(50):
        n = int(rng.integers(4, 14))
        c = rng.uniform(-300, 300, size=(n, 2))
        r = rng.uniform(20, 70, size=n)
        poly = hull_of_disks(c, r).polyline(step_px=0.5)
        # Support function of the polygon must dominate every disk's.
        th = np.linspace(0, TWO_PI, 720, endpoint=False)
        nrm = np.stack([np.cos(th), np.sin(th)], axis=1)
        worst = max(worst, float(((c @ nrm.T + r[:, None]) - (poly @ nrm.T).max(axis=0)).max()))
    check("outline encloses every disk", worst < 0.5, f"max protrusion {worst:.3f} px")


def test_sharp_corners() -> None:
    """Each corner must sit on both of its tangent lines, still outside every disk."""
    rng = np.random.default_rng(23)
    worst_on_line, worst_inside, rounded = 0.0, -np.inf, 0
    for _ in range(200):
        n = int(rng.integers(3, 18))
        c = rng.uniform(-500, 500, size=(n, 2))
        r = rng.uniform(10, 90, size=n)
        h = hull_of_disks(c, r)
        m = len(h.arcs)
        for k in range(m):
            v = h.corner(k)
            if v is None:
                rounded += 1
                continue
            for t in (h.tangents[(k - 1) % m], h.tangents[k]):
                nx, ny, off = t.line
                worst_on_line = max(worst_on_line, abs(float(nx * v[0] + ny * v[1] - off)))
            worst_inside = max(worst_inside, float((r - np.linalg.norm(c - v, axis=1)).max()))
    check(
        "corners lie on both tangent lines", worst_on_line < 1e-6, f"max error {worst_on_line:.2e}"
    )
    check(
        "corners stay outside every disk", worst_inside < 1e-6, f"max intrusion {worst_inside:.2e}"
    )

    # A capsule's two arcs each span half a turn, so its tangents are parallel
    # and there is no corner to be had -- it has to stay rounded.
    capsule = hull_of_disks(np.array([[0.0, 0.0], [200.0, 0.0]]), np.array([40.0, 40.0]))
    check("a capsule keeps its round ends", all(capsule.corner(k) is None for k in (0, 1)))

    square = hull_of_disks(
        np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0]]), np.full(4, 20.0)
    )
    corner = square.corner(0)
    centre = np.asarray(square.arcs[0].center)
    got = float(np.hypot(*(corner - centre))) if corner is not None else float("nan")
    want = 20.0 * np.sqrt(2.0)  # r / cos(45 deg), measured from the disc centre
    check(
        "a right-angle corner reaches r*sqrt(2)", abs(got - want) < 1e-9, f"{got:.6f} vs {want:.6f}"
    )
    poly = square.polyline(1.0, sharp=True)
    check("sharp square is a 4-gon", len(poly) == 4, f"{len(poly)} vertices")


def _dist_to_polyline(poly: np.ndarray, pt: np.ndarray) -> float:
    a = poly
    b = np.roll(poly, -1, axis=0)
    e = b - a
    t = np.clip(((pt - a) * e).sum(1) / np.maximum((e**2).sum(1), 1e-12), 0.0, 1.0)
    return float(np.hypot(*(pt - (a + t[:, None] * e)).T).min())


def _ring_gaps(c: np.ndarray, r: np.ndarray, order: list, sharp: bool) -> np.ndarray:
    """Signed gap per disk: 0 = touched, >0 = the outline never reaches it,
    <0 = the outline cuts into it."""
    poly = tangent_ring(c, r, order).polyline(0.25, sharp=sharp)
    return np.array([_dist_to_polyline(poly, c[k]) - r[k] for k in order])


def test_tangent_ring_touches_everything() -> None:
    """Ring mode must reach every disk on the ring, to a bound we can state.

    Where the ring turns outward the outline touches exactly.  Where a disk sits
    recessed the turn goes inward, and the crossing of its two tangents stands
    off the disk by exactly ``r * (sec(turn / 2) - 1)`` -- you cannot have both a
    simple polygon and exact tangency there.  Assert that bound rather than some
    picked number, so the check keeps meaning if the geometry changes.
    """
    rng = np.random.default_rng(31)
    worst_excess, worst_cut, worst_convex = -np.inf, np.inf, 0.0
    worst_bound = 0.0
    for _ in range(120):
        rows, cols = 4, 4
        order = ring_order(rows, cols)
        c, r = [], []
        for k in range(rows * cols):
            v, u = divmod(k, cols)
            c.append([100.0 * u + rng.normal(0, 9), 100.0 * v + rng.normal(0, 9)])
            r.append(rng.uniform(30, 38))
        c, r = np.array(c), np.array(r)
        ring = tangent_ring(c, r, order)
        poly = ring.polyline(0.25, sharp=True)
        for i, k in enumerate(order):
            gap = _dist_to_polyline(poly, c[k]) - r[k]
            sweep = ring.arcs[i].sweep
            if sweep >= 0.0:
                worst_convex = max(worst_convex, abs(gap))
            else:
                bound = r[k] * (1.0 / np.cos(abs(sweep) / 2.0) - 1.0)
                worst_excess = max(worst_excess, gap - bound)
                worst_bound = max(worst_bound, bound)
            worst_cut = min(worst_cut, gap)
    check("outward turns touch exactly", worst_convex < 1e-6, f"max error {worst_convex:.2e} px")
    check(
        "recessed disks stand off by no more than r*(sec(turn/2)-1)",
        worst_excess < 1e-6,
        f"worst overshoot {worst_excess:.2e} px, bound reached {worst_bound:.2f} px",
    )
    check("ring never cuts into a disk", worst_cut > -1e-6, f"deepest cut {worst_cut:.2e} px")


def test_ring_beats_hull_on_a_recessed_disk() -> None:
    """Three disks with the middle one pulled inward: the hull edge skips it."""
    c = np.array([[0.0, 0.0], [100.0, 18.0], [200.0, 0.0], [200.0, 200.0], [0.0, 200.0]])
    r = np.full(5, 40.0)
    order = [0, 1, 2, 3, 4]

    hull_poly = hull_of_disks(c, r).polyline(0.25, sharp=True)
    hull_gap = _dist_to_polyline(hull_poly, c[1]) - r[1]
    ring_gap = _dist_to_polyline(tangent_ring(c, r, order).polyline(0.25, sharp=True), c[1]) - r[1]
    check("hull misses the recessed disk", hull_gap > 5.0, f"gap {hull_gap:.2f} px")
    check("ring reaches the recessed disk", abs(ring_gap) < 1.0, f"gap {ring_gap:.3f} px")


def test_ring_turn_directions() -> None:
    """Turns must sum to a full revolution, and dip negative only where recessed."""
    c = np.array([[0.0, 0.0], [100.0, 18.0], [200.0, 0.0], [200.0, 200.0], [0.0, 200.0]])
    r = np.full(5, 40.0)
    ring = tangent_ring(c, r, [0, 1, 2, 3, 4])
    total = sum(a.sweep for a in ring.arcs)
    check(
        "ring turns sum to one revolution",
        abs(total - 2 * np.pi) < 1e-9,
        f"{np.degrees(total):.4f} deg",
    )
    check(
        "only the recessed disk turns the other way", sum(1 for a in ring.arcs if a.sweep < 0) == 1
    )


def test_hull_shapes() -> None:
    square = hull_of_disks(
        np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0]]), np.full(4, 20.0)
    )
    check(
        "square of 4 disks -> 4 arcs + 4 tangents",
        len(square.arcs) == 4 and len(square.tangents) == 4,
    )
    nested = hull_of_disks(np.array([[0.0, 0.0], [3.0, 0.0]]), np.array([50.0, 5.0]))
    check("a swallowed disk leaves the hull", nested.disks == [0] and not nested.tangents)
    capsule = hull_of_disks(np.array([[0.0, 0.0], [200.0, 0.0]]), np.array([20.0, 60.0]))
    expected = np.degrees(np.arcsin((60.0 - 20.0) / 200.0))
    got = abs(90.0 - abs(np.degrees(capsule.tangents[0].theta) - 180.0))
    check(
        "unequal radii tilt the tangent",
        abs(got - expected) < 1e-6,
        f"{got:.4f} deg vs {expected:.4f} deg",
    )


def _synthetic_grid(
    rows: int,
    cols: int,
    *,
    drop: set[int] = frozenset(),
    shear: float = 0.0,
    jitter: float = 0.0,
    seed: int = 3,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = []
    for k in range(rows * cols):
        v, u = divmod(k, cols)
        if k in drop:
            continue
        x = 100.0 + 90.0 * u + shear * v
        y = 120.0 + 88.0 * v
        out.append([x + rng.normal(0, jitter), y + rng.normal(0, jitter), 30.0])
    return np.array(out)


def test_grid_initialise() -> None:
    pts = _synthetic_grid(4, 4)
    fit = initialise(pts, 4, 4)
    check("clean 4x4 fully explained", fit is not None and len(fit[1]) == 16)

    sheared = _synthetic_grid(4, 4, shear=34.0, jitter=3.0)
    fit = initialise(sheared, 4, 4)
    check("sheared, jittered 4x4 fully explained", fit is not None and len(fit[1]) == 16)

    holed = _synthetic_grid(4, 4, drop={5, 10}, shear=20.0)
    fit = initialise(holed, 4, 4)
    ok = fit is not None and len(fit[1]) == 14
    cells = set() if fit is None else {int(v) * 4 + int(u) for u, v in fit[0]}
    check(
        "two empty cells still lands on the right lattice",
        ok and 5 not in cells and 10 not in cells,
    )

    # A distractor cluster off to the side must not drag the window over.
    noisy = np.concatenate(
        [_synthetic_grid(4, 4), np.array([[600.0, 500.0, 30.0], [610.0, 590.0, 30.0]])]
    )
    fit = initialise(noisy, 4, 4)
    check("outliers do not shift the window", fit is not None and len(fit[1]) == 16)


def test_tracker_learns_vacancies() -> None:
    tracker = GridTracker(4, 4, occupancy_warmup=5)
    for step in range(40):
        pts = _synthetic_grid(4, 4, drop={5, 10}, shear=0.4 * step, jitter=1.0, seed=step)
        model = tracker.update(pts)
    check("tracker keeps the lock while the lattice shears", model is not None)
    vacant = set(np.flatnonzero(tracker.vacant()).tolist())
    check("tracker finds the empty cells", vacant == {5, 10}, f"found {sorted(vacant)}")
    check("only real discs are reported", model is not None and int(model.present.sum()) == 14)


def main() -> int:
    for fn in (
        test_hull_is_tangent,
        test_hull_contains_every_disk,
        test_sharp_corners,
        test_tangent_ring_touches_everything,
        test_ring_beats_hull_on_a_recessed_disk,
        test_ring_turn_directions,
        test_hull_shapes,
        test_grid_initialise,
        test_tracker_learns_vacancies,
    ):
        print(f"{fn.__name__}:")
        fn()
    print()
    if _failures:
        print(f"{len(_failures)} check(s) failed: {', '.join(_failures)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
