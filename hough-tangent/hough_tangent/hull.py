"""Exact convex hull of a set of disks.

The boundary of the convex hull of a union of disks alternates between straight
outer common tangent segments and circular arcs.  We recover it from the
support function: for an outward unit normal ``n(t) = (cos t, sin t)`` the
supporting line of disk ``i`` sits at ``h_i(t) = c_i . n(t) + r_i``.  The hull's
own support function is the upper envelope ``max_i h_i(t)``, and the disk that
attains the max at ``t`` owns the boundary point ``c_i + r_i n(t)``.

So the disks that show up on the hull are exactly the disks that win the
envelope somewhere, in angular order, and every breakpoint of the envelope is
an outer common tangent touching two disks at once.  Sampling ``t`` finds the
ordering; each breakpoint is then solved in closed form, so the tangents are
exact rather than a polygon approximation of them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class Arc:
    """A stretch of hull boundary owned by one disk, walked counter-clockwise."""

    disk: int
    center: tuple[float, float]
    radius: float
    theta0: float
    theta1: float  # always > theta0

    @property
    def sweep(self) -> float:
        return self.theta1 - self.theta0

    def sample(self, step_px: float = 1.0) -> np.ndarray:
        span = abs(self.sweep)
        n = max(2, int(np.ceil(self.radius * span / max(step_px, 1e-3))) + 1)
        t = np.linspace(self.theta0, self.theta1, n)
        cx, cy = self.center
        return np.stack([cx + self.radius * np.cos(t), cy + self.radius * np.sin(t)], axis=1)


@dataclass(frozen=True)
class Tangent:
    """An outer common tangent segment bridging two disks on the hull."""

    disk_a: int
    disk_b: int
    theta: float  # outward normal angle of the tangent line
    p_a: tuple[float, float]
    p_b: tuple[float, float]

    @property
    def line(self) -> tuple[float, float, float]:
        """The tangent as ``(nx, ny, c)`` with ``nx*x + ny*y = c``."""
        nx, ny = np.cos(self.theta), np.sin(self.theta)
        return float(nx), float(ny), float(nx * self.p_a[0] + ny * self.p_a[1])


def _intersect(l1: tuple[float, float, float], l2: tuple[float, float, float]) -> np.ndarray | None:
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-12:
        return None
    return np.array([(c1 * b2 - c2 * b1) / det, (a1 * c2 - a2 * c1) / det])


@dataclass
class DiskHull:
    arcs: list[Arc]
    tangents: list[Tangent]

    @property
    def disks(self) -> list[int]:
        return [a.disk for a in self.arcs]

    def corner(self, k: int, max_reach: float = 6.0) -> np.ndarray | None:
        """Where the two tangents touching disk ``k`` meet, if that is a real point.

        The corner sits at ``r / cos(span / 2)`` from the centre, so it runs away
        to infinity as a disk's arc approaches half a turn.  ``max_reach`` caps
        that in radii; past the cap there is no usable corner and the caller
        should keep the arc.
        """
        arc = self.arcs[k]
        m = len(self.arcs)
        if m < 2:
            return None
        v = _intersect(self.tangents[(k - 1) % m].line, self.tangents[k].line)
        if v is None:
            return None
        if float(np.hypot(*(v - np.asarray(arc.center)))) > max_reach * arc.radius:
            return None
        return v

    def split_long_arcs(self, target_sweep: float, max_reach: float = 6.0) -> DiskHull:
        """Cut long arcs with extra tangent lines so their ends corner.

        A one-disk-wide arm ends in a half turn, and two tangent lines half a
        turn apart are parallel -- there is no crossing to build a corner from,
        so the tip can only come out round.  Cutting that arc with a tangent
        line at the pole leaves two quarter turns, each with a real crossing,
        and the tip squares off like the rest of the outline.  Every line
        involved still touches the disk, so tangency is untouched.

        ``target_sweep`` is what each piece aims for, not a ceiling: an arm
        tip measures a shade over half a turn on real footage, and rounding
        that up gives three shallow corners instead of two right angles.
        """
        if target_sweep <= 0 or not self.arcs or len(self.tangents) != len(self.arcs):
            return self
        # Past this the two tangents sit too near parallel for their crossing
        # to land anywhere useful, so a piece that wide has to be cut smaller.
        hard_limit = 2.0 * float(np.arccos(1.0 / max(max_reach, 1.0 + 1e-9)))
        arcs: list[Arc] = []
        tangents: list[Tangent] = []
        for k, arc in enumerate(self.arcs):
            n = max(1, int(np.floor(abs(arc.sweep) / target_sweep + 0.5)))
            while abs(arc.sweep) / n > hard_limit:
                n += 1
            step = arc.sweep / n
            for j in range(n):
                t0 = arc.theta0 + j * step
                arcs.append(Arc(arc.disk, arc.center, arc.radius, t0, t0 + step))
                if j < n - 1:
                    t1 = t0 + step
                    nv = np.array([np.cos(t1), np.sin(t1)])
                    touch = tuple(np.asarray(arc.center) + arc.radius * nv)
                    tangents.append(Tangent(arc.disk, arc.disk, _wrap(t1), touch, touch))
            tangents.append(self.tangents[k])
        return DiskHull(arcs, tangents)

    def max_standoff(self) -> float:
        """Largest distance by which the outline misses a disk it walks over.

        Zero everywhere the outline turns outward.  At a recessed disk the two
        tangents cross ``r / cos(turn / 2)`` from its centre, so the outline
        clears it by ``r * (sec(turn / 2) - 1)`` -- selftest pins this to 1e-11.
        """
        worst = 0.0
        for arc in self.arcs:
            if arc.sweep < 0.0:
                worst = max(worst, arc.radius * (1.0 / np.cos(abs(arc.sweep) / 2.0) - 1.0))
        return worst

    def polyline(
        self, step_px: float = 1.0, *, sharp: bool = False, max_reach: float = 6.0
    ) -> np.ndarray:
        """The closed boundary as an (N, 2) float array.

        With ``sharp`` the arcs are replaced by the point where their two
        tangents cross, giving a polygon whose every edge is an exact outer
        common tangent.  A disk whose corner would fly off past ``max_reach``
        keeps its arc.
        """
        parts: list[np.ndarray] = []
        for k, arc in enumerate(self.arcs):
            want_corner = sharp or arc.sweep < 0.0
            corner = self.corner(k, max_reach) if want_corner else None
            if corner is not None:
                parts.append(corner.reshape(1, 2))
            else:
                parts.append(arc.sample(step_px))
            if not sharp and k < len(self.tangents):
                tan = self.tangents[k]
                parts.append(np.array([tan.p_a, tan.p_b], dtype=float))
        if not parts:
            return np.zeros((0, 2))
        pts = np.concatenate(parts, axis=0)
        # Drop consecutive duplicates left by arc/segment endpoints coinciding.
        keep = np.ones(len(pts), dtype=bool)
        keep[1:] = np.any(np.abs(np.diff(pts, axis=0)) > 1e-6, axis=1)
        return pts[keep]


def _wrap(t: float) -> float:
    return float(np.mod(t, TWO_PI))


def _circ_dist(a: float, b: float) -> float:
    d = abs(_wrap(a) - _wrap(b))
    return min(d, TWO_PI - d)


def _breakpoint(ci, ri, cj, rj, near: float) -> float:
    """Normal angle where disks i and j share a supporting line, nearest ``near``."""
    d = ci - cj
    dist = float(np.hypot(d[0], d[1]))
    if dist < 1e-9:
        return near
    k = np.clip((rj - ri) / dist, -1.0, 1.0)
    psi = float(np.arctan2(d[1], d[0]))
    a = float(np.arccos(k))
    roots = (psi + a, psi - a)
    return _wrap(min(roots, key=lambda t: _circ_dist(t, near)))


def hull_of_disks(
    centers: np.ndarray,
    radii: np.ndarray,
    *,
    margin: float = 0.0,
    samples: int = 4096,
) -> DiskHull:
    """Convex hull of the union of the given disks, as exact tangents and arcs.

    ``margin`` inflates every radius first, which pads the outline outwards by a
    constant distance without changing its shape.
    """
    c = np.asarray(centers, dtype=float).reshape(-1, 2)
    r = np.asarray(radii, dtype=float).reshape(-1) + float(margin)
    if len(c) == 0:
        return DiskHull([], [])
    if len(c) == 1:
        return DiskHull([Arc(0, tuple(c[0]), float(r[0]), 0.0, TWO_PI)], [])

    t = np.arange(samples) * (TWO_PI / samples)
    n = np.stack([np.cos(t), np.sin(t)], axis=1)
    support = c @ n.T + r[:, None]
    active = np.argmax(support, axis=0)

    starts = np.flatnonzero(active != np.roll(active, 1))
    if starts.size == 0:  # one disk swallows every other
        i = int(active[0])
        return DiskHull([Arc(i, tuple(c[i]), float(r[i]), 0.0, TWO_PI)], [])

    order = [int(active[s]) for s in starts]
    if len(order) == 1:
        i = order[0]
        return DiskHull([Arc(i, tuple(c[i]), float(r[i]), 0.0, TWO_PI)], [])

    # Solve each envelope breakpoint exactly, bracketed by the sampled crossing.
    breaks: list[float] = []
    for k, s in enumerate(starts):
        prev_disk = order[k - 1]
        disk = order[k]
        breaks.append(_breakpoint(c[prev_disk], r[prev_disk], c[disk], r[disk], float(t[s])))

    arcs: list[Arc] = []
    tangents: list[Tangent] = []
    m = len(order)
    for k in range(m):
        disk = order[k]
        nxt = order[(k + 1) % m]
        th0 = breaks[k]
        th1 = breaks[(k + 1) % m]
        if th1 <= th0:
            th1 += TWO_PI
        arcs.append(Arc(disk, tuple(c[disk]), float(r[disk]), th0, th1))
        nv = np.array([np.cos(th1), np.sin(th1)])
        tangents.append(
            Tangent(
                disk_a=disk,
                disk_b=nxt,
                theta=_wrap(th1),
                p_a=tuple(c[disk] + r[disk] * nv),
                p_b=tuple(c[nxt] + r[nxt] * nv),
            )
        )
    return DiskHull(arcs, tangents)


def _wrap_signed(t: float) -> float:
    """Shortest signed turn, in (-pi, pi]."""
    return float(np.mod(t + np.pi, TWO_PI) - np.pi)


def _outward_tangent(ca: np.ndarray, ra: float, cb: np.ndarray, rb: float) -> float | None:
    """Normal angle of the common tangent on the outward side of the walk.

    Which side is "outward" is decided by the direction of travel, not by which
    way the centre of the shape lies.  Inside a notch those disagree: the
    material is behind you there, so a centre-based test picks the tangent that
    cuts straight across the gap.
    """
    d = ca - cb
    dist = float(np.hypot(d[0], d[1]))
    if dist < 1e-9:
        return None
    k = (rb - ra) / dist
    if abs(k) > 1.0:  # one disk swallows the other, no outer tangent exists
        return None
    psi = float(np.arctan2(d[1], d[0]))
    a = float(np.arccos(np.clip(k, -1.0, 1.0)))
    travel = cb - ca
    best_t, best_cross = None, np.inf
    for t in (psi + a, psi - a):
        nx, ny = np.cos(t), np.sin(t)
        cross = float(travel[0] * ny - travel[1] * nx)  # negative on the outward side
        if cross < best_cross:
            best_t, best_cross = t, cross
    return _wrap(best_t)


def tangent_ring(
    centers: np.ndarray, radii: np.ndarray, order: list[int], *, margin: float = 0.0
) -> DiskHull:
    """Outline that runs tangent to every disk in ``order``, in that cyclic order.

    The convex hull only touches the disks that reach it -- a disk sitting a
    little inside a hull edge is enclosed but never touched.  When the caller
    knows which disks form the outer ring (here the lattice tells us), walking
    that ring instead guarantees each one is touched.  The price is that the
    outline is no longer convex: at a recessed disk the turn goes the other way,
    and the boundary there is the crossing of its two tangents.
    """
    c = np.asarray(centers, dtype=float).reshape(-1, 2)
    r = np.asarray(radii, dtype=float).reshape(-1) + float(margin)
    m = len(order)
    if m == 0:
        return DiskHull([], [])
    if m == 1:
        i = order[0]
        return DiskHull([Arc(i, tuple(c[i]), float(r[i]), 0.0, TWO_PI)], [])

    thetas: list[float] = []
    for k in range(m):
        a, b = order[k], order[(k + 1) % m]
        t = _outward_tangent(c[a], r[a], c[b], r[b])
        if t is None:  # degenerate ring, the convex hull is still well defined
            return hull_of_disks(c, r)
        thetas.append(t)

    arcs: list[Arc] = []
    tangents: list[Tangent] = []
    for k in range(m):
        disk = order[k]
        nxt = order[(k + 1) % m]
        prv = order[k - 1]
        th_in, th_out = thetas[k - 1], thetas[k]
        # How far the normal swings across a disk is close to how far the walk
        # itself turns there, so let the walk pick the branch.  Taking the
        # shortest turn instead gets the tip of a one-disk-wide arm exactly
        # backwards: the turn there is half a revolution, and the short way
        # round cuts across the disk instead of wrapping it.
        d_in = c[disk] - c[prv]
        d_out = c[nxt] - c[disk]
        phi = _wrap_signed(
            float(np.arctan2(d_out[1], d_out[0])) - float(np.arctan2(d_in[1], d_in[0]))
        )
        if abs(abs(phi) - np.pi) < 1e-6:
            phi = np.pi  # doubling back round a tip always wraps outwards
        sweep = phi + _wrap_signed(th_out - th_in - phi)
        arcs.append(Arc(disk, tuple(c[disk]), float(r[disk]), th_in, th_in + sweep))
        nv = np.array([np.cos(th_out), np.sin(th_out)])
        tangents.append(
            Tangent(
                disk_a=disk,
                disk_b=nxt,
                theta=th_out,
                p_a=tuple(c[disk] + r[disk] * nv),
                p_b=tuple(c[nxt] + r[nxt] * nv),
            )
        )
    return DiskHull(arcs, tangents)
