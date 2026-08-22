"""Fit and track a rows x cols lattice over raw Hough candidates.

The lattice is a *soft* prior.  It decides which candidate belongs to which
cell, throws away everything that does not fit a cell (glass reflections,
wire highlights, half circles at the frame edge), and invents a position for
any cell whose disc went undetected this frame.  It does not move a disc that
was actually detected -- the array physically deforms, so snapping observations
onto a rigid model would be a lie.  Use ``snap`` if you want that anyway.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import cv2
import numpy as np


@dataclass
class GridModel:
    rows: int
    cols: int
    nodes: np.ndarray  # (rows*cols, 2), row-major over (v, u)
    radii: np.ndarray  # (rows*cols,)
    observed: np.ndarray  # (rows*cols,) bool
    spacing: float
    transform: np.ndarray | None = None  # 3x3, lattice (u, v) -> image
    raw: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    vacant: np.ndarray | None = None  # cells the lattice has, the board does not

    @property
    def present(self) -> np.ndarray:
        """Mask of cells that actually hold a disc."""
        if self.vacant is None:
            return np.ones(self.rows * self.cols, dtype=bool)
        return ~self.vacant

    @property
    def count(self) -> int:
        return self.rows * self.cols

    def cell_of(self, k: int) -> tuple[int, int]:
        return divmod(k, self.cols)  # (v, u)


def ring_order(rows: int, cols: int) -> list[int]:
    """Perimeter cell ids in cyclic order: top row, right column, bottom, left."""
    if rows < 1 or cols < 1:
        return []
    if rows == 1:
        return list(range(cols))
    if cols == 1:
        return [v * cols for v in range(rows)]
    top = [u for u in range(cols)]
    right = [v * cols + (cols - 1) for v in range(1, rows)]
    bottom = [(rows - 1) * cols + u for u in range(cols - 2, -1, -1)]
    left = [v * cols for v in range(rows - 2, 0, -1)]
    return top + right + bottom + left


def _median_nn(pts: np.ndarray) -> float:
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    return float(np.median(np.min(d, axis=1)))


def _dominant_vectors(
    pts: np.ndarray, d0: float, tol_deg: float = 15.0, max_out: int = 5
) -> list[np.ndarray]:
    """Candidate lattice basis vectors, from clustering short pairwise offsets."""
    iu, iv = np.triu_indices(len(pts), 1)
    d = (pts[iu] - pts[iv]).astype(float)
    nrm = np.linalg.norm(d, axis=1)
    d = d[(nrm > 0.60 * d0) & (nrm < 1.30 * d0)]
    if len(d) == 0:
        return []
    flip = (d[:, 1] < 0) | ((np.abs(d[:, 1]) < 1e-9) & (d[:, 0] < 0))
    d[flip] *= -1.0
    ang = np.arctan2(d[:, 1], d[:, 0])  # [0, pi)
    order = np.argsort(ang)
    ang, d = ang[order], d[order]

    tol = np.deg2rad(tol_deg)
    groups: list[list[int]] = [[0]]
    for i in range(1, len(ang)):
        if ang[i] - ang[groups[-1][0]] <= tol:
            groups[-1].append(i)
        else:
            groups.append([i])
    if len(groups) > 1 and (np.pi - ang[groups[-1][0]]) + ang[groups[0][0]] <= tol:
        groups[0].extend(groups.pop())

    reps: list[np.ndarray] = []
    for g in sorted(groups, key=len, reverse=True)[:max_out]:
        # Members were canonicalised into [0, pi), so a cluster straddling the
        # wrap holds vectors pointing opposite ways; align them before taking a
        # median or they cancel out into a degenerate near-zero vector.
        member = d[g]
        member = member * np.where(member @ member[0] >= 0.0, 1.0, -1.0)[:, None]
        rep = np.median(member, axis=0)
        if np.linalg.norm(rep) > 0.5 * d0:
            reps.append(rep)
    return reps


def _fit_transform(uv: np.ndarray, xy: np.ndarray) -> np.ndarray | None:
    """Least-squares lattice -> image map: homography if we can, affine if not."""
    n = len(uv)
    if n < 3:
        return None
    if n >= 6:
        H, _ = cv2.findHomography(
            uv.astype(np.float32).reshape(-1, 1, 2), xy.astype(np.float32).reshape(-1, 1, 2), 0
        )
        if H is not None and np.all(np.isfinite(H)):
            return H
    A = np.concatenate([uv, np.ones((n, 1))], axis=1)
    sol, *_ = np.linalg.lstsq(A, xy, rcond=None)
    H = np.eye(3)
    H[:2, :2] = sol[:2].T
    H[:2, 2] = sol[2]
    return H


def _apply(H: np.ndarray, uv: np.ndarray) -> np.ndarray:
    pts = np.concatenate([uv, np.ones((len(uv), 1))], axis=1) @ H.T
    w = np.where(np.abs(pts[:, 2:3]) < 1e-12, 1e-12, pts[:, 2:3])
    return pts[:, :2] / w


def _lattice_uv(rows: int, cols: int) -> np.ndarray:
    v, u = np.divmod(np.arange(rows * cols), cols)
    return np.stack([u, v], axis=1).astype(float)


def _assign(pred: np.ndarray, cands: np.ndarray, tol: float) -> np.ndarray:
    """Greedy one-to-one nearest match; returns candidate index per node, -1 if none."""
    match = np.full(len(pred), -1, dtype=int)
    if len(cands) == 0:
        return match
    d = np.linalg.norm(pred[:, None, :] - cands[None, :, :2], axis=2)
    flat = np.dstack(np.unravel_index(np.argsort(d, axis=None), d.shape))[0]
    used_c: set[int] = set()
    for ni, ci in flat:
        if d[ni, ci] > tol:
            break
        if match[ni] >= 0 or ci in used_c:
            continue
        match[ni] = ci
        used_c.add(int(ci))
    return match


def _radius_model(uv: np.ndarray, r: np.ndarray, all_uv: np.ndarray) -> np.ndarray:
    """Radius varies smoothly with perspective, so fit r ~ a + b*u + c*v."""
    if len(r) >= 4:
        A = np.concatenate([uv, np.ones((len(uv), 1))], axis=1)
        sol, *_ = np.linalg.lstsq(A, r, rcond=None)
        pred = np.concatenate([all_uv, np.ones((len(all_uv), 1))], axis=1) @ sol
        med = float(np.median(r))
        return np.clip(pred, 0.6 * med, 1.5 * med)
    med = float(np.median(r)) if len(r) else 1.0
    return np.full(len(all_uv), med)


def initialise(
    cands: np.ndarray, rows: int, cols: int, resid_frac: float = 0.40, outside_penalty: float = 0.75
) -> tuple[np.ndarray, np.ndarray] | None:
    """Search for a rows x cols lattice among the candidates.

    Returns ``(uv, cand_index)`` for the cells we could explain, or None.
    """
    pts = cands[:, :2].astype(float)
    if len(pts) < 4:
        return None
    d0 = _median_nn(pts)
    if not np.isfinite(d0) or d0 <= 1e-6:
        return None
    vecs = _dominant_vectors(pts, d0)
    if len(vecs) < 2:
        return None

    centre = np.median(pts, axis=0)
    anchors = np.argsort(np.linalg.norm(pts - centre, axis=1))[:8]

    best: tuple[float, float, np.ndarray, np.ndarray] | None = None
    for a, b in combinations(vecs, 2):
        cross = abs(a[0] * b[1] - a[1] * b[0])
        if cross < 0.25 * np.linalg.norm(a) * np.linalg.norm(b):
            continue
        M = np.stack([a, b], axis=1)
        try:
            Minv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            continue
        for anchor_i in anchors:
            uv = (pts - pts[anchor_i]) @ Minv.T
            iuv = np.rint(uv)
            resid = np.linalg.norm(uv - iuv, axis=1) * min(np.linalg.norm(a), np.linalg.norm(b))
            ok = resid < resid_frac * d0
            if ok.sum() < 4:
                continue
            picked = _best_window(iuv[ok], np.flatnonzero(ok), resid[ok], rows, cols, M)
            if picked is None:
                continue
            uv_out, idx_out = picked
            # Validation already threw out everything that is not a disc, so a
            # correct window should account for nearly every surviving candidate.
            outside = len(pts) - len(idx_out)
            score = len(idx_out) - outside_penalty * outside
            area = abs(cross)
            if (
                best is None
                or score > best[0] + 1e-9
                or (abs(score - best[0]) <= 1e-9 and area < best[1])
            ):
                best = (score, area, uv_out, idx_out)
    if best is None or len(best[3]) < 4:
        return None
    return best[2], best[3]


def _best_window(
    iuv: np.ndarray, idx: np.ndarray, resid: np.ndarray, rows: int, cols: int, M: np.ndarray
):
    """Pick the rows x cols window of the integer lattice holding the most points.

    Also canonicalises the axes so u runs rightwards and v downwards, which keeps
    cell numbering reproducible whenever we have to re-initialise.
    """
    a, b = M[:, 0], M[:, 1]
    u, v = iuv[:, 0].copy(), iuv[:, 1].copy()
    if abs(a[0]) < abs(b[0]):
        a, b = b, a
        u, v = v, u
    if a[0] < 0:
        u = -u
    if b[1] < 0:
        v = -v

    # A hypothesis built on clutter can span a huge integer range; keep the
    # search near the bulk of the points instead of scanning all of it.
    span = 3 * max(rows, cols)
    u_lo = max(int(u.min()), int(np.median(u)) - span)
    u_hi = min(int(u.max()) - cols + 2, int(np.median(u)) + span)
    v_lo = max(int(v.min()), int(np.median(v)) - span)
    v_hi = min(int(v.max()) - rows + 2, int(np.median(v)) + span)

    best = None
    for du in range(u_lo, max(u_hi, u_lo + 1)):
        for dv in range(v_lo, max(v_hi, v_lo + 1)):
            uu, vv = u - du, v - dv
            inside = (uu >= 0) & (uu < cols) & (vv >= 0) & (vv < rows)
            if inside.sum() < 4:
                continue
            cells = (vv[inside] * cols + uu[inside]).astype(int)
            # Keep the lowest-residual point per cell.
            order = np.argsort(resid[inside])
            seen: dict[int, int] = {}
            for o in order:
                c = int(cells[o])
                if c not in seen:
                    seen[c] = int(np.flatnonzero(inside)[o])
            score = len(seen)
            if best is None or score > best[0]:
                sel = np.array(sorted(seen.items()), dtype=int)  # (n, 2): cell, local idx
                uv_out = np.stack([sel[:, 0] % cols, sel[:, 0] // cols], axis=1).astype(float)
                best = (score, uv_out, idx[sel[:, 1]])
    if best is None:
        return None
    return best[1], best[2]


class GridTracker:
    """Keeps one lattice locked onto the board across frames."""

    def __init__(
        self,
        rows: int = 4,
        cols: int = 4,
        *,
        snap: float = 0.0,
        gate_frac: float = 0.45,
        min_observed: int = 6,
        max_coast: int = 20,
        occupancy_alpha: float = 0.05,
        vacancy_threshold: float = 0.15,
        occupancy_warmup: int = 40,
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.snap = float(np.clip(snap, 0.0, 1.0))
        self.gate_frac = gate_frac
        self.min_observed = min_observed
        self.max_coast = max_coast
        self.occupancy_alpha = occupancy_alpha
        self.vacancy_threshold = vacancy_threshold
        self.occupancy_warmup = occupancy_warmup
        self.model: GridModel | None = None
        self.coasting = 0
        self.generation = 0  # bumps on every fresh lock, so filters can reset
        # A 4x4 lattice does not promise 16 discs: this board leaves two cells
        # empty.  Learn which cells are never occupied instead of inventing a
        # disc for every node.
        self.occupancy = np.zeros(rows * cols)
        self.locked_frames = 0
        self.forced_occupied: np.ndarray | None = None
        # Optional callback(predicted_xy, predicted_radius) -> [x, y, r] | None,
        # used to hunt for cells the global detector missed.
        self.refiner = None
        self._uv = _lattice_uv(rows, cols)
        self._neighbour = self._neighbour_table()

    def _neighbour_table(self) -> list[np.ndarray]:
        uv = self._uv
        d = np.abs(uv[:, None, :] - uv[None, :, :]).max(axis=2)
        return [np.flatnonzero((d[k] > 0) & (d[k] <= 2)) for k in range(len(uv))]

    def reset(self) -> None:
        self.model = None
        self.coasting = 0

    def vacant(self) -> np.ndarray:
        if self.forced_occupied is not None:
            return ~self.forced_occupied
        if self.locked_frames < self.occupancy_warmup:
            return np.zeros(self.count, dtype=bool)
        return self.occupancy < self.vacancy_threshold

    def update(self, cands: np.ndarray) -> GridModel | None:
        model = self._track(cands) if self.model is not None else None
        if model is None:
            model = self._bootstrap(cands)
            if model is not None:
                self.generation += 1
        if model is None:
            if self.model is not None and self.coasting < self.max_coast:
                self.coasting += 1
                self.model.raw = cands
                self.model.observed = np.zeros(self.model.count, dtype=bool)
                return self.model
            self.reset()
            return None
        self.coasting = 0
        self.locked_frames += 1
        self.occupancy += self.occupancy_alpha * (model.observed.astype(float) - self.occupancy)
        model.vacant = self.vacant()
        self.model = model
        return model

    def _bootstrap(self, cands: np.ndarray) -> GridModel | None:
        found = initialise(cands, self.rows, self.cols)
        if found is None:
            return None
        uv, cand_idx = found
        xy = cands[cand_idx, :2]
        H = _fit_transform(uv, xy)
        if H is None:
            return None
        nodes = _apply(H, self._uv)
        observed = np.zeros(self.count, dtype=bool)
        cells = (uv[:, 1] * self.cols + uv[:, 0]).astype(int)
        nodes[cells] = xy
        observed[cells] = True
        radii = _radius_model(uv, cands[cand_idx, 2], self._uv)
        radii[cells] = cands[cand_idx, 2]
        return GridModel(
            self.rows, self.cols, nodes, radii, observed, self._spacing(nodes), H, cands
        )

    def _track(self, cands: np.ndarray) -> GridModel | None:
        prev = self.model
        assert prev is not None
        tol = self.gate_frac * prev.spacing
        match = _assign(prev.nodes, cands, tol)
        obs = match >= 0
        if obs.sum() < self.min_observed:
            return None

        uv_obs = self._uv[obs]
        xy_obs = cands[match[obs], :2]
        H = _fit_transform(uv_obs, xy_obs)
        pred = _apply(H, self._uv) if H is not None else prev.nodes.copy()

        vacant = self.vacant()
        found_x = {k: cands[match[k], :2] for k in np.flatnonzero(obs)}
        found_r = {k: float(cands[match[k], 2]) for k in np.flatnonzero(obs)}

        if self.refiner is not None:
            r_guess = (
                float(np.median([found_r[k] for k in found_x])) if found_x else prev.spacing / 2.6
            )
            for k in np.flatnonzero(~obs & ~vacant):
                hit = self.refiner(pred[k], r_guess)
                if hit is not None:
                    found_x[k] = hit[:2]
                    found_r[k] = float(hit[2])
                    obs[k] = True
            if len(found_x) > int(np.count_nonzero(match >= 0)):
                uv_obs = self._uv[obs]
                xy_obs = np.array([found_x[k] for k in np.flatnonzero(obs)])
                H = _fit_transform(uv_obs, xy_obs)
                pred = _apply(H, self._uv) if H is not None else pred

        nodes = pred.copy()
        if obs.any():
            xy_obs = np.array([found_x[k] for k in np.flatnonzero(obs)])
            nodes[obs] = xy_obs
            if self.snap > 0.0:
                nodes[obs] = (1.0 - self.snap) * xy_obs + self.snap * pred[obs]

        # Missing cells follow their detected neighbours rather than the global
        # model, so a locally deforming lattice does not drag them off.
        for k in np.flatnonzero(~obs & ~vacant):
            near = self._neighbour[k][obs[self._neighbour[k]]]
            if len(near):
                disp = np.median(nodes[near] - prev.nodes[near], axis=0)
                nodes[k] = prev.nodes[k] + disp

        r_obs = np.array([found_r[k] for k in np.flatnonzero(obs)]) if obs.any() else np.zeros(0)
        radii = _radius_model(self._uv[obs], r_obs, self._uv)
        radii[obs] = r_obs
        return GridModel(self.rows, self.cols, nodes, radii, obs, self._spacing(nodes), H, cands)

    @property
    def count(self) -> int:
        return self.rows * self.cols

    def _spacing(self, nodes: np.ndarray) -> float:
        d = np.linalg.norm(nodes[:, None, :] - nodes[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        return float(np.median(np.min(d, axis=1)))
