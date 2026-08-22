"""Hough circle detection tuned for a board of identical light-coloured discs."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .validate import DEFAULT_PROFILE, DiscProfile, filter_candidates, measure

_HAS_ALT = hasattr(cv2, "HOUGH_GRADIENT_ALT")

# How far the guided pass may fall below the frame's own confirmed discs.
RELAX_RADIAL = 0.55
REF_RADIAL_FRAC = 0.45
REF_CONTRAST_FRAC = 0.35


@dataclass
class DetectParams:
    """Knobs for the Hough stage.  Radii are in working-resolution pixels."""

    r_min: float = 20.0
    r_max: float = 140.0
    min_dist: float = 60.0
    # HOUGH_GRADIENT_ALT: param1 is the Scharr edge threshold, param2 the
    # "circle perfectness" in 0..1.  Sweeping param2 trades recall for noise;
    # the grid stage downstream removes noise, so we favour recall.
    alt_param1: float = 300.0
    alt_param2: tuple[float, ...] = (0.88, 0.78)
    # Classic HOUGH_GRADIENT, run as a second opinion: param2 is a vote count.
    std_param1: float = 120.0
    std_param2: tuple[float, ...] = (34.0, 24.0)
    dp: float = 1.5
    blur: int = 5
    clahe: bool = False
    clahe_clip: float = 2.0
    use_alt: bool = _HAS_ALT
    use_std: bool = True
    merge_frac: float = 0.6  # centres closer than merge_frac * min_dist are one circle
    validate: bool = True
    profile: DiscProfile = field(default_factory=DiscProfile)


@dataclass
class Detection:
    circles: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))  # x, y, r
    raw: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))  # before validation
    gray: np.ndarray | None = None


def preprocess(bgr: np.ndarray, params: DetectParams) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if params.clahe:
        clahe = cv2.createCLAHE(clipLimit=params.clahe_clip, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    if params.blur >= 3:
        gray = cv2.medianBlur(gray, params.blur | 1)
    return gray


def _hough(
    gray: np.ndarray,
    method: int,
    dp: float,
    min_dist: float,
    p1: float,
    p2: float,
    r_min: float,
    r_max: float,
) -> np.ndarray:
    found = cv2.HoughCircles(
        gray,
        method,
        dp=dp,
        minDist=max(min_dist, 1.0),
        param1=p1,
        param2=p2,
        minRadius=max(round(r_min), 1),
        maxRadius=max(round(r_max), 2),
    )
    if found is None:
        return np.zeros((0, 3))
    return np.asarray(found[0], dtype=float)


def _merge(circles: np.ndarray, tol: float) -> np.ndarray:
    """Greedy non-maximum suppression on centres, keeping earlier (stronger) hits."""
    kept: list[np.ndarray] = []
    for cand in circles:
        for k in kept:
            if float(np.hypot(cand[0] - k[0], cand[1] - k[1])) < tol:
                break
        else:
            kept.append(cand)
    return np.asarray(kept, dtype=float) if kept else np.zeros((0, 3))


def detect_circles(bgr: np.ndarray, params: DetectParams) -> Detection:
    """Run Hough at several sensitivities and merge the candidates.

    ALT gives accurate radii but drops low-contrast discs; the classic gradient
    method catches those with sloppier radii.  Union first, filter later.
    """
    gray = preprocess(bgr, params)
    batches: list[np.ndarray] = []
    if params.use_alt and _HAS_ALT:
        batches.extend(
            _hough(
                gray,
                cv2.HOUGH_GRADIENT_ALT,
                params.dp,
                params.min_dist,
                params.alt_param1,
                p2,
                params.r_min,
                params.r_max,
            )
            for p2 in params.alt_param2
        )
    if params.use_std or not batches:
        batches.extend(
            _hough(
                gray,
                cv2.HOUGH_GRADIENT,
                1.0,
                params.min_dist,
                params.std_param1,
                p2,
                params.r_min,
                params.r_max,
            )
            for p2 in params.std_param2
        )
    stacked = (
        np.concatenate([b for b in batches if len(b)], axis=0)
        if any(len(b) for b in batches)
        else np.zeros((0, 3))
    )
    raw = _merge(stacked, params.min_dist * params.merge_frac)
    circles = raw
    if params.validate and len(raw):
        circles, _ = filter_candidates(bgr, gray, raw, params.profile)
    return Detection(circles=circles, raw=raw, gray=gray)


def refine_in_roi(
    bgr: np.ndarray,
    gray: np.ndarray,
    centre: np.ndarray,
    radius: float,
    *,
    half: float = 1.8,
    r_lo: float = 0.72,
    r_hi: float = 1.32,
    accept: float = 1.0,
    param1: float = 100.0,
    param2: tuple[float, ...] = (26.0, 18.0, 13.0),
    profile: DiscProfile | None = None,
    ref: dict[str, float] | None = None,
) -> np.ndarray | None:
    """Look again for one disc, in a small window around where we expect it.

    The discs tilt as the array deforms, so their top face projects to an
    ellipse and the global pass loses them.  Searching a predicted cell on its
    own lets us drop the vote threshold far enough to catch those without
    letting the rest of the frame flood us with false circles.
    """
    h, w = gray.shape[:2]
    pad = max(round(half * radius), 8)
    cx, cy = float(centre[0]), float(centre[1])
    x0, y0 = max(int(cx) - pad, 0), max(int(cy) - pad, 0)
    x1, y1 = min(int(cx) + pad, w), min(int(cy) + pad, h)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    roi = gray[y0:y1, x0:x1]

    hits: list[np.ndarray] = []
    for p2 in param2:
        found = _hough(
            roi, cv2.HOUGH_GRADIENT, 1.0, radius, param1, p2, r_lo * radius, r_hi * radius
        )
        if len(found):
            hits.append(found)
        if hits:
            break  # the strictest threshold that finds anything wins
    if not hits:
        return None
    cand = hits[0]
    cand[:, 0] += x0
    cand[:, 1] += y0
    d = np.hypot(cand[:, 0] - cx, cand[:, 1] - cy)
    ok = d <= accept * radius
    if not ok.any():
        return None
    best = cand[ok][np.argmin(d[ok])]

    prof = profile or DEFAULT_PROFILE
    m = measure(bgr, gray, best.reshape(1, 3), prof)
    # A tilted disc projects to an ellipse and loses rim support, so the guided
    # pass has to accept less of it -- but it must still look like the discs this
    # frame already confirmed.  Fixed low thresholds here will cheerfully invent
    # a disc out of a blank wall.
    min_radial = prof.min_radial_support * RELAX_RADIAL
    min_contrast = prof.min_contrast
    if ref is not None:
        min_radial = max(min_radial, ref["radial"] * REF_RADIAL_FRAC)
        min_contrast = max(min_contrast, ref["contrast"] * REF_CONTRAST_FRAC)
    if (
        m["radial"][0] < min_radial
        or m["contrast"][0] < min_contrast
        or m["saturation"][0] > prof.max_saturation
    ):
        return None
    return best
