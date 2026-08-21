"""Throw away Hough candidates that are not actually discs.

HoughCircles is happy to hallucinate circles in cluttered texture -- on this
footage the cable bundle under the board produces a hundred of them.  Three
cheap image measurements separate a real disc from a coincidence:

* radial support -- does the intensity gradient along the candidate's rim
  actually point across the rim, all the way round?
* contrast -- is the inside brighter than the surrounding ring?
* colour -- the discs are unpainted white plastic, so nearly unsaturated.

Everything is sampled on rings with fancy indexing, so the cost is a few
hundred lookups per candidate rather than a per-pixel pass.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class DiscProfile:
    """What a genuine disc looks like.  Loosen these for a different subject."""

    min_radial_support: float = 0.34
    min_contrast: float = 14.0
    max_saturation: float = 95.0
    bright_on_dark: bool = True
    grad_threshold: float = 12.0
    radius_tolerance: float = 0.26  # keep radii within +/- this of the modal radius
    n_angles: int = 64


DEFAULT_PROFILE = DiscProfile()


def _sample(
    img: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    rr: np.ndarray,
    cos_a: np.ndarray,
    sin_a: np.ndarray,
) -> np.ndarray:
    h, w = img.shape[:2]
    xs = np.rint(cx[:, None] + rr[:, None] * cos_a[None, :]).astype(int)
    ys = np.rint(cy[:, None] + rr[:, None] * sin_a[None, :]).astype(int)
    np.clip(xs, 0, w - 1, out=xs)
    np.clip(ys, 0, h - 1, out=ys)
    return img[ys, xs]


def measure(
    bgr: np.ndarray, gray: np.ndarray, circles: np.ndarray, profile: DiscProfile = DEFAULT_PROFILE
) -> dict[str, np.ndarray]:
    """Per-candidate radial support, inside/outside contrast and saturation."""
    n = len(circles)
    if n == 0:
        return {k: np.zeros(0) for k in ("radial", "contrast", "saturation")}
    cx, cy, r = circles[:, 0], circles[:, 1], circles[:, 2]
    ang = np.linspace(0.0, 2.0 * np.pi, profile.n_angles, endpoint=False)
    ca, sa = np.cos(ang), np.sin(ang)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    sx = _sample(gx, cx, cy, r, ca, sa)
    sy = _sample(gy, cx, cy, r, ca, sa)
    mag = np.hypot(sx, sy)
    with np.errstate(invalid="ignore", divide="ignore"):
        align = (sx * ca[None, :] + sy * sa[None, :]) / np.maximum(mag, 1e-6)
    if profile.bright_on_dark:
        align = -align  # inside is brighter, so the gradient points inwards
    radial = np.mean((mag > profile.grad_threshold) & (align > 0.70), axis=1)

    grayf = gray.astype(np.float32)
    inner = np.mean(
        [_sample(grayf, cx, cy, r * f, ca, sa).mean(axis=1) for f in (0.30, 0.55, 0.78)], axis=0
    )
    outer = np.mean(
        [_sample(grayf, cx, cy, r * f, ca, sa).mean(axis=1) for f in (1.22, 1.42)], axis=0
    )
    contrast = inner - outer

    sat = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32)
    saturation = np.mean(
        [_sample(sat, cx, cy, r * f, ca, sa).mean(axis=1) for f in (0.30, 0.55, 0.78)], axis=0
    )
    return {"radial": radial, "contrast": contrast, "saturation": saturation}


def _modal_radius(r: np.ndarray, tol: float) -> float:
    """Radius shared by the largest group of candidates (all discs are alike)."""
    if len(r) == 0:
        return 0.0
    best, best_n = float(np.median(r)), -1
    for cand in r:
        inside = np.abs(r - cand) <= tol * cand
        if inside.sum() > best_n:
            best_n, best = int(inside.sum()), float(np.median(r[inside]))
    return best


def filter_candidates(
    bgr: np.ndarray,
    gray: np.ndarray,
    circles: np.ndarray,
    profile: DiscProfile = DEFAULT_PROFILE,
    use_radius_mode: bool = True,
) -> tuple[np.ndarray, dict]:
    """Return the candidates that look like discs, plus the raw measurements."""
    if len(circles) == 0:
        return circles, {}
    m = measure(bgr, gray, circles, profile)
    keep = (
        (m["radial"] >= profile.min_radial_support)
        & (m["contrast"] >= profile.min_contrast)
        & (m["saturation"] <= profile.max_saturation)
    )
    kept = circles[keep]
    if use_radius_mode and len(kept) >= 4:
        r0 = _modal_radius(kept[:, 2], profile.radius_tolerance)
        kept = kept[np.abs(kept[:, 2] - r0) <= profile.radius_tolerance * r0]
    return kept, m
