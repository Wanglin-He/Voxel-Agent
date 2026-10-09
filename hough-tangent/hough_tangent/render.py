"""Overlay drawing.  Everything is drawn with 1/16-pixel subpixel precision."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .hull import DiskHull

_SHIFT = 4
_SUB = 1 << _SHIFT


def _p(pt) -> tuple[int, int]:
    return (round(float(pt[0]) * _SUB), round(float(pt[1]) * _SUB))


def _poly(pts: np.ndarray) -> np.ndarray:
    return np.rint(np.asarray(pts, dtype=float) * _SUB).astype(np.int32).reshape(-1, 1, 2)


@dataclass
class Style:
    hull: tuple[int, int, int] = (255, 216, 0)  # BGR, cyan
    hull_thickness: float = 3.0
    fill_alpha: float = 0.10
    observed: tuple[int, int, int] = (90, 240, 90)  # detected this frame
    inferred: tuple[int, int, int] = (70, 190, 255)  # filled in from the lattice
    circle_thickness: float = 1.6
    centre: tuple[int, int, int] = (255, 255, 255)
    tangent_point: tuple[int, int, int] = (255, 255, 255)
    tangent_point_radius: float = 3.0
    tangent_extend: float = 0.0  # px beyond each tangency point
    tangent_line: tuple[int, int, int] = (200, 200, 200)
    raw: tuple[int, int, int] = (110, 110, 110)
    hud: tuple[int, int, int] = (255, 255, 255)
    hud_scale: float = 1.0
    show_hud: bool = True
    show_circles: bool = True
    show_tangent_points: bool = True
    sharp_corners: bool = True  # meet the tangents at a point instead of rounding
    max_arc_deg: float = 90.0  # split longer arcs so their ends corner instead of rounding
    max_corner_reach: float = 6.0


DEFAULT_STYLE = Style()


def _thick(v: float, scale: float) -> int:
    return max(1, round(v * scale))


def draw_overlay(
    canvas: np.ndarray,
    centres: np.ndarray,
    radii: np.ndarray,
    observed: np.ndarray | None,
    hull: DiskHull | None,
    *,
    style: Style = DEFAULT_STYLE,
    px: float = 1.0,
    raw: np.ndarray | None = None,
    hud_lines: list[str] | None = None,
    holes: list[DiskHull] | None = None,
) -> np.ndarray:
    """Draw onto ``canvas`` in place.  ``px`` scales line weights with resolution.

    ``holes`` are outlines round empty pockets inside ``hull``; they are drawn
    the same way and left out of the fill.
    """
    if raw is not None and len(raw):
        for x, y, r in raw:
            cv2.circle(
                canvas,
                _p((x, y)),
                round(float(r) * _SUB),
                style.raw,
                _thick(1.0, px),
                cv2.LINE_AA,
                _SHIFT,
            )

    outlines = [h for h in [hull, *(holes or [])] if h is not None]
    if style.sharp_corners and style.max_arc_deg > 0:
        outlines = [h.split_long_arcs(np.deg2rad(style.max_arc_deg)) for h in outlines]
    polys = [
        h.polyline(1.0, sharp=style.sharp_corners, max_reach=style.max_corner_reach)
        for h in outlines
    ]
    polys = [p for p in polys if len(p) >= 3]
    if polys and style.fill_alpha > 0:
        overlay = canvas.copy()
        # fillPoly fills even-odd, so a hole drawn alongside its outline stays clear.
        cv2.fillPoly(overlay, [_poly(p) for p in polys], style.hull, cv2.LINE_AA, _SHIFT)
        cv2.addWeighted(overlay, style.fill_alpha, canvas, 1.0 - style.fill_alpha, 0.0, canvas)

    if style.show_circles:
        for k in range(len(centres)):
            x, y = centres[k]
            r = radii[k]
            seen = True if observed is None else bool(observed[k])
            colour = style.observed if seen else style.inferred
            cv2.circle(
                canvas,
                _p((x, y)),
                round(float(r) * _SUB),
                colour,
                _thick(style.circle_thickness, px),
                cv2.LINE_AA,
                _SHIFT,
            )
            cv2.circle(
                canvas, _p((x, y)), round(2.0 * px * _SUB), style.centre, -1, cv2.LINE_AA, _SHIFT
            )

    tangents = [t for h in outlines for t in h.tangents]
    if style.tangent_extend > 0:
        for t in tangents:
            a, b = np.array(t.p_a, dtype=float), np.array(t.p_b, dtype=float)
            d = b - a
            n = float(np.linalg.norm(d))
            if n < 1e-6:
                continue
            d /= n
            ext = style.tangent_extend * px
            cv2.line(
                canvas,
                _p(a - d * ext),
                _p(b + d * ext),
                style.tangent_line,
                _thick(1.0, px),
                cv2.LINE_AA,
                _SHIFT,
            )
    if polys:
        cv2.polylines(
            canvas,
            [_poly(p) for p in polys],
            True,
            style.hull,
            _thick(style.hull_thickness, px),
            cv2.LINE_AA,
            _SHIFT,
        )
    if style.show_tangent_points:
        for t in tangents:
            for pt in (t.p_a, t.p_b):
                cv2.circle(
                    canvas,
                    _p(pt),
                    round(style.tangent_point_radius * px * _SUB),
                    style.tangent_point,
                    -1,
                    cv2.LINE_AA,
                    _SHIFT,
                )

    if style.show_hud and hud_lines:
        scale = 0.5 * style.hud_scale * px
        y = int(28 * px)
        for line in hud_lines:
            cv2.putText(
                canvas,
                line,
                (int(14 * px), y),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                (0, 0, 0),
                _thick(3.0, px),
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                line,
                (int(14 * px), y),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                style.hud,
                _thick(1.0, px),
                cv2.LINE_AA,
            )
            y += int(26 * px * style.hud_scale)
    return canvas
