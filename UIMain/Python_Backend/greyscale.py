"""
Projector greyscale (gamma) correction for exported projection frames.

VAMToolbox assumes the light a projector delivers is proportional to the pixel
value.  Real DLP light engines are not linear: a typical one delivers ~25% of
full irradiance at 50% grey and ~2% at 10% grey.  Optimized sinograms are mostly
dim pixels, so displaying them uncorrected crushes the low end and distorts the
dose.  This module builds a 256-entry lookup table (LUT) that maps the linear
intensity VAMToolbox asks for onto the 8-bit value that makes the projector
actually deliver it, and applies it to the uint8 frames.

A correction config lives on the projector profile (the GUI's ``greyscale``
field) and is one of:

    {"mode": "off"}
    {"mode": "gamma", "gamma": 2.2}                 # power-law display model
    {"mode": "lut", "levels": [...], "response": [...], "source": "file.csv"}

For "lut", ``levels`` are displayed greyscale fractions (0..1) and ``response``
the measured output at each level (any units, e.g. mW/cm^2 from a meter).
"""
import csv
import os

import numpy as np

_BLACK_LEVEL_TOL = 1e-6


def _num(cell):
    try:
        return float(str(cell).strip())
    except (TypeError, ValueError):
        return None


def parse_calibration_csv(path):
    """Read a greyscale calibration CSV and return a "lut" config.

    The file has two columns and one row per greyscale level:

        greyscale_fraction,irradiance_mW_cm2
        0.0,0.079
        0.5,3.373
        1.0,13.602

    Column 1 is the displayed greyscale level as a fraction (0 = black,
    1 = full white).  Column 2 is the averaged irradiance measured at that level;
    other units work too, because readings are normalised to full white.  A
    header row is optional, and a full-white row is required.
    """
    with open(path, newline="", encoding="utf-8-sig") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        rows = [r for r in csv.reader(f, dialect) if any(c.strip() for c in r)]

    points = {}                                     # greyscale fraction -> irradiance
    for i, r in enumerate(rows):
        cells = [c for c in r if c.strip()]
        nums = [_num(c) for c in cells]
        if any(v is None for v in nums):
            if i == 0:
                continue                            # header row
            raise ValueError(f"Row {i + 1} is not numeric: {', '.join(r)}")
        if len(nums) != 2:
            raise ValueError(f"Row {i + 1} has {len(nums)} values; expected 2 "
                             "(greyscale fraction, irradiance)")
        level, value = nums
        if not 0.0 <= level <= 1.0:
            raise ValueError(f"Row {i + 1}: greyscale fraction {level:g} is outside 0-1")
        if level in points:
            raise ValueError(f"Greyscale fraction {level:g} is listed more than once")
        points[level] = value

    if len(points) < 2:
        raise ValueError("Calibration needs at least two greyscale levels")
    levels = sorted(points)
    response = [points[lv] for lv in levels]
    if abs(levels[-1] - 1.0) > 1e-6:
        raise ValueError("Calibration must include a full-white (1.0) row")
    if response[-1] <= 0:
        raise ValueError("Full-white irradiance must be positive")
    return {"mode": "lut", "levels": levels, "response": response,
            "source": os.path.basename(path)}


def _normalised_response(cfg):
    """Monotone response curve on [0, 1], normalised so full white = 1."""
    lv = np.asarray(cfg["levels"], dtype=float)
    rs = np.asarray(cfg["response"], dtype=float)
    order = np.argsort(lv)
    lv, rs = lv[order], rs[order]
    if lv[0] > _BLACK_LEVEL_TOL:                    # no black reading: assume no output
        lv, rs = np.r_[0.0, lv], np.r_[0.0, rs]
    rs = np.maximum.accumulate(rs / rs[-1])         # meter noise must not fold the curve
    return lv, rs


def correction_lut(cfg):
    """uint8[256] mapping requested linear intensity -> displayed value, or None."""
    if not cfg or cfg.get("mode", "off") == "off":
        return None
    k = np.arange(256) / 255.0
    if cfg["mode"] == "gamma":
        gamma = float(cfg.get("gamma", 2.2))
        if gamma <= 0:
            raise ValueError("Gamma must be positive")
        out = k ** (1.0 / gamma)
    elif cfg["mode"] == "lut":
        lv, rs = _normalised_response(cfg)
        g = np.linspace(0.0, 1.0, 4097)
        r = np.interp(g, lv, rs) + g * 1e-9         # strictly increasing for the inverse
        # Requests below the black level map to 0 (the dimmest the projector can do).
        out = np.interp(k, r, g)
    else:
        raise ValueError(f"Unknown greyscale correction mode: {cfg['mode']!r}")
    return np.clip(np.round(out * 255.0), 0, 255).astype(np.uint8)


def apply_lut(frames, lut):
    """Apply a LUT to a list of uint8 frames (returns them unchanged when lut is None)."""
    if lut is None:
        return frames
    return [lut[f] for f in frames]


def describe(cfg):
    """Short summary for logs / the run record: mode, effective gamma, black level."""
    if not cfg or cfg.get("mode", "off") == "off":
        return {"mode": "off"}
    if cfg["mode"] == "gamma":
        return {"mode": "gamma", "gamma": float(cfg.get("gamma", 2.2))}
    lv, rs = _normalised_response(cfg)
    black = float(rs[0])
    sel = (lv > 0) & (lv < 1) & (rs > black)
    gamma = None
    if sel.sum() >= 2:                               # log-log slope above the black level
        gamma = float(np.polyfit(np.log(lv[sel]), np.log((rs[sel] - black) / (1 - black)), 1)[0])
    return {"mode": "lut", "source": cfg.get("source"), "n_levels": int(len(cfg["levels"])),
            "effective_gamma": gamma, "black_level_pct": 100.0 * black}
