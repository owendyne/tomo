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
_AVG_LABELS = ("avg", "average", "mean", "std", "sd", "stdev")


def _num(cell):
    try:
        return float(str(cell).strip())
    except (TypeError, ValueError):
        return None


def _to_fraction(levels):
    """Greyscale levels may be fractions (0..1), percent (0..100) or 8-bit (0..255)."""
    top = max(levels)
    if top > 100.5:
        return [v / 255.0 for v in levels]
    if top > 1.5:
        return [v / 100.0 for v in levels]
    return list(levels)


def parse_calibration_csv(path):
    """Read a greyscale calibration CSV and return a "lut" config.

    Two layouts are accepted:
      * long  — one reading per row: a level column and a measured-output column
                (header optional; repeated levels are averaged).  Columns are found
                by header name (grey/gray/level/pixel ... and irradiance/power/
                intensity/output/measured ...), else the first and last numeric
                columns are used.
      * wide  — first row is the levels (an optional label cell first), each
                following row is one set of readings.  Rows labelled
                Average/Mean/SD are skipped, as are blank cells.
    """
    with open(path, newline="", encoding="utf-8-sig") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        rows = [r for r in csv.reader(f, dialect) if any(c.strip() for c in r)]
    if len(rows) < 2:
        raise ValueError("Calibration file has fewer than two rows")

    readings = {}                                   # level -> list of measured values
    first_numeric = [v for v in (_num(c) for c in rows[0]) if v is not None]
    if len(first_numeric) >= 3:                     # wide layout
        head = rows[0]
        lead = 0 if _num(head[0]) is not None else 1
        levels = [_num(c) for c in head[lead:]]
        for r in rows[1:]:
            label = (r[0] if r else "").strip().lower()
            if lead and label.startswith(_AVG_LABELS):
                continue
            for lv, cell in zip(levels, r[lead:]):
                v = _num(cell)
                if lv is not None and v is not None:
                    readings.setdefault(lv, []).append(v)
    else:                                           # long layout
        header = [c.strip().lower() for c in rows[0]]
        has_header = all(_num(c) is None for c in rows[0] if c.strip())
        data = rows[1:] if has_header else rows

        def find(*key_groups):
            # Earlier groups win, so an exact 0-255 pixel column beats a nominal
            # "greyscale fraction" column when a file has both.
            if has_header:
                for keys in key_groups:
                    for i, h in enumerate(header):
                        if any(k in h for k in keys):
                            return i
            return None
        li = find(("pixel", "code", "dn"), ("grey", "gray", "level", "input", "command"))
        oi = find(("irradiance", "power", "intensity", "output", "measured", "mw", "reading"))
        for r in data:
            nums = [(i, _num(c)) for i, c in enumerate(r) if _num(c) is not None]
            if len(nums) < 2:
                continue
            lv = _num(r[li]) if li is not None and li < len(r) else nums[0][1]
            v = _num(r[oi]) if oi is not None and oi < len(r) else nums[-1][1]
            if lv is not None and v is not None:
                readings.setdefault(lv, []).append(v)

    if len(readings) < 2:
        raise ValueError("Calibration needs readings at two or more greyscale levels")
    raw_levels = sorted(readings)
    levels = _to_fraction(raw_levels)
    response = [float(np.mean(readings[lv])) for lv in raw_levels]
    if abs(levels[-1] - 1.0) > 1e-3:
        raise ValueError("Calibration must include a full-white (255 / 100% / 1.0) reading")
    if response[-1] <= 0:
        raise ValueError("Full-white reading must be positive")
    return {"mode": "lut", "levels": levels, "response": response,
            "source": os.path.basename(path),
            "n_readings": int(sum(len(v) for v in readings.values()))}


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
