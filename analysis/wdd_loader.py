"""
wdd_loader.py
-------------
by Andre R. Barbosa, April - October 2026

Parser for WebDAQ-504 .wdd binary files (Limnologen, Linnaeus University).

File structure
--------------
  [binary preamble]  ~564 bytes
  [JSON header]      variable length — contains channel info, Fs, sample count
  [raw data]         N_samples × N_channels × float64 (little-endian), interleaved

Usage
-----
    from wdd_loader import load_wdd
    data, fs, ch_names, meta = load_wdd("path/to/file.wdd")
    ch1, ch2 = data[:, 0], data[:, 1]   # units: g
"""

import json
import numpy as np


def load_wdd(fpath: str):
    """Load a WebDAQ-504 .wdd file.

    Returns
    -------
    data     : np.ndarray, shape (N_samples, N_channels), units = g
    fs       : int, sampling rate in Hz
    ch_names : list[str], channel names from header
    meta     : dict, full JSON metadata from header
    """
    with open(fpath, "rb") as f:
        raw = f.read()

    # ── locate JSON block ──────────────────────────────────────────────────
    json_start = raw.find(b'{"systemInfo"')
    if json_start == -1:
        raise ValueError("Could not find JSON header in .wdd file")

    depth = 0
    json_end = json_start
    for i in range(json_start, len(raw)):
        c = raw[i : i + 1]
        if c == b"{":
            depth += 1
        elif c == b"}":
            depth -= 1
            if depth == 0:
                json_end = i + 1
                break

    meta = json.loads(raw[json_start:json_end].decode("utf-8"))

    # ── extract metadata ───────────────────────────────────────────────────
    fs       = meta["jobDescriptor"]["acquisition"]["sample"]["rate"]
    channels = meta["jobDescriptor"]["channels"]
    n_ch     = len(channels)
    ch_names = [c["name"] for c in channels]

    # ── parse binary data (float64, interleaved) ───────────────────────────
    data = (
        np.frombuffer(raw[json_end:], dtype="<f8")
        .reshape(-1, n_ch)
    )

    return data, fs, ch_names, meta
