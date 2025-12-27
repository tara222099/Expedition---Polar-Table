from __future__ import annotations

import csv
import io
from datetime import datetime, timezone
from typing import Iterable, Optional

from dateutil import parser as dtparser

from .expedition_detector import Sample


class CsvFormatError(ValueError):
    pass


def _norm_header(h: str) -> str:
    return (h or "").strip().lower()


def _pick_time_column(fieldnames: Iterable[str]) -> Optional[str]:
    # Expedition logs vary; prefer explicit "time"/"date" style columns.
    candidates = list(fieldnames)
    normalized = {c: _norm_header(c) for c in candidates}

    # Strong preferences first
    for wanted in ("time", "utc", "date", "datetime", "timestamp"):
        for c in candidates:
            if normalized[c] == wanted:
                return c

    # Substring match
    for c in candidates:
        nh = normalized[c]
        if "time" in nh or "date" in nh:
            return c
        if "utc" in nh:
            return c

    return None


def _pick_twa_column(fieldnames: Iterable[str]) -> Optional[str]:
    candidates = list(fieldnames)
    normalized = {c: _norm_header(c) for c in candidates}

    # Strong preferences first
    for wanted in ("twa", "true wind angle"):
        for c in candidates:
            if normalized[c] == wanted:
                return c

    # Substring match (covers e.g. "TWA (deg)", "True Wind Angle (TWA)")
    for c in candidates:
        nh = normalized[c]
        if "twa" in nh:
            return c
        if "true wind angle" in nh:
            return c
    return None


def _parse_datetime_utc(s: str) -> datetime:
    dt = dtparser.parse(s)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def read_samples_from_csv_bytes(data: bytes) -> list[Sample]:
    """
    Read Expedition CSV as bytes, auto-detect delimiter, and extract (time, TWA).
    Assumes the time column is in UTC (or has TZ info).
    """
    if not data:
        raise CsvFormatError("CSV is empty.")

    # Handle UTF-8 BOM if present
    text = data.decode("utf-8-sig", errors="replace")
    buf = io.StringIO(text)

    # Sniff dialect (fallback to excel)
    sample = text[:8192]
    try:
        dialect = csv.Sniffer().sniff(sample)
    except Exception:
        dialect = csv.excel

    reader = csv.DictReader(buf, dialect=dialect)
    if not reader.fieldnames:
        raise CsvFormatError("CSV header row not found.")

    time_col = _pick_time_column(reader.fieldnames)
    twa_col = _pick_twa_column(reader.fieldnames)
    if time_col is None:
        raise CsvFormatError(f"Could not find a time column. Headers: {reader.fieldnames}")
    if twa_col is None:
        raise CsvFormatError(f"Could not find a TWA column. Headers: {reader.fieldnames}")

    out: list[Sample] = []
    for row in reader:
        raw_t = (row.get(time_col) or "").strip()
        raw_twa = (row.get(twa_col) or "").strip()
        if not raw_t or not raw_twa:
            continue
        try:
            t_utc = _parse_datetime_utc(raw_t)
            twa = float(raw_twa)
        except Exception:
            continue
        out.append(Sample(t_utc=t_utc, twa=twa))

    if not out:
        raise CsvFormatError(
            f"No usable rows found. Detected time='{time_col}', twa='{twa_col}'."
        )
    out.sort(key=lambda s: s.t_utc)
    return out

