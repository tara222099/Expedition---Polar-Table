from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional


JST = timezone(timedelta(hours=9))


@dataclass(frozen=True)
class Sample:
    t_utc: datetime
    twa: float


@dataclass(frozen=True)
class Event:
    time_jst: datetime
    type: str  # "Tack" or "Jibe"
    twa_before: float
    twa_after: float


def _sign(x: float, *, eps: float = 1e-9) -> int:
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def to_jst(dt_utc: datetime) -> datetime:
    return _ensure_utc(dt_utc).astimezone(JST)


def detect_tack_jibe_events(
    samples: Iterable[Sample],
    *,
    min_duration_seconds: float = 20.0,
    tack_abs_twa_deg: float = 30.0,
    jibe_abs_twa_deg: float = 160.0,
) -> list[Event]:
    """
    Detect all tack/jibe events based on:
      - sign(TWA) flips (+ -> - or - -> +)
      - then |TWA| >= threshold continuously for >= min_duration_seconds

    Event time is the instant when the continuous condition reaches min_duration_seconds.
    TWA_before is the representative value right before the sign flip (last non-zero old sign).
    TWA_after is the representative value at the start of the qualifying run after the flip.
    """
    s = sorted(samples, key=lambda x: x.t_utc)
    if len(s) < 2:
        return []

    # Normalize timestamps to UTC (aware) and drop non-finite-ish values defensively
    normalized: list[Sample] = []
    for row in s:
        try:
            t_utc = _ensure_utc(row.t_utc)
            twa = float(row.twa)
        except Exception:
            continue
        normalized.append(Sample(t_utc=t_utc, twa=twa))

    events: list[Event] = []
    i = 1
    while i < len(normalized):
        prev = normalized[i - 1]
        cur = normalized[i]
        prev_sign = _sign(prev.twa)
        cur_sign = _sign(cur.twa)

        # Only react to a true sign flip (+/-). Ignore transitions involving 0.
        if prev_sign == 0 or cur_sign == 0 or prev_sign == cur_sign:
            i += 1
            continue

        old_sign = prev_sign
        new_sign = cur_sign

        # Representative TWA_before: last sample before i with old_sign (skipping zeros).
        twa_before = prev.twa
        j = i - 1
        while j >= 0:
            sj = normalized[j]
            if _sign(sj.twa) == old_sign:
                twa_before = sj.twa
                break
            j -= 1

        def try_threshold(abs_threshold: float) -> Optional[tuple[int, int]]:
            """
            Returns (start_idx, event_idx) where:
              - start_idx is the first index after flip where condition is true
              - event_idx is the first index where duration >= min_duration_seconds
            """
            # Find start of qualifying run after the flip.
            start_idx: Optional[int] = None
            k = i
            while k < len(normalized):
                sk = normalized[k]
                if _sign(sk.twa) == new_sign and abs(sk.twa) >= abs_threshold:
                    start_idx = k
                    break
                # If it flips back before starting, abort.
                if _sign(sk.twa) == old_sign:
                    return None
                k += 1
            if start_idx is None:
                return None

            start_t = normalized[start_idx].t_utc
            last_good_idx = start_idx
            k = start_idx
            while k < len(normalized):
                sk = normalized[k]
                if _sign(sk.twa) != new_sign or abs(sk.twa) < abs_threshold:
                    break
                last_good_idx = k
                if (sk.t_utc - start_t).total_seconds() >= min_duration_seconds:
                    return (start_idx, k)
                k += 1
            return None

        chosen: Optional[tuple[str, float, int, int]] = None
        # Priority: Jibe first (higher threshold), then Tack.
        for typ, thr in (("Jibe", jibe_abs_twa_deg), ("Tack", tack_abs_twa_deg)):
            res = try_threshold(thr)
            if res is not None:
                start_idx, event_idx = res
                chosen = (typ, thr, start_idx, event_idx)
                break

        if chosen is None:
            i += 1
            continue

        typ, _thr, start_idx, event_idx = chosen
        twa_after = normalized[start_idx].twa
        t_event_jst = to_jst(normalized[event_idx].t_utc)

        events.append(
            Event(
                time_jst=t_event_jst,
                type=typ,
                twa_before=float(twa_before),
                twa_after=float(twa_after),
            )
        )

        # Skip forward past the qualifying run to avoid double-counting within one maneuver.
        i = event_idx + 1

    events.sort(key=lambda e: e.time_jst)
    return events

