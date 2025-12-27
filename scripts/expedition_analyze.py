#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import pandas as pd
from openpyxl import load_workbook


@dataclass(frozen=True)
class PolarRow:
    tws: float
    beat_twa: float | None
    beat_bs: float | None
    run_twa: float | None
    run_bs: float | None
    points: tuple[tuple[float, float], ...]  # (twa, bs)


_WS_RE = re.compile(r"\s+")


def _tokenize_numeric_row(line: str) -> list[str]:
    line = line.strip()
    if not line:
        return []
    if line.startswith("!"):
        return []
    return [t for t in _WS_RE.split(line) if t]


def read_expedition_polar(path: Path) -> tuple[list[PolarRow], pd.DataFrame]:
    """
    Reads Expedition polar-style text files.

    Supported formats:
    - TWS BEAT_TWA BEAT_BS RUN_TWA RUN_BS (TWA BS)...
    - TWS (TWA BS)...
    """
    rows: list[PolarRow] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        toks = _tokenize_numeric_row(raw)
        if not toks:
            continue
        try:
            tws = float(toks[0])
        except ValueError:
            continue

        rest = toks[1:]
        beat_twa = beat_bs = run_twa = run_bs = None

        if len(rest) >= 4 and (len(rest) - 4) % 2 == 0:
            # Assume BEAT/RUN optimum fields present
            beat_twa = float(rest[0])
            beat_bs = float(rest[1])
            run_twa = float(rest[2])
            run_bs = float(rest[3])
            pair_tokens = rest[4:]
        elif len(rest) % 2 == 0:
            pair_tokens = rest
        else:
            raise ValueError(f"Unrecognized polar row format in {path}: {raw!r}")

        pts: list[tuple[float, float]] = []
        for i in range(0, len(pair_tokens), 2):
            pts.append((float(pair_tokens[i]), float(pair_tokens[i + 1])))

        rows.append(
            PolarRow(
                tws=tws,
                beat_twa=beat_twa,
                beat_bs=beat_bs,
                run_twa=run_twa,
                run_bs=run_bs,
                points=tuple(pts),
            )
        )

    if not rows:
        raise ValueError(f"No polar rows found in {path}")

    # Build a wide table: index=TWS, columns=TWA, values=BS
    tws_values = sorted({r.tws for r in rows})
    twa_values = sorted({twa for r in rows for (twa, _bs) in r.points})
    table = pd.DataFrame(index=tws_values, columns=twa_values, dtype=float)
    for r in rows:
        for twa, bs in r.points:
            table.loc[r.tws, twa] = bs
    table.index.name = "tws"
    table.columns.name = "twa"

    return rows, table


def read_expedition_heel_table(path: Path) -> pd.DataFrame:
    """Reads Expedition heel-style table text: TWS (TWA HEEL)..."""
    rows = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        toks = _tokenize_numeric_row(raw)
        if not toks:
            continue
        if toks[0].lower() == "angle":
            continue
        try:
            tws = float(toks[0])
        except ValueError:
            continue
        rest = toks[1:]
        if len(rest) % 2 != 0:
            raise ValueError(f"Unrecognized heel row format in {path}: {raw!r}")
        pts = []
        for i in range(0, len(rest), 2):
            pts.append((float(rest[i]), float(rest[i + 1])))
        rows.append((tws, pts))

    tws_values = sorted({tws for tws, _ in rows})
    twa_values = sorted({twa for _tws, pts in rows for twa, _ in pts})
    table = pd.DataFrame(index=tws_values, columns=twa_values, dtype=float)
    for tws, pts in rows:
        for twa, heel in pts:
            table.loc[tws, twa] = heel
    table.index.name = "tws"
    table.columns.name = "twa"
    return table


def read_sailchart_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Drop trailing empty columns created by extra commas.
    # NOTE: many Expedition-exported CSVs have the first column as "Unnamed: 0" but it
    # actually contains the TWS row labels, so we keep it.
    keep_cols: list[str] = []
    for i, c in enumerate(df.columns):
        if str(c).startswith("Unnamed:") and i != 0:
            continue
        keep_cols.append(c)
    df = df.loc[:, keep_cols]

    # First column is TWS; remaining columns are TWA values as strings
    df = df.rename(columns={df.columns[0]: "tws"})
    df["tws"] = pd.to_numeric(df["tws"], errors="raise")
    df = df.set_index("tws")
    # Normalize column names to float angles where possible
    df.columns = [float(c) for c in df.columns]
    df.columns.name = "twa"
    return df.sort_index().reindex(sorted(df.columns), axis=1)


def read_crossover_xml(path: Path) -> pd.DataFrame:
    tree = ET.parse(path)
    root = tree.getroot()
    rows: list[dict[str, object]] = []
    for el in root.findall(".//element"):
        name = el.attrib.get("name", "")
        group = el.attrib.get("group", "")
        typ = el.attrib.get("type", "")
        for p in el.findall(".//bezierpoints/point"):
            rows.append(
                {
                    "name": name,
                    "group": group,
                    "type": typ,
                    "tws": float(p.attrib["tws"]),
                    "twa": float(p.attrib["twa"]),
                }
            )
    return pd.DataFrame(rows).sort_values(["name", "tws", "twa"]).reset_index(drop=True)


def _interp1(x: float, xp: Iterable[float], fp: Iterable[float]) -> float:
    xp = list(xp)
    fp = list(fp)
    if len(xp) != len(fp) or not xp:
        raise ValueError("xp/fp must have same non-zero length")

    # Ensure sorted by xp
    pairs = sorted(zip(xp, fp))
    xp = [p[0] for p in pairs]
    fp = [p[1] for p in pairs]

    if x <= xp[0]:
        return fp[0]
    if x >= xp[-1]:
        return fp[-1]

    # Find right interval
    lo = 0
    hi = len(xp) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xp[mid] <= x:
            lo = mid
        else:
            hi = mid
    x0, x1 = xp[lo], xp[hi]
    y0, y1 = fp[lo], fp[hi]
    if x1 == x0:
        return y0
    t = (x - x0) / (x1 - x0)
    return (1 - t) * y0 + t * y1


def polar_speed(polar: pd.DataFrame, tws: float, twa: float) -> float:
    """Bilinear-ish interpolation: interpolate in TWA, then interpolate across TWS."""
    tws_grid = [float(x) for x in polar.index.to_list()]
    if not tws_grid:
        raise ValueError("Empty polar table")
    tws_grid_sorted = sorted(tws_grid)

    # Bracket TWS
    if tws <= tws_grid_sorted[0]:
        lo = hi = tws_grid_sorted[0]
    elif tws >= tws_grid_sorted[-1]:
        lo = hi = tws_grid_sorted[-1]
    else:
        lo = max(v for v in tws_grid_sorted if v <= tws)
        hi = min(v for v in tws_grid_sorted if v >= tws)

    def speed_at_tws(tws0: float) -> float:
        row = polar.loc[tws0].dropna()
        if row.empty:
            raise ValueError(f"No data for TWS={tws0}")
        return _interp1(twa, row.index.astype(float).to_list(), row.to_list())

    s_lo = speed_at_tws(lo)
    s_hi = speed_at_tws(hi)
    if hi == lo:
        return float(s_lo)
    return float(_interp1(tws, [lo, hi], [s_lo, s_hi]))


def read_target_speed_xlsx(path: Path) -> pd.DataFrame:
    wb = load_workbook(path, data_only=True)
    ws = wb[wb.sheetnames[0]]

    records: list[dict[str, float | str]] = []
    mode: str | None = None  # "TWA" or "AWA"

    for r in range(1, ws.max_row + 1):
        a = ws.cell(r, 1).value
        b = ws.cell(r, 2).value
        c = ws.cell(r, 3).value
        d = ws.cell(r, 4).value
        e = ws.cell(r, 5).value

        if a == "TWS" and b in {"TWA", "AWA"}:
            mode = str(b)
            continue
        if mode is None:
            continue
        if a is None:
            continue
        try:
            tws = float(a)
            ang1 = float(b)
            bs1 = float(c)
            ang2 = float(d)
            bs2 = float(e)
        except Exception:
            continue

        records.append({"mode": mode, "side": "upwind", "tws": tws, "angle": ang1, "bs": bs1})
        records.append({"mode": mode, "side": "downwind", "tws": tws, "angle": ang2, "bs": bs2})

    if not records:
        raise ValueError(f"No target-speed rows found in {path}")
    return pd.DataFrame.from_records(records)


def write_report(
    out_dir: Path,
    builder_polar: pd.DataFrame,
    latest_polar: pd.DataFrame,
    target: pd.DataFrame,
) -> None:
    rows = []
    for _, t in target.iterrows():
        mode = str(t["mode"])
        angle = float(t["angle"])
        tws = float(t["tws"])
        tgt = float(t["bs"])

        if mode == "AWA":
            # Placeholder: needs TWA/AWA conversion (true wind vs apparent).
            # For now, skip AWA validation to avoid misleading results.
            continue

        b = polar_speed(builder_polar, tws=tws, twa=angle)
        l = polar_speed(latest_polar, tws=tws, twa=angle)
        rows.append(
            {
                "side": str(t["side"]),
                "tws": tws,
                "twa": angle,
                "target_bs": tgt,
                "builder_bs": b,
                "latest_bs": l,
                "builder_err": b - tgt,
                "latest_err": l - tgt,
            }
        )

    df = pd.DataFrame(rows).sort_values(["side", "tws", "twa"])
    df.to_csv(out_dir / "target_vs_polar.csv", index=False)

    lines = []
    lines.append("## Expedition データ解析レポート（自動生成）")
    lines.append("")
    lines.append(f"- builder polar: `{builder_polar.shape[0]} TWS × {builder_polar.shape[1]} TWA`")
    lines.append(f"- latest polar: `{latest_polar.shape[0]} TWS × {latest_polar.shape[1]} TWA`")
    lines.append("")
    if df.empty:
        lines.append("- Target Speed.xlsx の AWA ブロックは現状未評価（TWA/AWA 変換が必要）")
    else:
        lines.append("### Target speed vs Polar（TWAブロックのみ）")
        lines.append("")
        try:
            lines.append(df.to_markdown(index=False))
        except Exception:
            # Avoid hard dependency on tabulate at runtime.
            lines.append("```")
            lines.append(df.to_csv(index=False).strip())
            lines.append("```")
        lines.append("")
        lines.append(
            f"- builder mean abs error: `{df['builder_err'].abs().mean():.3f} kn` / latest mean abs error: `{df['latest_err'].abs().mean():.3f} kn`"
        )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    repo_dir = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description="Expedition polar / sailchart analysis")
    ap.add_argument("--builder-polar", type=Path, default=repo_dir / "Builder Polar ZX.txt")
    ap.add_argument("--latest-polar", type=Path, default=repo_dir / "ZX porlar latest 0823.txt")
    ap.add_argument("--heel", type=Path, default=repo_dir / "Builder Heel ZX.txt")
    ap.add_argument("--sailchart", type=Path, default=repo_dir / "Sailchart North.csv")
    ap.add_argument("--crossover-xml", type=Path, default=repo_dir / "CrossoverSail.xml")
    ap.add_argument("--target-speed", type=Path, default=repo_dir / "Target Speed.xlsx")
    ap.add_argument("--out", type=Path, default=repo_dir / "out")
    args = ap.parse_args()

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    _, builder = read_expedition_polar(args.builder_polar)
    _, latest = read_expedition_polar(args.latest_polar)
    heel = read_expedition_heel_table(args.heel)
    sailchart = read_sailchart_csv(args.sailchart)
    crossover = read_crossover_xml(args.crossover_xml)
    target = read_target_speed_xlsx(args.target_speed)

    # Save normalized tables
    builder.to_csv(out_dir / "polar_builder_wide.csv")
    latest.to_csv(out_dir / "polar_latest_wide.csv")
    heel.to_csv(out_dir / "heel_wide.csv")
    sailchart.to_csv(out_dir / "sailchart.csv")
    crossover.to_csv(out_dir / "crossover_points.csv", index=False)
    target.to_csv(out_dir / "target_speed_long.csv", index=False)

    # Compare builder vs latest on builder grid
    diff = builder.copy()
    for tws in diff.index.astype(float):
        for twa in diff.columns.astype(float):
            b = polar_speed(builder, tws=tws, twa=twa)
            l = polar_speed(latest, tws=tws, twa=twa)
            diff.loc[tws, twa] = l - b
    diff.to_csv(out_dir / "polar_latest_minus_builder.csv")

    # Plot: diff heatmap
    plt.figure(figsize=(12, 5))
    z = diff.to_numpy(dtype=float)
    plt.imshow(
        z,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[
            float(diff.columns.min()),
            float(diff.columns.max()),
            float(diff.index.min()),
            float(diff.index.max()),
        ],
    )
    plt.colorbar(label="Latest - Builder (kn)")
    plt.xlabel("TWA (deg)")
    plt.ylabel("TWS (kn)")
    plt.title("Polar difference heatmap")
    plt.tight_layout()
    plt.savefig(out_dir / "plots" / "polar_diff_heatmap.png", dpi=160)
    plt.close()

    # Plot: sample curve at closest TWS=12
    tws_target = 12.0
    tws_choices = [float(x) for x in builder.index]
    tws_pick = min(tws_choices, key=lambda v: abs(v - tws_target))
    twa_grid = [float(x) for x in builder.columns]
    b_curve = [polar_speed(builder, tws=tws_pick, twa=twa) for twa in twa_grid]
    l_curve = [polar_speed(latest, tws=tws_pick, twa=twa) for twa in twa_grid]

    plt.figure(figsize=(10, 4))
    plt.plot(twa_grid, b_curve, label=f"Builder (TWS={tws_pick:g})")
    plt.plot(twa_grid, l_curve, label=f"Latest (interp)")
    plt.xlabel("TWA (deg)")
    plt.ylabel("Boat speed (kn)")
    plt.title("Polar curve comparison")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "plots" / "polar_curve_tws12.png", dpi=160)
    plt.close()

    write_report(out_dir, builder_polar=builder, latest_polar=latest, target=target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

