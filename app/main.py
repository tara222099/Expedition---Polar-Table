from __future__ import annotations

import io
import json
import os
import uuid
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates


APP_TITLE = "Expedition Log Analyzer (TWA correction)"

app = FastAPI(title=APP_TITLE)
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))


# In-memory store (ok for MVP; replace with redis/sqlite later)
_ANALYSES: dict[str, dict[str, Any]] = {}


def _normalize_colname(s: str) -> str:
    return (
        str(s)
        .strip()
        .lower()
        .replace("°", "deg")
        .replace("(", " ")
        .replace(")", " ")
        .replace("/", " ")
        .replace("-", " ")
        .replace("_", " ")
    )


def read_expedition_log(file_bytes: bytes) -> pd.DataFrame:
    """
    Expedition exports are often CSV but delimiter varies (comma/semicolon/tab).
    We try a small set and fall back to pandas auto.
    """
    buf = io.BytesIO(file_bytes)
    # Try common separators. Keep engine="python" for robust parsing.
    for sep in [",", ";", "\t"]:
        buf.seek(0)
        try:
            df = pd.read_csv(buf, sep=sep, engine="python")
        except Exception:
            continue
        if df.shape[1] >= 3:
            return df

    buf.seek(0)
    return pd.read_csv(buf, sep=None, engine="python")


def suggest_column_map(df: pd.DataFrame) -> dict[str, str | None]:
    """
    Suggest likely columns based on normalized names.
    """
    norm_to_orig: dict[str, str] = {_normalize_colname(c): str(c) for c in df.columns}

    def pick(*needles: str) -> str | None:
        for n in needles:
            for norm, orig in norm_to_orig.items():
                if n in norm:
                    return orig
        return None

    return {
        "twa_deg": pick("twa", "true wind angle", "truewindangle"),
        "tws": pick("tws", "true wind speed", "truewindspeed"),
        "bsp": pick("bsp", "boat speed", "boatspeed", "stw", "speed through water"),
        "awa_deg": pick("awa", "apparent wind angle", "apparentwindangle"),
    }


def _to_numeric(series: pd.Series) -> pd.Series:
    # Coerce commas as decimal separators, strip units etc.
    s = series.astype(str).str.replace(",", ".", regex=False)
    s = s.str.replace(r"[^0-9eE\+\-\.]+", "", regex=True)
    return pd.to_numeric(s, errors="coerce")


def _ensure_signed_twa(twa: pd.Series) -> pd.Series:
    """
    If TWA is 0..360, convert to signed -180..180.
    If TWA is already -180..180, keep.
    """
    x = twa.copy()
    x = ((x + 180.0) % 360.0) - 180.0
    # handle edge-case of 180 -> -180 mapping; keep 180 positive
    x = x.where(x != -180.0, 180.0)
    return x


def estimate_twa_offset_correction(
    df: pd.DataFrame,
    col_twa: str,
    col_sign_from_awa: str | None = None,
    abs_twa_min: float = 25.0,
    abs_twa_max: float = 175.0,
    bin_size_deg: float = 2.0,
    min_points_per_side: int = 50,
) -> pd.DataFrame:
    """
    Symmetry-based estimate for additive bias b(|TWA|) where:
      TWA_meas = TWA_true + b(|TWA_true|)
    For each abs(TWA) bin, we estimate:
      b ~= (median_starboard + median_port) / 2
    and recommend correction = -b (add to measured).
    """
    twa_raw = _to_numeric(df[col_twa])
    twa = _ensure_signed_twa(twa_raw)

    # If TWA looks unsigned (almost all >=0 or <=0), try to sign it using AWA.
    # This helps when the log has TWA as 0..180 and AWA carries port/stb sign.
    if col_sign_from_awa and col_sign_from_awa in df.columns:
        x = twa.dropna()
        if len(x) > 0:
            frac_neg = float((x < 0).mean())
            frac_pos = float((x > 0).mean())
            if frac_neg < 0.01 or frac_pos < 0.01:
                awa = _to_numeric(df[col_sign_from_awa])
                sign = pd.Series(np.sign(awa)).replace({0.0: np.nan})
                twa = twa.abs() * sign
    work = pd.DataFrame({"twa": twa}).dropna()
    work["abs_twa"] = work["twa"].abs()
    work = work[(work["abs_twa"] >= abs_twa_min) & (work["abs_twa"] <= abs_twa_max)]
    if work.empty:
        return pd.DataFrame(columns=["abs_twa_bin_deg", "bias_deg", "correction_deg", "n_port", "n_starboard"])

    # Bin by abs TWA.
    bins = np.arange(abs_twa_min, abs_twa_max + bin_size_deg, bin_size_deg)
    if len(bins) < 2:
        raise ValueError("Invalid bins; adjust abs_twa_min/max/bin_size_deg")

    # Labels as bin centers.
    centers = (bins[:-1] + bins[1:]) / 2.0
    work["bin"] = pd.cut(work["abs_twa"], bins=bins, labels=centers, include_lowest=True)

    out_rows: list[dict[str, Any]] = []
    for center, g in work.groupby("bin", observed=True):
        if center is pd.NA:
            continue
        port = g.loc[g["twa"] < 0, "twa"]
        stb = g.loc[g["twa"] > 0, "twa"]
        if len(port) < min_points_per_side or len(stb) < min_points_per_side:
            continue
        med_port = float(np.nanmedian(port))
        med_stb = float(np.nanmedian(stb))
        bias = (med_stb + med_port) / 2.0
        out_rows.append(
            {
                "abs_twa_bin_deg": float(center),
                "bias_deg": bias,
                "correction_deg": -bias,
                "n_port": int(len(port)),
                "n_starboard": int(len(stb)),
            }
        )

    return pd.DataFrame(out_rows).sort_values("abs_twa_bin_deg").reset_index(drop=True)


def plot_correction_table(table: pd.DataFrame) -> str:
    fig = go.Figure()
    if not table.empty:
        fig.add_trace(
            go.Scatter(
                x=table["abs_twa_bin_deg"],
                y=table["correction_deg"],
                mode="lines+markers",
                name="TWA correction (deg)",
            )
        )
        fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="#888")
    fig.update_layout(
        title="Estimated TWA correction vs |TWA| (symmetry-based)",
        xaxis_title="|TWA| bin center (deg)",
        yaxis_title="Correction to ADD (deg)",
        template="plotly_white",
        height=420,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "app_title": APP_TITLE,
        },
    )


@app.post("/upload", response_class=HTMLResponse)
async def upload(request: Request, log_file: UploadFile = File(...)) -> HTMLResponse:
    raw = await log_file.read()
    df = read_expedition_log(raw)
    suggestions = suggest_column_map(df)
    analysis_id = uuid.uuid4().hex

    # Store df as JSON (split) to avoid extra parquet deps
    df_json = df.to_json(orient="split")
    _ANALYSES[analysis_id] = {
        "filename": log_file.filename,
        "df_json": df_json,
        "columns": [str(c) for c in df.columns],
        "suggestions": suggestions,
    }

    return templates.TemplateResponse(
        "map_columns.html",
        {
            "request": request,
            "app_title": APP_TITLE,
            "analysis_id": analysis_id,
            "filename": log_file.filename,
            "columns": [str(c) for c in df.columns],
            "suggestions": suggestions,
        },
    )


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(
    request: Request,
    analysis_id: str = Form(...),
    col_twa: str = Form(...),
    col_awa: str = Form(""),
    abs_twa_min: float = Form(25.0),
    abs_twa_max: float = Form(175.0),
    bin_size_deg: float = Form(2.0),
    min_points_per_side: int = Form(50),
) -> HTMLResponse:
    item = _ANALYSES.get(analysis_id)
    if not item:
        return HTMLResponse("Analysis ID not found. Please upload again.", status_code=404)

    df = pd.read_json(io.StringIO(item["df_json"]), orient="split")

    table = estimate_twa_offset_correction(
        df=df,
        col_twa=col_twa,
        col_sign_from_awa=(col_awa or None),
        abs_twa_min=abs_twa_min,
        abs_twa_max=abs_twa_max,
        bin_size_deg=bin_size_deg,
        min_points_per_side=min_points_per_side,
    )

    plot_html = plot_correction_table(table)
    item["result"] = {
        "col_twa": col_twa,
        "params": {
            "abs_twa_min": abs_twa_min,
            "abs_twa_max": abs_twa_max,
            "bin_size_deg": bin_size_deg,
            "min_points_per_side": min_points_per_side,
        },
        "table_csv": table.to_csv(index=False),
        "table_records": json.loads(table.to_json(orient="records")),
        "plot_html": plot_html,
        "n_rows": int(df.shape[0]),
    }

    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "app_title": APP_TITLE,
            "analysis_id": analysis_id,
            "filename": item["filename"],
            "n_rows": int(df.shape[0]),
            "col_twa": col_twa,
            "params": item["result"]["params"],
            "table": item["result"]["table_records"],
            "plot_html": plot_html,
            "table_empty": bool(table.empty),
        },
    )


@app.get("/download/{analysis_id}/twa_correction.csv")
async def download_twa_correction_csv(analysis_id: str) -> Response:
    item = _ANALYSES.get(analysis_id)
    if not item or "result" not in item:
        return Response("Not found", status_code=404)
    csv_data = item["result"]["table_csv"]
    filename = "twa_correction.csv"
    return Response(
        content=csv_data,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )

