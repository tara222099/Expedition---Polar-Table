from __future__ import annotations

from flask import Flask, render_template, request

from .expedition_csv import CsvFormatError, read_samples_from_csv_bytes
from .expedition_detector import detect_tack_jibe_events


app = Flask(__name__)


@app.get("/")
def index_get():
    return render_template("index.html", rows=None, error=None)


@app.post("/")
def index_post():
    up = request.files.get("file")
    if up is None or not up.filename:
        return render_template("index.html", rows=None, error="CSVファイルを選択してください。")

    try:
        data = up.read()
        samples = read_samples_from_csv_bytes(data)
        events = detect_tack_jibe_events(samples)
        rows = [
            {
                "time_jst": e.time_jst.strftime("%Y-%m-%d %H:%M:%S"),
                "type": e.type,
                "twa_before": f"{e.twa_before:.1f}",
                "twa_after": f"{e.twa_after:.1f}",
            }
            for e in events
        ]
        return render_template("index.html", rows=rows, error=None)
    except CsvFormatError as e:
        return render_template("index.html", rows=None, error=str(e))
    except Exception as e:
        return render_template("index.html", rows=None, error=f"処理に失敗しました: {e}")


if __name__ == "__main__":
    # Minimal dev server
    app.run(host="0.0.0.0", port=8000, debug=True)

