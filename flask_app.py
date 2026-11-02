from io import BytesIO
from pathlib import Path

import pandas as pd
from flask import Flask, Response, redirect, render_template_string, request, url_for


app = Flask(__name__)
DATA_DIR = Path(__file__).parent / "data"
current_data = pd.DataFrame()
current_name = ""

PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Business Data Analyzer</title>
  <style>
    :root { color-scheme: light; font-family: system-ui, sans-serif; color: #17212b; background: #f5f7f4; }
    body { margin: 0; }
    main { max-width: 1180px; margin: 0 auto; padding: 38px 22px 56px; }
    header { border-bottom: 1px solid #dfe7e2; margin-bottom: 24px; padding-bottom: 20px; }
    .kicker { color: #0f766e; font-size: .75rem; font-weight: 800; letter-spacing: .14em; }
    h1 { margin: 7px 0; font-size: clamp(2rem, 5vw, 3.2rem); letter-spacing: -.04em; }
    .muted { color: #52615d; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }
    .card { background: white; border: 1px solid #dfe7e2; border-radius: 12px; padding: 18px; box-shadow: 0 6px 20px #17212b0d; }
    .metric { font-size: 1.8rem; font-weight: 750; margin-top: 5px; }
    section { margin-top: 22px; }
    form { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
    input[type=file], button, a.button { border: 1px solid #b9c9c4; border-radius: 8px; padding: 10px 13px; background: #eef1ee; color: #17212b; font: inherit; text-decoration: none; }
    button { cursor: pointer; background: #0f766e; border-color: #0f766e; color: white; font-weight: 700; }
    table { width: 100%; border-collapse: collapse; background: white; font-size: .9rem; }
    th, td { padding: 9px 10px; border-bottom: 1px solid #e6ece8; text-align: left; white-space: nowrap; }
    th { background: #eef3ef; }
    .table-wrap { overflow-x: auto; }
    .notice { padding: 12px 14px; border-radius: 8px; background: #e7f3ef; color: #14594f; }
  </style>
</head>
<body>
<main>
  <header>
    <div class="kicker">BUSINESS INTELLIGENCE WORKSPACE</div>
    <h1>Business Data Analyzer</h1>
    <div class="muted">A direct Python web app powered by Flask, Pandas, and Plotly-compatible data.</div>
  </header>
  {% if message %}<div class="notice">{{ message }}</div>{% endif %}
  <section class="card">
    <form method="post" action="{{ url_for('upload') }}" enctype="multipart/form-data">
      <input type="file" name="dataset" accept=".csv,.xlsx,.xls" required>
      <button type="submit">Load dataset</button>
      <a class="button" href="{{ url_for('download') }}">Download CSV</a>
    </form>
    <p class="muted">Current dataset: <strong>{{ name }}</strong></p>
  </section>
  {% if has_data %}
  <section class="grid">
    <div class="card"><div class="muted">Rows</div><div class="metric">{{ rows }}</div></div>
    <div class="card"><div class="muted">Columns</div><div class="metric">{{ columns }}</div></div>
    <div class="card"><div class="muted">Missing cells</div><div class="metric">{{ missing }}</div></div>
    <div class="card"><div class="muted">Duplicate rows</div><div class="metric">{{ duplicates }}</div></div>
  </section>
  <section class="card">
    <h2>Data preview</h2>
    <div class="table-wrap">{{ preview|safe }}</div>
  </section>
  <section class="card">
    <h2>Column profile</h2>
    <div class="table-wrap">{{ profile|safe }}</div>
  </section>
  <section class="card">
    <h2>Automatic insights</h2>
    <ul>{% for insight in insights %}<li>{{ insight }}</li>{% endfor %}</ul>
  </section>
  {% endif %}
</main>
</body>
</html>
"""


def profile(dataframe):
    return pd.DataFrame(
        [
            {
                "Column": column,
                "Type": str(dataframe[column].dtype),
                "Missing": int(dataframe[column].isna().sum()),
                "Missing %": round(dataframe[column].isna().mean() * 100, 2),
                "Unique": int(dataframe[column].nunique()),
            }
            for column in dataframe.columns
        ]
    )


def insights(dataframe):
    result = [
        f"Dataset has {len(dataframe):,} rows and {len(dataframe.columns)} columns.",
        f"Missing cells: {int(dataframe.isna().sum().sum()):,}.",
        f"Duplicate rows: {int(dataframe.duplicated().sum()):,}.",
    ]
    numeric = dataframe.select_dtypes(include="number")
    for column in numeric.columns[:3]:
        result.append(f"Average {column}: {numeric[column].mean():,.2f}.")
    return result


def read_dataset(file_storage):
    filename = file_storage.filename.lower()
    if filename.endswith(".csv"):
        return pd.read_csv(file_storage)
    return pd.read_excel(file_storage)


def load_sample():
    global current_data, current_name
    sample = DATA_DIR / "sample_sales.csv"
    if sample.exists():
        current_data = pd.read_csv(sample)
        current_name = sample.name


@app.get("/")
def index():
    data = current_data
    return render_template_string(
        PAGE,
        name=current_name or "No dataset loaded",
        has_data=not data.empty,
        rows=f"{len(data):,}",
        columns=len(data.columns),
        missing=f"{int(data.isna().sum().sum()):,}",
        duplicates=f"{int(data.duplicated().sum()):,}",
        preview=data.head(100).to_html(index=False, classes="dataframe") if not data.empty else "",
        profile=profile(data).to_html(index=False) if not data.empty else "",
        insights=insights(data) if not data.empty else [],
        message=request.args.get("message", ""),
    )


@app.post("/upload")
def upload():
    global current_data, current_name
    uploaded = request.files.get("dataset")
    if not uploaded or not uploaded.filename:
        return redirect(url_for("index", message="Choose a CSV or Excel file first."))
    try:
        current_data = read_dataset(uploaded)
        current_name = uploaded.filename
        return redirect(url_for("index", message=f"Loaded {len(current_data):,} rows."))
    except Exception as error:
        return redirect(url_for("index", message=f"Could not load dataset: {error}"))


@app.get("/download")
def download():
    csv_data = current_data.to_csv(index=False).encode()
    return Response(
        BytesIO(csv_data),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=dataset.csv"},
    )


if __name__ == "__main__":
    load_sample()
    app.run(host="127.0.0.1", port=5000, debug=False)