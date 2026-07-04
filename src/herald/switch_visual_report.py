# ruff: noqa: E501
"""Render a visual HTML report for switch-level EDA results."""

from html import escape
from typing import Any


def render_visual_report(summary: dict[str, Any]) -> str:
    """Return a self-contained HTML report from EDA summary data."""
    inv = summary["inventory"]
    labels = summary["labels"]["overall"]
    cell = summary["cell_structure"]
    grouped = summary["labels"]["by_task_compressor_ratio"]
    raw_corr = summary["feature_vs_damage"]["top_corr_dq"][:12]
    resid_corr = summary["feature_vs_damage"]["top_corr_dq_resid_cell"][:12]

    task_bars = _bar_rows(inv["by_task"], max_width=520)
    label_stack = _damage_stack(labels)
    r2_bars = _metric_bars(
        {
            "task + compressor + ratio + s": cell[
                "r2_task_compressor_ratio_s"
            ],
            "compressor + ratio + s": cell["r2_compressor_ratio_s"],
            "task + compressor + ratio": cell["r2_task_compressor_ratio"],
        }
    )
    heatmap = _heatmap(grouped)
    raw_chart = _corr_chart(raw_corr)
    resid_chart = _corr_chart(resid_corr)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>HERALD Switch Dataset EDA</title>
  <style>
    :root {{
      --ink: #101418;
      --muted: #5B6672;
      --paper: #F4F1EA;
      --panel: #FFFFFF;
      --line: #D7D0C4;
      --steel: #315A70;
      --steel-soft: #D6E4EA;
      --rust: #B64A2F;
      --amber: #D49D2A;
      --green: #2F7D62;
      --black: #07090B;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background:
        linear-gradient(90deg, rgba(16,20,24,.035) 1px, transparent 1px),
        linear-gradient(0deg, rgba(16,20,24,.035) 1px, transparent 1px),
        var(--paper);
      background-size: 22px 22px;
      font-family: Charter, "Iowan Old Style", Georgia, serif;
    }}
    main {{
      width: min(1180px, calc(100vw - 32px));
      margin: 28px auto 56px;
    }}
    .hero {{
      display: grid;
      grid-template-columns: 1.2fr .8fr;
      gap: 24px;
      align-items: stretch;
      border-top: 6px solid var(--black);
      padding-top: 22px;
    }}
    h1 {{
      font-family: "Avenir Next Condensed", "DIN Condensed", Impact, sans-serif;
      font-size: clamp(48px, 8vw, 112px);
      line-height: .86;
      letter-spacing: 0;
      margin: 0;
      text-transform: uppercase;
    }}
    h2 {{
      font-family: "Avenir Next Condensed", "DIN Condensed", Impact, sans-serif;
      font-size: 34px;
      margin: 0 0 14px;
      letter-spacing: 0;
      text-transform: uppercase;
    }}
    h3 {{
      font-family: "Avenir Next Condensed", "DIN Condensed", Impact, sans-serif;
      font-size: 22px;
      margin: 0 0 10px;
      letter-spacing: 0;
      text-transform: uppercase;
    }}
    p {{
      font-size: 18px;
      line-height: 1.45;
      margin: 0;
    }}
    .lede {{
      max-width: 760px;
      margin-top: 18px;
      color: var(--muted);
    }}
    .stamp {{
      background: var(--black);
      color: #F7F3EA;
      padding: 20px;
      display: grid;
      align-content: space-between;
      min-height: 240px;
    }}
    .stamp strong {{
      font-family: "Avenir Next Condensed", "DIN Condensed", Impact, sans-serif;
      display: block;
      font-size: 56px;
      line-height: .9;
      letter-spacing: 0;
    }}
    .stamp span {{
      color: #C7BDAA;
      font-size: 15px;
      text-transform: uppercase;
      letter-spacing: .08em;
    }}
    .kpis {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin: 26px 0;
    }}
    .kpi, .panel {{
      background: rgba(255,255,255,.82);
      border: 1px solid var(--line);
      box-shadow: 0 16px 30px rgba(20,18,12,.08);
    }}
    .kpi {{
      padding: 16px;
      min-height: 112px;
      display: grid;
      align-content: space-between;
    }}
    .kpi span {{
      color: var(--muted);
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: .08em;
    }}
    .kpi strong {{
      font-family: "Avenir Next Condensed", "DIN Condensed", Impact, sans-serif;
      font-size: 42px;
      line-height: 1;
      letter-spacing: 0;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
      margin-top: 18px;
    }}
    .wide {{ grid-column: 1 / -1; }}
    .panel {{ padding: 20px; }}
    .note {{
      border-left: 5px solid var(--rust);
      padding: 12px 14px;
      background: #FFF8EA;
      color: #3B332A;
      margin-top: 14px;
    }}
    .bars {{ display: grid; gap: 10px; }}
    .bar-row {{
      display: grid;
      grid-template-columns: 150px 1fr 88px;
      gap: 10px;
      align-items: center;
      font-size: 15px;
    }}
    .track {{
      height: 16px;
      background: #E9E1D2;
      border: 1px solid var(--line);
      position: relative;
      overflow: hidden;
    }}
    .fill {{
      height: 100%;
      background: var(--steel);
    }}
    .stack {{
      height: 34px;
      display: flex;
      border: 1px solid var(--line);
      overflow: hidden;
      margin-top: 12px;
    }}
    .seg {{ height: 100%; }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-top: 12px;
      color: var(--muted);
      font-size: 15px;
    }}
    .chip {{
      display: inline-flex;
      align-items: center;
      gap: 7px;
    }}
    .swatch {{
      width: 14px;
      height: 14px;
      border: 1px solid rgba(0,0,0,.15);
    }}
    .heatmap {{
      display: grid;
      grid-template-columns: 170px repeat(4, 1fr);
      gap: 4px;
      align-items: stretch;
      font-size: 14px;
    }}
    .hm-cell, .hm-head, .hm-label {{
      min-height: 42px;
      padding: 8px;
      border: 1px solid rgba(16,20,24,.12);
    }}
    .hm-head {{
      background: var(--black);
      color: #F7F3EA;
      font-family: "Avenir Next Condensed", "DIN Condensed", Impact, sans-serif;
      font-size: 18px;
      text-transform: uppercase;
    }}
    .hm-label {{
      background: #EAE2D4;
      font-weight: 700;
    }}
    .hm-cell {{
      color: #101418;
      display: grid;
      align-content: center;
      font-variant-numeric: tabular-nums;
    }}
    .corr {{
      display: grid;
      gap: 8px;
    }}
    .corr-row {{
      display: grid;
      grid-template-columns: minmax(180px, 1fr) 220px 68px;
      gap: 10px;
      align-items: center;
      font-size: 14px;
    }}
    .axis {{
      position: relative;
      height: 14px;
      background: linear-gradient(90deg, #DDBD77 50%, #BBD1DB 50%);
      border: 1px solid var(--line);
    }}
    .axis::after {{
      content: "";
      position: absolute;
      left: 50%;
      top: -4px;
      bottom: -4px;
      width: 1px;
      background: var(--black);
    }}
    .dot {{
      position: absolute;
      top: 50%;
      width: 12px;
      height: 12px;
      border-radius: 50%;
      background: var(--rust);
      border: 2px solid #FFF;
      transform: translate(-50%, -50%);
      box-shadow: 0 1px 5px rgba(0,0,0,.25);
    }}
    .dot.pos {{ background: var(--steel); }}
    .small {{
      color: var(--muted);
      font-size: 14px;
      line-height: 1.35;
    }}
    @media (max-width: 820px) {{
      .hero, .grid, .kpis {{ grid-template-columns: 1fr; }}
      .wide {{ grid-column: auto; }}
      .heatmap {{ grid-template-columns: 130px repeat(4, minmax(64px, 1fr)); }}
      .corr-row {{ grid-template-columns: 1fr; }}
      .bar-row {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <div>
        <h1>Switch Damage Atlas</h1>
        <p class="lede">
          A visual audit of HERALD's switch-level forecasting table.
          Each row asks what happens if compression starts at a specific
          generation position, with damage measured against the reference
          answer quality.
        </p>
      </div>
      <aside class="stamp">
        <span>Counterfactual Forecasting</span>
        <strong>{_fmt_int(inv["n_rows"])} rows</strong>
        <span>Built from completed GSM8K, HumanEval, and current IFEval</span>
      </aside>
    </section>

    <section class="kpis">
      <div class="kpi"><span>Damage Rate</span><strong>{_pct(labels["frac_damage"])}</strong></div>
      <div class="kpi"><span>Zero Damage</span><strong>{_pct(labels["frac_zero"])}</strong></div>
      <div class="kpi"><span>Mean dq</span><strong>{_num(labels["mean"])}</strong></div>
      <div class="kpi"><span>Cell R2</span><strong>{_num(cell["r2_task_compressor_ratio_s"])}</strong></div>
    </section>

    <section class="grid">
      <article class="panel">
        <h2>Coverage</h2>
        <div class="bars">{task_bars}</div>
        <p class="note">
          The table is already large enough for classical analysis:
          completed GSM8K and HumanEval, plus a substantial IFEval snapshot.
        </p>
      </article>
      <article class="panel">
        <h2>Label Shape</h2>
        {label_stack}
        <p class="note">
          Most switches do no measurable harm. The modeling problem is
          sparse but not rare: roughly one third of rows are damaging.
        </p>
      </article>
      <article class="panel wide">
        <h2>Damage Heatmap</h2>
        <p class="small">
          Cell color is mean dq by task, compressor, and ratio.
          Deeper red means more quality loss.
        </p>
        {heatmap}
      </article>
      <article class="panel">
        <h2>How Much Metadata Explains</h2>
        <div class="bars">{r2_bars}</div>
        <p class="note">
          Cell means explain a meaningful share, but most variance remains
          inside cells. That residual is the target for cheap logit features.
        </p>
      </article>
      <article class="panel">
        <h2>Raw Feature Signal</h2>
        {raw_chart}
      </article>
      <article class="panel wide">
        <h2>Residual Feature Signal</h2>
        <p class="small">
          These correlations are after subtracting the mean damage of each
          task, compressor, ratio, and switch-position cell.
        </p>
        {resid_chart}
      </article>
    </section>
  </main>
</body>
</html>
"""


def _bar_rows(values: dict[str, int], *, max_width: int) -> str:
    if not values:
        return ""
    max_value = max(values.values())
    rows: list[str] = []
    for label, value in values.items():
        pct = 100.0 * value / max_value if max_value else 0.0
        rows.append(
            '<div class="bar-row">'
            f"<span>{escape(label)}</span>"
            '<span class="track">'
            f'<span class="fill" style="width:{pct:.2f}%"></span>'
            "</span>"
            f"<strong>{_fmt_int(value)}</strong>"
            "</div>"
        )
    return "\n".join(rows)


def _damage_stack(labels: dict[str, Any]) -> str:
    damage = float(labels["frac_damage"]) * 100.0
    lift = float(labels["frac_lift"]) * 100.0
    zero = float(labels["frac_zero"]) * 100.0
    return f"""
    <div class="stack">
      <div class="seg" style="width:{damage:.4f}%;background:var(--rust)"></div>
      <div class="seg" style="width:{zero:.4f}%;background:#E4DAC8"></div>
      <div class="seg" style="width:{lift:.4f}%;background:var(--green)"></div>
    </div>
    <div class="legend">
      <span class="chip"><i class="swatch" style="background:var(--rust)"></i>Damage {_pct(labels["frac_damage"])}</span>
      <span class="chip"><i class="swatch" style="background:#E4DAC8"></i>Zero {_pct(labels["frac_zero"])}</span>
      <span class="chip"><i class="swatch" style="background:var(--green)"></i>Lift {_pct(labels["frac_lift"])}</span>
    </div>
    """


def _metric_bars(values: dict[str, float | None]) -> str:
    rows: list[str] = []
    for label, value in values.items():
        v = float(value or 0.0)
        rows.append(
            '<div class="bar-row">'
            f"<span>{escape(label)}</span>"
            '<span class="track">'
            f'<span class="fill" style="width:{100.0 * v:.2f}%"></span>'
            "</span>"
            f"<strong>{_num(v)}</strong>"
            "</div>"
        )
    return "\n".join(rows)


def _heatmap(grouped: dict[str, dict[str, Any]]) -> str:
    ratios = ("0.25", "0.5", "0.75", "0.875")
    tasks = sorted({key.split("|")[0] for key in grouped})
    compressors = sorted({key.split("|")[1] for key in grouped})
    lines = ['<div class="heatmap">', '<div class="hm-head">Cell</div>']
    for ratio in ratios:
        lines.append(f'<div class="hm-head">r={ratio}</div>')
    for task in tasks:
        for comp in compressors:
            label = f"{task}<br>{comp}"
            lines.append(f'<div class="hm-label">{label}</div>')
            for ratio in ratios:
                key = f"{task}|{comp}|{ratio}"
                value = grouped.get(key, {}).get("mean")
                lines.append(_heat_cell(value))
    lines.append("</div>")
    return "\n".join(lines)


def _heat_cell(value: object) -> str:
    if value is None:
        return '<div class="hm-cell" style="background:#EEE7DB">NA</div>'
    v = max(-0.1, min(0.9, float(value)))
    norm = (v + 0.1) / 1.0
    hue = 172.0 - 160.0 * norm
    light = 88.0 - 34.0 * norm
    bg = f"hsl({hue:.1f} 55% {light:.1f}%)"
    return (
        f'<div class="hm-cell" style="background:{bg}">'
        f"{_num(float(value))}</div>"
    )


def _corr_chart(rows: list[dict[str, Any]]) -> str:
    out = ['<div class="corr">']
    for row in rows:
        corr = float(row["corr"])
        pos = 50.0 + max(-0.25, min(0.25, corr)) * 200.0
        dot_class = "dot pos" if corr >= 0 else "dot"
        label = escape(str(row["feature"]).replace("feat__", ""))
        out.append(
            '<div class="corr-row">'
            f"<span>{label}</span>"
            '<span class="axis">'
            f'<i class="{dot_class}" style="left:{pos:.2f}%"></i>'
            "</span>"
            f"<strong>{_num(corr)}</strong>"
            "</div>"
        )
    out.append("</div>")
    return "\n".join(out)


def _pct(value: object) -> str:
    return f"{100.0 * float(value):.1f}%"


def _num(value: object) -> str:
    return f"{float(value):.3f}"


def _fmt_int(value: object) -> str:
    return f"{int(value):,}"
