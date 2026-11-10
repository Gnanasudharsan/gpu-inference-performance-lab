"""Render the dashboard's 1200x630 link-preview card from results/summary.json (real data).

Usage: python scripts/make_og_image.py   ->  web/public/og.png  (needs Google Chrome)
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def main() -> None:
    s = json.loads((ROOT / "results/summary.json").read_text())
    comps = [
        c
        for c in s["comparisons"]
        if c["device"] == "mps"
        and c["dtype"] == "float16"
        and c["mode"] == "throughput"
        and c["candidate"] == "metal"
        and c["d_model"] == 1024
        and c["rows"] in (1, 2, 4, 8)
    ]
    lo = min(c["speedup_geomean"] for c in comps)
    hi = max(c["speedup_geomean"] for c in comps)
    lp = s["kernels"]["launches_per_forward"]
    bw = next(
        b["roofline"]["achieved_gbps"]
        for b in s["benchmarks"]
        if b["impl"] == "metal"
        and b["mode"] == "throughput"
        and b["dtype"] == "float16"
        and b["d_model"] == 1024
        and b["rows"] == 8
    )
    tiles = [
        (f"{lo:.2f}–{hi:.2f}×", "fused kernels vs eager, 1–8 rows"),
        (
            f"{lp['eager/r8']['total']:.0f} → {lp['metal/r8']['total']:.0f}",
            "kernel launches per forward",
        ),
        (f"{bw:.0f} GB/s", "weight-streaming bandwidth"),
    ]
    tile_html = "".join(
        f'<div class="t"><div class="v">{val}</div><div class="l">{label}</div></div>'
        for val, label in tiles
    )
    html = f"""<html><head><style>
body{{margin:0;width:1200px;height:630px;background:#0b1220;color:#f3f6fc;
font-family:-apple-system,'Inter',system-ui,sans-serif;display:flex;flex-direction:column;
justify-content:center;padding:0 80px;box-sizing:border-box}}
.e{{font:600 20px ui-monospace,Menlo,monospace;letter-spacing:.18em;color:#36d6e7}}
h1{{font-size:60px;line-height:1.08;margin:18px 0 14px;letter-spacing:-.02em}}
p{{font-size:24px;color:#9aaac0;margin:0 0 42px;max-width:980px;line-height:1.4}}
.r{{display:flex;gap:22px}}
.t{{flex:1;background:#111c2e;border:1px solid #22324d;border-radius:16px;padding:24px 26px}}
.v{{font-size:44px;font-weight:650;letter-spacing:-.02em}}.l{{font-size:19px;color:#9aaac0;margin-top:6px}}
.f{{margin-top:34px;font-size:18px;color:#7d8ca3}}</style></head><body>
<div class="e">GPU INFERENCE PERFORMANCE LAB</div>
<h1>Small-batch GPU inference,<br>measured end to end</h1>
<p>Fused kernels, paired speedups with 95% confidence intervals, and latency-aware serving.</p>
<div class="r">{tile_html}</div>
<div class="f">Measured on Apple M4 (Metal/MPS) · paired, interleaved trials · full provenance</div>
</body></html>"""
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "og.html"
        page.write_text(html)
        out = ROOT / "web/public/og.png"
        subprocess.run(
            [
                CHROME,
                "--headless=new",
                "--disable-gpu",
                "--hide-scrollbars",
                "--window-size=1200,630",
                f"--screenshot={out}",
                page.as_uri(),
            ],
            check=True,
            capture_output=True,
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
