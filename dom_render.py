"""Render a canonical DOM-JSON (the render-state CWM predicts) to PIXELS via headless Chromium (REPORT §35).

This closes the symbolic->pixel half: CWM predicts the DOM tree; a REAL browser rasterizes it. Compare a
PREDICTED DOM's render vs the TRUE DOM's render through the SAME engine (pred==truth => 0 pixel diff). Demo of
the user's "frame as generation": predicted next render-state -> an actual image.
"""
from __future__ import annotations
import argparse, json, html as _html
from playwright.sync_api import sync_playwright

# minimal CSS so the canonical-DOM semantics are visible in pixels
DEFAULT_CSS = """
  body{font:16px system-ui;margin:16px;background:#fff;color:#111}
  [hidden]{display:none}
  button{margin:2px;padding:4px 10px;border:1px solid #aaa;border-radius:6px;background:#f4f4f4}
  [aria-selected="true"]{font-weight:700;border-bottom:3px solid #2563eb;background:#e8efff}
  [aria-expanded="true"]{background:#e8efff}
  [data-done="true"]{text-decoration:line-through;color:#999}
  li{list-style:none;padding:2px 6px}
  input[disabled]{background:#eee}
  .err,[class~="err"]{color:#c00;font-size:13px}
  span{margin-right:8px}
"""


def to_html(node) -> str:
    if not isinstance(node, dict):
        return _html.escape(str(node))
    tag = node.get("tag", "div")
    parts = []
    if node.get("id"):
        parts.append(f'id="{_html.escape(str(node["id"]))}"')
    cls = node.get("class")
    if cls:
        parts.append(f'class="{_html.escape(" ".join(cls) if isinstance(cls, list) else str(cls))}"')
    for k, v in (node.get("attrs") or {}).items():
        if v is True:
            parts.append(_html.escape(str(k)))
        elif v is False or v is None:
            continue
        else:
            parts.append(f'{_html.escape(str(k))}="{_html.escape(str(v))}"')
    attr = (" " + " ".join(parts)) if parts else ""
    inner = _html.escape(str(node.get("text", "")))
    for ch in node.get("children", []) or []:
        inner += to_html(ch)
    void = tag in ("input", "br", "img", "hr")
    return f"<{tag}{attr}>" if void else f"<{tag}{attr}>{inner}</{tag}>"


def dom_to_page(dom, css=DEFAULT_CSS) -> str:
    return f"<!doctype html><html><head><meta charset='utf-8'><style>{css}</style></head><body>{to_html(dom)}</body></html>"


def render_many(items, out_dir, css=DEFAULT_CSS, width=420, height=240):
    """items: list of (name, dom_json). Writes out_dir/<name>.png. Returns list of paths."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--force-color-profile=srgb", "--hide-scrollbars"])
        page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=1)
        for name, dom in items:
            page.set_content(dom_to_page(dom, css), wait_until="load")
            path = os.path.join(out_dir, f"{name}.png")
            page.screenshot(path=path)
            paths.append(path)
        browser.close()
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/uitrans_uidom.jsonl")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out_dir", default="results/render_demo")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.data) if l.strip()][: a.n]
    items = []
    for i, r in enumerate(rows):
        items.append((f"{i}_{r.get('app','app')}_before", r["state_before"]))
        items.append((f"{i}_{r.get('app','app')}_after_TRUE", r["truth_state"]))
    paths = render_many(items, a.out_dir)
    print(f"rendered {len(paths)} PNGs -> {a.out_dir}")
    for pth in paths:
        print("  ", pth)


if __name__ == "__main__":
    main()
