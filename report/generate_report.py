#!/usr/bin/env python3
"""builds the validation report from screenshots in report/screenshots/.

usage:
    python3 report/generate_report.py

screenshots are grouped into sections by filename prefix (parta_, partb_, ...
partf_), sorted alphabetically within a section. anything without a prefix
goes into an extra section at the end. captions come from the filename, or
from report/captions.txt (lines like "file.png: my caption").

outputs report/report.html (self contained, images embedded) and
report/report.md. git branch and merge history are appended if run inside
the repo. stdlib only.
"""

import base64
import datetime
import html
import mimetypes
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
SCREENSHOT_DIR = HERE / "screenshots"
CAPTIONS_FILE = HERE / "captions.txt"
OUT_HTML = HERE / "report.html"
OUT_MD = HERE / "report.md"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}

SECTIONS = {
    "parta": (1, "Part A - Repository Setup & Git Workflow",
              "repository structure, branch strategy, merged pull requests"),
    "partb": (2, "Part B - PyTorch Model",
              "model, dataset, training loop with json logging and early stopping, serving app"),
    "partc": (3, "Part C - Docker Containerization",
              "multi stage training image, hardened serving image, local verification"),
    "partd": (4, "Part D - Kubernetes Training Job",
              "namespace, configmap, pvcs, training job with resource limits"),
    "parte": (5, "Part E - Kubernetes Model Serving",
              "2 replica deployment with probes, rolling updates, service"),
    "partf": (6, "Part F - End-to-End Validation",
              "full pipeline applied on the cluster, prediction endpoint tested via port-forward"),
}
EXTRA_SECTION = (99, "Additional Evidence", "screenshots without a part prefix")


def load_captions():
    captions = {}
    if CAPTIONS_FILE.exists():
        for line in CAPTIONS_FILE.read_text(encoding="utf-8").splitlines():
            if ":" in line and not line.lstrip().startswith("#"):
                name, cap = line.split(":", 1)
                captions[name.strip()] = cap.strip()
    return captions


def derive_caption(filename):
    stem = Path(filename).stem
    stem = re.sub(r"^part[a-f]_?", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"^\d+_?", "", stem)
    words = stem.replace("-", " ").replace("_", " ").strip()
    return words.capitalize() if words else Path(filename).stem


def collect_screenshots():
    grouped = {}
    if not SCREENSHOT_DIR.exists():
        return grouped
    for path in sorted(SCREENSHOT_DIR.iterdir()):
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        match = re.match(r"^(part[a-f])", path.name, flags=re.IGNORECASE)
        key = match.group(1).lower() if match else "extra"
        grouped.setdefault(key, []).append(path)
    return grouped


def git(*args):
    try:
        out = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=15
        )
        return out.stdout.strip()
    except Exception:
        return ""


def embed_image(path):
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def build(grouped, captions):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    commit_log = git("log", "--oneline", "--decorate", "-30")
    merge_log = git("log", "--merges", "--pretty=format:%h %s (%an, %ad)", "--date=short", "-20")
    branch_list = git("branch", "-a", "--format=%(refname:short)")

    total = sum(len(v) for v in grouped.values())

    md = [
        "# MLOps Assignment 2 - Validation Report",
        "",
        "**Project:** `mlops-pytorch-pipeline` - PyTorch training and serving on Docker + Kubernetes",
        f"**Generated:** {now}  |  **Screenshots included:** {total}",
        "",
    ]
    body = []

    keys = list(grouped.keys())
    for key in SECTIONS:
        if key not in keys:
            keys.append(key)
    keys.sort(key=lambda k: SECTIONS.get(k, EXTRA_SECTION)[0])

    for key in keys:
        _, title, blurb = SECTIONS.get(key, EXTRA_SECTION)
        images = grouped.get(key, [])
        md += [f"## {title}", "", blurb, ""]
        body.append(f"<section><h2>{html.escape(title)}</h2>"
                    f"<p class='blurb'>{html.escape(blurb)}</p>")
        if not images:
            if key in SECTIONS:
                note = f"no screenshots for this section yet, add files prefixed {key}_ to report/screenshots/"
                md += [f"_{note}_", ""]
                body.append(f"<p class='missing'>{html.escape(note)}</p>")
        for img in images:
            caption = captions.get(img.name, derive_caption(img.name))
            md += [f"![{caption}](screenshots/{img.name})", "", f"*Figure: {caption}*", ""]
            body.append(
                "<figure>"
                f"<img src='{embed_image(img)}' alt='{html.escape(caption)}'/>"
                f"<figcaption>{html.escape(caption)}</figcaption>"
                "</figure>"
            )
        body.append("</section>")

    if commit_log or merge_log:
        md += ["## Git Evidence", ""]
        body.append("<section><h2>Git Evidence</h2>")
        if branch_list:
            md += ["**Branches:**", "```", branch_list, "```", ""]
            body.append(f"<h3>Branches</h3><pre>{html.escape(branch_list)}</pre>")
        if merge_log:
            md += ["**Merged PRs / merge commits:**", "```", merge_log, "```", ""]
            body.append(f"<h3>Merged PRs</h3><pre>{html.escape(merge_log)}</pre>")
        if commit_log:
            md += ["**Recent commits:**", "```", commit_log, "```", ""]
            body.append(f"<h3>Recent commits</h3><pre>{html.escape(commit_log)}</pre>")
        body.append("</section>")

    html_doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MLOps Assignment 2 - Validation Report</title>
<style>
  :root {{ --ink:#1b2230; --sub:#5a6478; --line:#e3e7ef; --accent:#2d5bd7; }}
  body {{ font-family: Georgia, serif; color: var(--ink); max-width: 880px;
         margin: 0 auto; padding: 40px 24px; line-height: 1.55; }}
  header {{ border-bottom: 3px solid var(--accent); padding-bottom: 14px; margin-bottom: 28px; }}
  h1 {{ font-size: 1.7rem; margin: 0 0 6px; }}
  .meta {{ color: var(--sub); font-size: .92rem; }}
  h2 {{ font-size: 1.2rem; margin-top: 40px; border-bottom: 1px solid var(--line);
        padding-bottom: 6px; color: var(--accent); }}
  .blurb {{ color: var(--sub); font-style: italic; margin-top: 4px; }}
  .missing {{ color: #a05a00; background: #fff7e8; padding: 8px 12px; border-radius: 6px; }}
  figure {{ margin: 22px 0; }}
  figure img {{ max-width: 100%; border: 1px solid var(--line); border-radius: 8px; }}
  figcaption {{ color: var(--sub); font-size: .88rem; margin-top: 6px; }}
  pre {{ background: #f6f8fb; border: 1px solid var(--line); border-radius: 8px;
        padding: 12px; overflow-x: auto; font-size: .82rem; }}
</style></head><body>
<header>
  <h1>MLOps Assignment 2 - Validation Report</h1>
  <div class="meta">Project: <code>mlops-pytorch-pipeline</code> | PyTorch on Docker + Kubernetes<br>
  Generated {now} | {total} screenshot(s) embedded</div>
</header>
{''.join(body)}
</body></html>"""

    return html_doc, "\n".join(md)


def main():
    grouped = collect_screenshots()
    total = sum(len(v) for v in grouped.values())
    if total == 0:
        print(f"[warn] no screenshots found in {SCREENSHOT_DIR}/")
    html_doc, md_doc = build(grouped, load_captions())
    OUT_HTML.write_text(html_doc, encoding="utf-8")
    OUT_MD.write_text(md_doc, encoding="utf-8")
    print(f"[ok] {total} screenshot(s) picked up")
    print(f"[ok] wrote {OUT_HTML}")
    print(f"[ok] wrote {OUT_MD}")


if __name__ == "__main__":
    main()
