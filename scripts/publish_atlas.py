from __future__ import annotations

import argparse
import shutil
from pathlib import Path


PUBLISH_DIRS = ("cohort", "sessions")


def _redirect_html(target: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="0; url={target}">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CLTR Atlas</title>
</head>
<body>
  <p>Redirecting to the published atlas: <a href="{target}">{target}</a></p>
</body>
</html>
"""


def _rewrite_text(path: Path, replacements: list[tuple[str, str]]) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    for old, new in replacements:
        text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")


def publish_atlas(results_dir: str | Path, docs_atlas_dir: str | Path, target: str = "") -> dict[str, str]:
    results_dir = Path(results_dir).resolve()
    docs_atlas_dir = Path(docs_atlas_dir).resolve()
    reports_dir = results_dir / "reports"
    if not reports_dir.exists():
        raise FileNotFoundError(f"Reports directory not found: {reports_dir}")

    normalized_target = str(target or "").strip().strip("/")
    target_dir = docs_atlas_dir if normalized_target in {"", "."} else docs_atlas_dir / normalized_target
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    for name in PUBLISH_DIRS:
        src = reports_dir / name
        if src.exists():
            shutil.copytree(src, target_dir / name)

    work_root = reports_dir / "work"
    atlas_index_src = work_root / "index.html"
    if not atlas_index_src.exists():
        raise FileNotFoundError(f"Atlas entry HTML not found: {atlas_index_src}")

    atlas_index_target = target_dir / "index.html"
    shutil.copy2(atlas_index_src, atlas_index_target)
    if target_dir == docs_atlas_dir:
        for html_path in (target_dir / "cohort").rglob("*.html"):
            _rewrite_text(html_path, [("../../work/index.html", "../../index.html")])
        for html_path in (target_dir / "sessions").rglob("*.html"):
            _rewrite_text(html_path, [("../../../work/index.html", "../../../index.html")])
    else:
        for html_path in (target_dir / "cohort").rglob("*.html"):
            _rewrite_text(html_path, [("../../work/index.html", "../index.html")])
        for html_path in (target_dir / "sessions").rglob("*.html"):
            _rewrite_text(html_path, [("../../../work/index.html", "../../index.html")])
        docs_atlas_dir.mkdir(parents=True, exist_ok=True)
        (docs_atlas_dir / "index.html").write_text(_redirect_html(f"./{normalized_target}/index.html"), encoding="utf-8")
    return {
        "reports_dir": str(reports_dir),
        "published_dir": str(target_dir),
        "atlas_index": str((target_dir if target_dir == docs_atlas_dir else docs_atlas_dir) / "index.html"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="publish_atlas")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--docs-atlas-dir", type=Path, required=True)
    parser.add_argument("--target", default="")
    args = parser.parse_args()
    publish_atlas(args.results_dir, args.docs_atlas_dir, target=args.target)


if __name__ == "__main__":
    main()
