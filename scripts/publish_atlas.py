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


def _ensure_hide_index_html(path: Path) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    marker = "window.history.replaceState"
    if marker in text:
        return
    snippet = (
        "if(window.location.pathname.endsWith('/index.html')){"
        "const cleanPath=window.location.pathname.slice(0,-'index.html'.length)||'/';"
        "window.history.replaceState({},'',cleanPath+window.location.search+window.location.hash);"
        "}\n"
    )
    if "</script>" in text:
        text = text.replace("</script>", f"{snippet}</script>", 1)
    else:
        text = text.replace("</body>", f"<script>\n{snippet}</script>\n</body>", 1)
    path.write_text(text, encoding="utf-8")


def _normalize_atlas_home_logo(path: Path) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        ".logoMark { width:58px; height:58px; display:block; flex-shrink:0; }",
        ".logoImage { width:58px; height:58px; object-fit:contain; display:block; flex-shrink:0; }",
    )
    anchor_variants = (
        "<a class='logoLink' href='index.html' title='Open report index' aria-label='Open report index'>",
        "<a class='logoLink' href='./' title='Open report index' aria-label='Open report index'>",
    )
    replacement = (
        "<a class='logoLink' href='../index.html' title='Open CLTR homepage' aria-label='Open CLTR homepage'>"
        "<img class='logoImage' src='../assets/logos/cltr.png' alt='CLTR logo'/>"
        "<span class='logoWordmark'>CLTR</span></a>"
    )
    for anchor in anchor_variants:
        start = text.find(anchor)
        if start == -1:
            continue
        svg_start = text.find("<svg class='logoMark'", start)
        end = text.find("</svg><span class='logoWordmark'>CLTR</span></a>", svg_start)
        if svg_start != -1 and end != -1:
            end += len("</svg><span class='logoWordmark'>CLTR</span></a>")
            text = text[:start] + replacement + text[end:]
            break
    path.write_text(text, encoding="utf-8")


def _sync_primary_header(path: Path, home_href: str, publication_href: str, logo_src: str) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        ".logoMark { width:58px; height:58px; display:block; flex-shrink:0; }",
        ".logoImage { width:58px; height:58px; object-fit:contain; display:block; flex-shrink:0; }",
    )
    anchor_variants = (
        "<a class='logoLink' href='index.html' title='Open report index' aria-label='Open report index'>",
        "<a class='logoLink' href='./' title='Open report index' aria-label='Open report index'>",
        "<a class='logoLink' href='../index.html' title='Open CLTR homepage' aria-label='Open CLTR homepage'>",
        "<a class='logoLink' href='../../index.html' title='Open report index' aria-label='Open report index'>",
        "<a class='logoLink' href='../../../index.html' title='Open report index' aria-label='Open report index'>",
        "<a class='logoLink' href='../../../../index.html' title='Open report index' aria-label='Open report index'>",
        "<a class='logoLink' href='../../../../index.html' title='Open CLTR homepage' aria-label='Open CLTR homepage'>",
    )
    replacement = (
        f"<a class='logoLink' href='{home_href}' title='Open CLTR homepage' aria-label='Open CLTR homepage'>"
        f"<img class='logoImage' src='{logo_src}' alt='CLTR logo'/>"
        "<span class='logoWordmark'>CLTR</span></a>"
    )
    for anchor in anchor_variants:
        start = text.find(anchor)
        if start == -1:
            continue
        svg_start = text.find("<svg class='logoMark'", start)
        end = text.find("</svg><span class='logoWordmark'>CLTR</span></a>", svg_start)
        if svg_start != -1 and end != -1:
            end += len("</svg><span class='logoWordmark'>CLTR</span></a>")
            text = text[:start] + replacement + text[end:]
            break
        img_start = text.find("<img class='logoImage'", start)
        img_end = text.find("</a>", img_start)
        if img_start != -1 and img_end != -1:
            img_end += len("</a>")
            text = text[:start] + replacement + text[img_end:]
            break
    github_link = (
        "<a class='socialLink' href='https://github.com/tomarp/cltr' title='Open GitHub' "
        "target='_blank' rel='noopener noreferrer'><span>GitHub</span></a>"
    )
    ordered_navigation = (
        f"<a class='socialLink' href='{home_href.replace('index.html', 'exp.html')}' title='Open Experiment'><span>Experiment</span></a>"
        f"<a class='socialLink' href='{publication_href}' title='Open Publication'><span>Publication</span></a>"
    )
    for label in ("Experiment", "Prediction Models", "Publication"):
        while True:
            marker = f"<span>{label}</span>"
            marker_pos = text.find(marker)
            if marker_pos == -1:
                break
            link_start = text.rfind("<a class='socialLink'", 0, marker_pos)
            link_end = text.find("</a>", marker_pos)
            if link_start == -1 or link_end == -1:
                break
            text = text[:link_start] + text[link_end + len("</a>"):]
    text = text.replace(github_link, ordered_navigation + github_link, 1)
    path.write_text(text, encoding="utf-8")


def _ensure_primary_menu(path: Path) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    if ".menuWrap {" not in text:
        text = text.replace(
            ".mastheadActions { display:flex; align-items:center; gap:12px; flex-shrink:0; }\n",
            ".mastheadActions { display:flex; align-items:center; gap:12px; flex-shrink:0; }\n"
            ".menuWrap { position:relative; display:flex; align-items:center; }\n",
            1,
        )
    if "id='siteMenuButton'" not in text:
        masthead_start = text.find("<div class='mastheadActions'>")
        social_start = text.find("<div class='socialLinks'>", masthead_start)
        theme_start = text.find("<button class='themeToggle'", social_start)
        social_end = text.find("</div>", social_start)
        if masthead_start != -1 and social_start != -1 and theme_start != -1 and social_end != -1 and social_end < theme_start:
            social_block = text[social_start:social_end + len("</div>")]
            menu_block = (
                "<div class='menuWrap'>"
                "<button class='menuButton' id='siteMenuButton' type='button' aria-expanded='false' "
                "aria-controls='siteMenuPanel' aria-label='Open site menu'>"
                "<span class='menuButtonBars' aria-hidden='true'><span></span><span></span><span></span></span>"
                "<span>Menu</span></button>"
                "<div class='menuPanel' id='siteMenuPanel' role='menu' aria-label='Site navigation'>"
                f"{social_block}"
                "</div></div>"
            )
            text = text[:social_start] + menu_block + text[social_end + len("</div>"):]
    text = text.replace(
        ".menuPanel { position:absolute; right:0; top:calc(100% + 10px); width:min(420px, calc(100vw - 32px)); max-height:min(70vh, 720px); overflow:auto; padding:14px 12px; background:rgba(255,255,255,0.97); border:1px solid rgba(148,163,184,0.22); border-radius:22px; box-shadow:0 22px 54px rgba(23,32,51,0.16); backdrop-filter:blur(18px); display:none; }\n",
        ".menuPanel { position:absolute; right:0; top:calc(100% + 10px); width:min(220px, calc(100vw - 32px)); max-height:min(70vh, 720px); overflow:auto; padding:0; background:transparent; border:0; border-radius:0; box-shadow:none; backdrop-filter:none; display:none; }\n",
    )
    text = text.replace("body.theme-dark .menuPanel { background:rgba(15,23,42,0.96); border-color:rgba(71,85,105,0.4); }\n", "")
    text = text.replace(
        ".socialLinks { display:flex; align-items:center; gap:10px; flex-wrap:wrap; }\n",
        ".socialLinks { display:flex; align-items:center; gap:10px; flex-wrap:wrap; }\n"
        ".menuPanel .socialLinks { display:grid; gap:8px; }\n"
        ".menuPanel .socialLink { width:100%; min-height:40px; justify-content:flex-start; padding:10px 12px; border-radius:14px; font-size:0.82rem; line-height:1.2; box-sizing:border-box; box-shadow:0 10px 20px rgba(23,32,51,0.12); background:linear-gradient(135deg,rgba(255,255,255,0.98) 0%,rgba(255,243,224,0.98) 52%,rgba(255,232,214,0.98) 100%); border:1px solid rgba(251,146,60,0.34); }\n",
        1,
    )
    text = text.replace(
        "body.theme-dark .socialLink,body.theme-dark .themeToggle,body.theme-dark .menuButton { color:#f8fafc; background:linear-gradient(180deg,rgba(30,41,59,0.96) 0%,rgba(15,23,42,0.96) 100%); border-color:rgba(71,85,105,0.5); }\n",
        "body.theme-dark .socialLink,body.theme-dark .themeToggle,body.theme-dark .menuButton { color:#f8fafc; background:linear-gradient(180deg,rgba(30,41,59,0.96) 0%,rgba(15,23,42,0.96) 100%); border-color:rgba(71,85,105,0.5); }\n"
        "body.theme-dark .menuPanel .socialLink { background:linear-gradient(135deg,rgba(30,41,59,0.98) 0%,rgba(37,99,235,0.34) 58%,rgba(15,23,42,0.98) 100%); border-color:rgba(96,165,250,0.34); box-shadow:0 10px 22px rgba(2,6,23,0.34); }\n",
        1,
    )
    text = text.replace(
        ".menuTitle { margin:0 0 2px; font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, \"Segoe UI\", sans-serif; font-size:0.78rem; letter-spacing:0.14em; text-transform:uppercase; color:#64748b; }\n",
        "",
    )
    text = text.replace("body.theme-dark .menuTitle { color:#94a3b8; }\n", "")
    text = text.replace(
        "@media (max-width:1000px) { .primaryBarInner,.secondaryBarInner,.hero,.grid,.heroFacts { grid-template-columns:1fr; } .primaryBarInner,.secondaryBarInner { display:grid; padding:12px 20px; } .mastheadActions,.secondaryBarActions { justify-content:space-between; } .secondaryBarText { white-space:normal; } .menuPanel { right:auto; left:0; width:min(100%, 420px); } .heroSticky { position:static; } .socialLinks { order:2; } }\n",
        "@media (max-width:1000px) { .primaryBarInner,.secondaryBarInner,.hero,.grid,.heroFacts { grid-template-columns:1fr; } .primaryBarInner,.secondaryBarInner { display:grid; padding:12px 20px; } .mastheadActions,.secondaryBarActions { justify-content:space-between; } .secondaryBarText { white-space:normal; } .mastheadActions .menuPanel { left:auto; right:0; width:min(220px, calc(100vw - 24px)); } .secondaryBarActions .menuPanel { right:auto; left:0; width:min(100%, 420px); } .heroSticky { position:static; } .secondaryBarActions .socialLinks { order:2; } }\n",
        1,
    )
    if "const siteMenuButton=document.getElementById('siteMenuButton');" not in text:
        snippet = (
            "const siteMenuButton=document.getElementById('siteMenuButton');\n"
            "const siteMenuPanel=document.getElementById('siteMenuPanel');\n"
            "const closeSiteMenu=()=>{if(!siteMenuPanel||!siteMenuButton)return;siteMenuPanel.classList.remove('open');siteMenuButton.setAttribute('aria-expanded','false');};\n"
            "const toggleSiteMenu=()=>{if(!siteMenuPanel||!siteMenuButton)return;const open=siteMenuPanel.classList.toggle('open');siteMenuButton.setAttribute('aria-expanded',open?'true':'false');};\n"
            "if(siteMenuButton&&siteMenuPanel){siteMenuButton.addEventListener('click',(event)=>{event.stopPropagation();toggleSiteMenu();});siteMenuPanel.querySelectorAll('a').forEach(link=>link.addEventListener('click',closeSiteMenu));document.addEventListener('click',(event)=>{if(!siteMenuPanel.contains(event.target)&&!siteMenuButton.contains(event.target))closeSiteMenu();});document.addEventListener('keydown',(event)=>{if(event.key==='Escape')closeSiteMenu();});}\n"
        )
        if "const sessionMenuButton" in text:
            text = text.replace("const sessionMenuButton", snippet + "const sessionMenuButton", 1)
        elif "</script>" in text:
            text = text.replace("</script>", snippet + "</script>", 1)
    path.write_text(text, encoding="utf-8")


def _ensure_atlas_footer_style(path: Path) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    if ".copyrightNote {" in text:
        return
    insert_after = (
        ".heroFacts { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; }\n"
    )
    footer_css = (
        ".copyrightNote { width:min(100%, 1360px); margin:0 auto; padding:0 clamp(16px,2.4vw,28px) 18px; "
        "box-sizing:border-box; text-align:center; color:#64748b; font:500 0.84rem/1.5 ui-sans-serif, "
        "-apple-system, BlinkMacSystemFont, \"Segoe UI\", sans-serif; }\n"
        "body.theme-dark .copyrightNote { color:#94a3b8; }\n"
    )
    if insert_after in text:
        text = text.replace(insert_after, insert_after + footer_css, 1)
    else:
        text = text.replace("</style>", footer_css + "</style>", 1)
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
    _rewrite_text(
        atlas_index_target,
        [
            ("../sessions/", "sessions/"),
            ("../cohort/", "cohort/"),
        ],
    )
    _normalize_atlas_home_logo(atlas_index_target)
    _sync_primary_header(atlas_index_target, "../index.html", "../publication.html", "../assets/logos/cltr.png")
    _ensure_primary_menu(atlas_index_target)
    _ensure_atlas_footer_style(atlas_index_target)
    _ensure_hide_index_html(atlas_index_target)
    if target_dir == docs_atlas_dir:
        for html_path in (target_dir / "cohort").rglob("*.html"):
            _rewrite_text(html_path, [("../../work/index.html", "../../../index.html")])
            _sync_primary_header(html_path, "../../../index.html", "../../../publication.html", "../../../assets/logos/cltr.png")
            _ensure_primary_menu(html_path)
            _ensure_hide_index_html(html_path)
        for html_path in (target_dir / "sessions").rglob("*.html"):
            _rewrite_text(html_path, [("../../../work/index.html", "../../../../index.html")])
            _sync_primary_header(html_path, "../../../../index.html", "../../../../publication.html", "../../../../assets/logos/cltr.png")
            _ensure_primary_menu(html_path)
            _ensure_hide_index_html(html_path)
    else:
        for html_path in (target_dir / "cohort").rglob("*.html"):
            _rewrite_text(html_path, [("../../work/index.html", "../../index.html")])
            _sync_primary_header(html_path, "../../index.html", "../../publication.html", "../../assets/logos/cltr.png")
            _ensure_primary_menu(html_path)
            _ensure_hide_index_html(html_path)
        for html_path in (target_dir / "sessions").rglob("*.html"):
            _rewrite_text(html_path, [("../../../work/index.html", "../../../index.html")])
            _sync_primary_header(html_path, "../../../index.html", "../../../publication.html", "../../../assets/logos/cltr.png")
            _ensure_primary_menu(html_path)
            _ensure_hide_index_html(html_path)
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
