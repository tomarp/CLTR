from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path


PUBLISH_DIRS = ("cohort", "sessions")
COHORT_ROUTE_MAP = {
    "cohort_report.html": "",
    "cohort.html": "",
    "cohort_ch01_overview_audit.html": "ch01/",
    "cohort_ch02_subjective_behavioral.html": "ch02/",
    "cohort_ch03_physiological.html": "ch03/",
    "cohort_ch04_environmental.html": "ch04/",
    "cohort_ch05_derived_results.html": "ch05/",
    "cohort_ch06_relationships_validation.html": "ch06/",
}
COHORT_CANONICAL_MAP = {
    "cohort_ch01_overview_audit.html": "ch01",
    "cohort_ch02_subjective_behavioral.html": "ch02",
    "cohort_ch03_physiological.html": "ch03",
    "cohort_ch04_environmental.html": "ch04",
    "cohort_ch05_derived_results.html": "ch05",
    "cohort_ch06_relationships_validation.html": "ch06",
}
COHORT_LEGACY_INDEX_NAMES = ("cohort_report.html", "cohort.html")
COHORT_FULL_LEGACY_NAME = "cohort_full_report.html"
COHORT_FULL_CANONICAL_DIR = "full"


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


def _ensure_logo_image_style(text: str) -> str:
    if ".logoImage {" in text:
        return text
    logo_rule = ".logoMark { width:58px; height:58px; object-fit:contain; display:block; flex-shrink:0; }"
    replacement = (
        ".logoMark,.logoImage { width:58px; height:58px; object-fit:contain; display:block; flex-shrink:0; }"
    )
    if logo_rule in text:
        return text.replace(logo_rule, replacement, 1)
    legacy_logo_rule = ".logoMark { width:58px; height:58px; display:block; flex-shrink:0; }"
    if legacy_logo_rule in text:
        return text.replace(legacy_logo_rule, replacement, 1)
    return text.replace("</style>", replacement + "\n</style>", 1)


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
    text = _ensure_logo_image_style(text)
    replacement = (
        "<a class='logoLink' href='../index.html' title='Open CLTR homepage' aria-label='Open CLTR homepage'>"
        "<img class='logoImage' src='../assets/logos/cltr.png' alt='CLTR logo'/>"
        "<span class='logoWordmark'>CLTR</span></a>"
    )
    start = text.find("<a class='logoLink'")
    if start != -1:
        end = text.find("</a>", start)
        if end != -1:
            end += len("</a>")
            text = text[:start] + replacement + text[end:]
    path.write_text(text, encoding="utf-8")


def _sync_primary_header(path: Path, home_href: str, publication_href: str, logo_src: str) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    text = _ensure_logo_image_style(text)
    replacement = (
        f"<a class='logoLink' href='{home_href}' title='Open CLTR homepage' aria-label='Open CLTR homepage'>"
        f"<img class='logoImage' src='{logo_src}' alt='CLTR logo'/>"
        "<span class='logoWordmark'>CLTR</span></a>"
    )
    start = text.find("<a class='logoLink'")
    if start != -1:
        end = text.find("</a>", start)
        if end != -1:
            end += len("</a>")
            text = text[:start] + replacement + text[end:]
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
    text = text.replace(
        ".mastheadActions { display:flex; align-items:center; gap:12px; flex-shrink:0; }\n",
        ".mastheadActions { display:flex; align-items:center; justify-content:flex-end; gap:12px; flex:1 1 auto; min-width:0; }\n",
    )
    if ".menuWrap {" not in text:
        text = text.replace(
            ".mastheadActions { display:flex; align-items:center; justify-content:flex-end; gap:12px; flex:1 1 auto; min-width:0; }\n",
            ".mastheadActions { display:flex; align-items:center; justify-content:flex-end; gap:12px; flex:1 1 auto; min-width:0; }\n"
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
        ".socialLinks { display:flex; align-items:center; gap:10px; flex-wrap:wrap; }\n",
        ".socialLinks { display:flex; align-items:center; justify-content:flex-end; gap:10px; flex-wrap:wrap; min-width:0; }\n",
    )
    text = text.replace(
        ".menuPanel { position:absolute; right:0; top:calc(100% + 10px); width:min(420px, calc(100vw - 32px)); max-height:min(70vh, 720px); overflow:auto; padding:14px 12px; background:rgba(255,255,255,0.97); border:1px solid rgba(148,163,184,0.22); border-radius:22px; box-shadow:0 22px 54px rgba(23,32,51,0.16); backdrop-filter:blur(18px); display:none; }\n",
        ".menuPanel { position:absolute; right:0; top:calc(100% + 10px); width:min(220px, calc(100vw - 32px)); max-height:min(70vh, 720px); overflow:auto; padding:0; background:transparent; border:0; border-radius:0; box-shadow:none; backdrop-filter:none; display:none; }\n",
    )
    secondary_menu_block = (
        ".secondaryBarActions .menuPanel { position:absolute; right:0; top:calc(100% + 10px); width:min(420px, calc(100vw - 32px)); "
        "max-height:min(70vh, 720px); overflow:auto; padding:14px 12px; background:rgba(255,255,255,0.97); "
        "border:1px solid rgba(148,163,184,0.22); border-radius:22px; box-shadow:0 22px 54px rgba(23,32,51,0.16); "
        "backdrop-filter:blur(18px); display:none; }\n"
        ".secondaryBarActions .menuPanel.open { display:grid; gap:10px; }\n"
        ".secondaryBarActions .menuTitle { margin:0 0 2px; font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, \"Segoe UI\", sans-serif; "
        "font-size:0.78rem; letter-spacing:0.14em; text-transform:uppercase; color:#64748b; }\n"
        "body.theme-dark .secondaryBarActions .menuPanel { background:rgba(15,23,42,0.96); border-color:rgba(71,85,105,0.4); }\n"
        "body.theme-dark .secondaryBarActions .menuTitle { color:#94a3b8; }\n"
    )
    if ".secondaryBarActions .menuPanel {" not in text:
        text = text.replace(".menuPanel.open { display:grid; gap:10px; }\n", ".menuPanel.open { display:grid; gap:10px; }\n" + secondary_menu_block, 1)
    while True:
        title_start = text.find("<h2 class='menuTitle'>")
        if title_start == -1:
            break
        title_end = text.find("</h2>", title_start)
        if title_end == -1:
            break
        title_end += len("</h2>")
        text = text[:title_start] + text[title_end:]
    text = text.replace("body.theme-dark .menuPanel { background:rgba(15,23,42,0.96); border-color:rgba(71,85,105,0.4); }\n", "")
    text = text.replace(
        ".socialLinks { display:flex; align-items:center; gap:10px; flex-wrap:wrap; }\n",
        ".socialLinks { display:flex; align-items:center; justify-content:flex-end; gap:10px; flex-wrap:wrap; min-width:0; }\n"
        ".menuPanel .socialLinks { display:grid; gap:8px; }\n"
        ".menuPanel .socialLink { width:100%; min-height:40px; justify-content:flex-start; padding:10px 12px; border-radius:14px; font-size:0.82rem; line-height:1.2; box-sizing:border-box; box-shadow:0 10px 20px rgba(23,32,51,0.12); background:linear-gradient(135deg,rgba(255,255,255,0.98) 0%,rgba(255,243,224,0.98) 52%,rgba(255,232,214,0.98) 100%); border:1px solid rgba(251,146,60,0.34); }\n",
        1,
    )
    text = text.replace(
        ".menuButton { appearance:none; border:1px solid rgba(148,163,184,0.28); background:rgba(255,255,255,0.92); color:#172033; border-radius:999px; min-height:42px; padding:0 14px; display:inline-flex; align-items:center; gap:10px; font:700 0.82rem/1 ui-sans-serif, -apple-system, BlinkMacSystemFont, \"Segoe UI\", sans-serif; letter-spacing:0.04em; cursor:pointer; box-shadow:0 12px 28px rgba(23,32,51,0.08); }\n",
        ".menuButton { appearance:none; border:1px solid rgba(148,163,184,0.28); background:linear-gradient(180deg,rgba(255,255,255,0.96) 0%,rgba(255,247,237,0.96) 100%); color:#172033; border-radius:999px; min-height:44px; padding:0 14px; display:inline-flex; align-items:center; gap:10px; font:700 0.82rem/1 ui-sans-serif, -apple-system, BlinkMacSystemFont, \"Segoe UI\", sans-serif; letter-spacing:0.04em; cursor:pointer; box-shadow:0 12px 28px rgba(23,32,51,0.08); }\n",
    )
    text = text.replace(
        "body.theme-dark .socialLink,body.theme-dark .themeToggle,body.theme-dark .menuButton { color:#f8fafc; background:linear-gradient(180deg,rgba(30,41,59,0.96) 0%,rgba(15,23,42,0.96) 100%); border-color:rgba(71,85,105,0.5); }\n",
        "body.theme-dark .socialLink,body.theme-dark .themeToggle,body.theme-dark .menuButton { color:#f8fafc; background:linear-gradient(180deg,rgba(30,41,59,0.96) 0%,rgba(15,23,42,0.96) 100%); border-color:rgba(71,85,105,0.5); }\n"
        "body.theme-dark .menuPanel .socialLink { background:linear-gradient(135deg,rgba(30,41,59,0.98) 0%,rgba(37,99,235,0.34) 58%,rgba(15,23,42,0.98) 100%); border-color:rgba(96,165,250,0.34); box-shadow:0 10px 22px rgba(2,6,23,0.34); }\n",
        1,
    )
    text = text.replace("body.theme-dark .menuPanel { background:rgba(15,23,42,0.96); border-color:rgba(71,85,105,0.4); }\n", "")
    text = text.replace(
        ".menuTitle { margin:0 0 2px; font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, \"Segoe UI\", sans-serif; font-size:0.78rem; letter-spacing:0.14em; text-transform:uppercase; color:#64748b; }\n",
        "",
    )
    text = text.replace("body.theme-dark .menuTitle { color:#94a3b8; }\n", "")
    text = text.replace(".menuTitle { display:none; }\n", "")
    text = text.replace(
        "@media (max-width:860px) { .primaryBarInner { flex-wrap:wrap; padding:12px 20px; } .mastheadActions { width:100%; justify-content:flex-end; } }\n",
        "",
    )
    text = text.replace(
        "@media (max-width:640px) { .mastheadActions { width:auto; } .menuPanel { right:0; left:auto; width:min(280px, calc(100vw - 24px)); } .logoMark { width:52px; height:52px; } .logoWordmark { height:52px; font-size:1.9rem; } }\n",
        "",
    )
    text = text.replace(
        "@media (max-width:1000px) { .primaryBarInner,.secondaryBarInner,.hero,.grid,.heroFacts { grid-template-columns:1fr; } .primaryBarInner,.secondaryBarInner { display:grid; padding:12px 20px; } .mastheadActions,.secondaryBarActions { justify-content:space-between; } .secondaryBarText { white-space:normal; } .menuPanel { right:auto; left:0; width:min(100%, 420px); } .heroSticky { position:static; } .socialLinks { order:2; } }\n",
        "@media (max-width:1000px) { .primaryBarInner,.secondaryBarInner,.hero,.grid,.heroFacts { grid-template-columns:1fr; } .primaryBarInner,.secondaryBarInner { display:grid; padding:12px 20px; } .mastheadActions,.secondaryBarActions { justify-content:space-between; } .secondaryBarText { white-space:normal; } .mastheadActions .menuPanel { left:auto; right:0; width:min(220px, calc(100vw - 24px)); } .secondaryBarActions .menuPanel { right:auto; left:0; width:min(100%, 420px); } .heroSticky { position:static; } .secondaryBarActions .socialLinks { order:2; } }\n",
        1,
    )
    if "@media (max-width:860px) { .primaryBarInner { flex-wrap:wrap; padding:12px 20px; } .mastheadActions { width:100%; justify-content:flex-end; } }\n" not in text:
        text = text.replace(
            "@media (max-width:1000px) { .primaryBarInner,.secondaryBarInner,.hero,.grid,.heroFacts { grid-template-columns:1fr; } .primaryBarInner,.secondaryBarInner { display:grid; padding:12px 20px; } .mastheadActions,.secondaryBarActions { justify-content:space-between; } .secondaryBarText { white-space:normal; } .mastheadActions .menuPanel { left:auto; right:0; width:min(220px, calc(100vw - 24px)); } .secondaryBarActions .menuPanel { right:auto; left:0; width:min(100%, 420px); } .heroSticky { position:static; } .secondaryBarActions .socialLinks { order:2; } }\n",
            "@media (max-width:1000px) { .primaryBarInner,.secondaryBarInner,.hero,.grid,.heroFacts { grid-template-columns:1fr; } .primaryBarInner,.secondaryBarInner { display:grid; padding:12px 20px; } .mastheadActions,.secondaryBarActions { justify-content:space-between; } .secondaryBarText { white-space:normal; } .mastheadActions .menuPanel { left:auto; right:0; width:min(220px, calc(100vw - 24px)); } .secondaryBarActions .menuPanel { right:auto; left:0; width:min(100%, 420px); } .heroSticky { position:static; } .secondaryBarActions .socialLinks { order:2; } }\n"
            "@media (max-width:860px) { .primaryBarInner { flex-wrap:wrap; padding:12px 20px; } .mastheadActions { width:100%; justify-content:flex-end; } }\n"
            "@media (max-width:640px) { .mastheadActions { width:auto; } .menuPanel { right:0; left:auto; width:min(280px, calc(100vw - 24px)); } .logoMark,.logoImage { width:52px; height:52px; } .logoWordmark { height:52px; font-size:1.9rem; } }\n",
            1,
        )
    text = re.sub(
        r"(?m)^\.menuButton \{[^}]*\}\n",
        ".menuButton { appearance:none; border:1px solid rgba(148,163,184,0.28); background:linear-gradient(180deg,rgba(255,255,255,0.96) 0%,rgba(255,247,237,0.96) 100%); color:#172033; border-radius:999px; min-height:44px; padding:0 14px; display:inline-flex; align-items:center; gap:10px; font:700 0.82rem/1 ui-sans-serif, -apple-system, BlinkMacSystemFont, \"Segoe UI\", sans-serif; letter-spacing:0.04em; cursor:pointer; box-shadow:0 12px 28px rgba(23,32,51,0.08); }\n",
        text,
        count=1,
    )
    text = re.sub(
        r"\.menuButton:hover \{[^}]*\}\n",
        ".menuButton:hover { background:#ffffff; border-color:#fb923c; box-shadow:0 16px 34px rgba(23,32,51,0.12); transform:translateY(-1px); }\n",
        text,
        count=1,
    )
    if ".menuPanel .socialLinks {" not in text:
        text = text.replace(
            ".socialLink.isDisabled { pointer-events:none; opacity:0.58; }\n",
            ".socialLink.isDisabled { pointer-events:none; opacity:0.58; }\n"
            ".menuPanel .socialLinks { display:grid; gap:8px; }\n"
            ".menuPanel .socialLink { width:100%; min-height:40px; justify-content:flex-start; padding:10px 12px; border-radius:14px; font-size:0.82rem; line-height:1.2; box-sizing:border-box; box-shadow:0 10px 20px rgba(23,32,51,0.12); background:linear-gradient(135deg,rgba(255,255,255,0.98) 0%,rgba(255,243,224,0.98) 52%,rgba(255,232,214,0.98) 100%); border:1px solid rgba(251,146,60,0.34); }\n",
            1,
        )
    text = re.sub(
        r"(?m)^body\.theme-dark \.socialLink,body\.theme-dark \.themeToggle,body\.theme-dark \.menuButton \{[^}]*\}\n",
        "body.theme-dark .socialLink,body.theme-dark .themeToggle,body.theme-dark .menuButton { color:#f8fafc; background:linear-gradient(180deg,rgba(30,41,59,0.96) 0%,rgba(15,23,42,0.96) 100%); border-color:rgba(71,85,105,0.5); }\n",
        text,
        count=1,
    )
    text = text.replace(
        ".secondaryBarActions body.theme-dark .secondaryBarActions .menuPanel { background:rgba(15,23,42,0.96); border-color:rgba(71,85,105,0.4); }\n",
        "body.theme-dark .secondaryBarActions .menuPanel { background:rgba(15,23,42,0.96); border-color:rgba(71,85,105,0.4); }\n",
    )
    text = text.replace(
        ".menuPanel { right:auto; left:0; width:min(100%, 420px); }",
        ".mastheadActions .menuPanel { left:auto; right:0; width:min(220px, calc(100vw - 24px)); }",
    )
    text = text.replace(
        ".socialLinks { order:2; }",
        ".secondaryBarActions .socialLinks { order:2; }",
    )
    text = re.sub(
        r"@media \(max-width:1000px\) \{ ([^}]*) \.menuPanel \{ right:auto; left:0; width:min\(100%, 420px\); \} ([^}]*) \.socialLinks \{ order:2; \} ([^}]*) \}\n",
        r"@media (max-width:1000px) { \1 .mastheadActions .menuPanel { left:auto; right:0; width:min(220px, calc(100vw - 24px)); } \2 .secondaryBarActions .socialLinks { order:2; } \3 }\n",
        text,
        count=1,
    )
    if "@media (max-width:860px) { .primaryBarInner { flex-wrap:wrap; padding:12px 20px; } .mastheadActions { width:100%; justify-content:flex-end; } }\n" not in text:
        text = text.replace(
            "</style>",
            "@media (max-width:860px) { .primaryBarInner { flex-wrap:wrap; padding:12px 20px; } .mastheadActions { width:100%; justify-content:flex-end; } }\n"
            "@media (max-width:640px) { .mastheadActions { width:auto; } .menuPanel { right:0; left:auto; width:min(280px, calc(100vw - 24px)); } .logoMark,.logoImage { width:52px; height:52px; } .logoWordmark { height:52px; font-size:1.9rem; } }\n"
            "</style>",
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
    text = re.sub(
        r"const (figure|session|chapter)MenuButton=document\.getElementById\('[^']+'\);"
        r" const \1MenuPanel=document\.getElementById\('[^']+'\);\s*"
        r"const close[A-Za-z]+=\(\)=>\{.*?\};\s*"
        r"const toggle[A-Za-z]+=\(\)=>\{.*?\};\s*"
        r"if\(\1MenuButton&&\1MenuPanel\)\{.*?\}\s*",
        "",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(r"^.*const (figure|session|chapter)MenuButton=document\.getElementById\(.*$\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"^.*const close(Figure|Session|Chapter)[A-Za-z]*=.*$\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"^.*const toggle(Figure|Session|Chapter)[A-Za-z]*=.*$\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"^.*if\((figure|session|chapter)MenuButton&&(figure|session|chapter)MenuPanel\)\{.*$\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"^.*MenuPanel\.querySelectorAll\('a'\)\.forEach\(link=>link\.addEventListener\('click', close(Figure|Session|Chapter)[A-Za-z]*\)\);.*$\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"^.*close(Figure|Session|Chapter)[A-Za-z]*\(\);.*$\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\);\s*$\n?", "", text, flags=re.MULTILINE)
    if "const secondaryMenuButtons=[...document.querySelectorAll('.secondaryBarActions .menuButton[aria-controls]')];" not in text:
        secondary_menu_snippet = (
            "const secondaryMenuButtons=[...document.querySelectorAll('.secondaryBarActions .menuButton[aria-controls]')];\n"
            "const closeSecondaryMenus=(exceptPanelId='')=>{secondaryMenuButtons.forEach((button)=>{const panelId=button.getAttribute('aria-controls');const panel=panelId?document.getElementById(panelId):null;if(!panel||panelId===exceptPanelId)return;panel.classList.remove('open');button.setAttribute('aria-expanded','false');});};\n"
            "secondaryMenuButtons.forEach((button)=>{const panelId=button.getAttribute('aria-controls');const panel=panelId?document.getElementById(panelId):null;if(!panel)return;button.addEventListener('click',(event)=>{event.preventDefault();event.stopPropagation();const willOpen=!panel.classList.contains('open');closeSecondaryMenus();if(willOpen){panel.classList.add('open');button.setAttribute('aria-expanded','true');}else{panel.classList.remove('open');button.setAttribute('aria-expanded','false');}});panel.querySelectorAll('a').forEach((link)=>link.addEventListener('click',()=>{closeSecondaryMenus();button.setAttribute('aria-expanded','false');}));});\n"
            "const hoverCapable=window.matchMedia&&window.matchMedia('(hover: hover) and (pointer: fine)').matches;\n"
            "if(hoverCapable){let hoverCloseTimer=0;const cancelHoverClose=()=>{if(hoverCloseTimer){clearTimeout(hoverCloseTimer);hoverCloseTimer=0;}};const scheduleHoverClose=()=>{cancelHoverClose();hoverCloseTimer=setTimeout(()=>closeSecondaryMenus(),140);};secondaryMenuButtons.forEach((button)=>{const panelId=button.getAttribute('aria-controls');const panel=panelId?document.getElementById(panelId):null;if(!panel)return;const openOnHover=()=>{cancelHoverClose();closeSecondaryMenus(panelId);panel.classList.add('open');button.setAttribute('aria-expanded','true');};button.addEventListener('mouseenter',openOnHover);button.addEventListener('mouseleave',scheduleHoverClose);panel.addEventListener('mouseenter',cancelHoverClose);panel.addEventListener('mouseleave',scheduleHoverClose);});}\n"
            "document.addEventListener('click',(event)=>{if(!event.target.closest('.secondaryBarActions'))closeSecondaryMenus();},true);\n"
            "document.addEventListener('keydown',(event)=>{if(event.key==='Escape')closeSecondaryMenus();},true);\n"
        )
        text = text.replace("</script>", secondary_menu_snippet + "</script>", 1)
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


def _relative_href(from_dir: Path, to_path: Path, *, directory: bool = False) -> str:
    rel = Path(os.path.relpath(to_path, from_dir)).as_posix()
    if directory:
        if rel == ".":
            return "./"
        rel_dir = Path(os.path.relpath(to_path.parent, from_dir)).as_posix()
        return "./" if rel_dir == "." else rel_dir.rstrip("/") + "/"
    return rel


def _relative_dir_href(from_dir: Path, to_dir: Path) -> str:
    rel = Path(os.path.relpath(to_dir, from_dir)).as_posix()
    return "./" if rel == "." else rel.rstrip("/") + "/"


def _canonicalize_cohort_routes(cohort_dir: Path) -> None:
    index_target = cohort_dir / "index.html"
    for legacy_name in COHORT_LEGACY_INDEX_NAMES:
        legacy_path = cohort_dir / legacy_name
        if legacy_path.exists():
            if not index_target.exists():
                index_target.write_bytes(legacy_path.read_bytes())
            legacy_path.unlink(missing_ok=True)

    # Publish each chapter once as cohort/ch01.html..ch06.html (flat), and remove legacy
    # duplicates so the published atlas stays clean.
    for legacy_name, canonical_dirname in COHORT_CANONICAL_MAP.items():
        legacy_path = cohort_dir / legacy_name
        if not legacy_path.exists():
            continue
        canonical_dir = cohort_dir / canonical_dirname
        canonical_dir.mkdir(parents=True, exist_ok=True)
        canonical_path = canonical_dir / "index.html"
        canonical_path.write_bytes(legacy_path.read_bytes())
        legacy_path.unlink(missing_ok=True)

    full_legacy = cohort_dir / COHORT_FULL_LEGACY_NAME
    if full_legacy.exists():
        full_dir = cohort_dir / COHORT_FULL_CANONICAL_DIR
        full_dir.mkdir(parents=True, exist_ok=True)
        (full_dir / "index.html").write_bytes(full_legacy.read_bytes())
        full_legacy.unlink(missing_ok=True)

    # Remove previously-used flat canonical names if present.
    for old in ("cohort_full.html", "ch01.html", "ch02.html", "ch03.html", "ch04.html", "ch05.html", "ch06.html"):
        (cohort_dir / old).unlink(missing_ok=True)

    # Remove older directory-based routes (cohort/ch01/index.html etc.) to avoid duplicates.
    for legacy_route in COHORT_ROUTE_MAP.values():
        legacy_route = legacy_route.strip("/")
        if not legacy_route:
            continue
        # COHORT_ROUTE_MAP matches our desired directory names; keep them.
        if legacy_route in set(COHORT_CANONICAL_MAP.values()):
            continue
        shutil.rmtree(cohort_dir / legacy_route, ignore_errors=True)


def _rewrite_work_index_links(path: Path, replacement: str) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    updated = re.sub(r"(?:\.\./)+(?:work|execution)/index\.html", replacement, text)
    if updated != text:
        path.write_text(updated, encoding="utf-8")


def _rewrite_cohort_links(path: Path, target_dir: Path) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    route_targets: list[tuple[str, str]] = []
    cohort_root = target_dir / "cohort"
    cohort_index_href = _relative_dir_href(path.parent, cohort_root)
    route_targets.append((r"(?:cohort/cohort_report\.html|cohort/index\.html|cohort/|cohort/cohort\.html)", cohort_index_href))
    for legacy_name, canonical_dirname in COHORT_CANONICAL_MAP.items():
        canonical_dir = cohort_root / canonical_dirname
        canonical_href = _relative_dir_href(path.parent, canonical_dir)
        route_prefix = canonical_dirname
        route_targets.append(
            (
                rf"(?:cohort/{re.escape(legacy_name)}|cohort/{re.escape(route_prefix)}/index\.html|cohort/{re.escape(route_prefix)}/)",
                canonical_href,
            )
        )
    for pattern, replacement in route_targets:
        text = re.sub(rf"((?:href|src)=['\"]){pattern}(['\"])", rf"\1{replacement}\2", text)
    for legacy_name, canonical_dirname in COHORT_CANONICAL_MAP.items():
        canonical_href = _relative_dir_href(path.parent, cohort_root / canonical_dirname)
        text = re.sub(rf"((?:href|src)=['\"]){re.escape(legacy_name)}(['\"])", rf"\1{canonical_href}\2", text)
        # Also catch values that include a path prefix ending in the legacy filename.
        text = re.sub(
            rf"((?:href|src)=['\"])(?:[^'\"]*/)?{re.escape(legacy_name)}(['\"])",
            rf"\1{canonical_href}\2",
            text,
        )
    for legacy_name in COHORT_LEGACY_INDEX_NAMES:
        text = re.sub(rf"((?:href|src)=['\"]){re.escape(legacy_name)}(['\"])", rf"\1{cohort_index_href}\2", text)
        text = re.sub(
            rf"((?:href|src)=['\"])(?:[^'\"]*/)?{re.escape(legacy_name)}(['\"])",
            rf"\1{cohort_index_href}\2",
            text,
        )
    full_href = _relative_dir_href(path.parent, cohort_root / COHORT_FULL_CANONICAL_DIR)
    text = re.sub(rf"((?:href|src)=['\"]){re.escape(COHORT_FULL_LEGACY_NAME)}(['\"])", rf"\1{full_href}\2", text)
    text = re.sub(
        rf"((?:href|src)=['\"])(?:[^'\"]*/)?{re.escape(COHORT_FULL_LEGACY_NAME)}(['\"])",
        rf"\1{full_href}\2",
        text,
    )
    path.write_text(text, encoding="utf-8")


def _rewrite_sessions_links(path: Path, target_dir: Path) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    sessions_href = _relative_dir_href(path.parent, target_dir / "sessions")
    text = re.sub(r"((?:href|src)=['\"])(?:\.\./)*(?:sessions_report|sessions)\.html", rf"\1{sessions_href}", text)
    path.write_text(text, encoding="utf-8")


def _rewrite_cohort_figure_links(path: Path, target_dir: Path) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    figures_href = _relative_dir_href(path.parent, target_dir / "cohort" / "figures")
    replacements = [
        ("src='figures/", f"src='{figures_href}"),
        ('src="figures/', f'src="{figures_href}'),
        ("href='figures/", f"href='{figures_href}"),
        ('href="figures/', f'href="{figures_href}'),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")


def _published_site_targets(target_dir: Path, docs_atlas_dir: Path) -> tuple[Path, Path, Path]:
    if target_dir == docs_atlas_dir:
        docs_root = docs_atlas_dir.parent
    else:
        docs_root = docs_atlas_dir
    return (
        docs_root / "index.html",
        docs_root / "publication.html",
        docs_root / "assets" / "logos" / "cltr.png",
    )


def _finalize_published_html(path: Path, target_dir: Path, docs_atlas_dir: Path) -> None:
    if not path.exists():
        return
    site_home, publication, logo = _published_site_targets(target_dir, docs_atlas_dir)
    home_href = _relative_href(path.parent, site_home)
    publication_href = _relative_href(path.parent, publication)
    logo_src = _relative_href(path.parent, logo)
    _rewrite_work_index_links(path, home_href)
    _rewrite_cohort_links(path, target_dir)
    _rewrite_sessions_links(path, target_dir)
    if target_dir / "cohort" in path.parents:
        _rewrite_cohort_figure_links(path, target_dir)
    _sync_primary_header(path, home_href, publication_href, logo_src)
    _ensure_primary_menu(path)
    _ensure_hide_index_html(path)


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

    atlas_index_candidates = [
        reports_dir / "execution" / "index.html",
        reports_dir / "work" / "index.html",
        reports_dir / "index.html",
    ]
    atlas_index_src = next((path for path in atlas_index_candidates if path.exists()), None)
    if atlas_index_src is None:
        checked = ", ".join(str(path) for path in atlas_index_candidates)
        raise FileNotFoundError(f"Atlas entry HTML not found. Checked: {checked}")

    atlas_index_target = target_dir / "index.html"
    shutil.copy2(atlas_index_src, atlas_index_target)
    sessions_index_candidates = [
        reports_dir / "sessions.html",
        reports_dir / "sessions_report.html",
    ]
    sessions_index_src = next((path for path in sessions_index_candidates if path.exists()), None)
    if sessions_index_src is not None:
        sessions_dir = target_dir / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sessions_index_src, sessions_dir / "index.html")
    (target_dir / "sessions_report.html").unlink(missing_ok=True)
    cohort_target_dir = target_dir / "cohort"
    if cohort_target_dir.exists():
        _canonicalize_cohort_routes(cohort_target_dir)
    _rewrite_text(
        atlas_index_target,
        [
            ("../sessions/", "sessions/"),
            ("../cohort/", "cohort/"),
        ],
    )
    _normalize_atlas_home_logo(atlas_index_target)
    _rewrite_cohort_links(atlas_index_target, target_dir)
    _rewrite_sessions_links(atlas_index_target, target_dir)
    _sync_primary_header(atlas_index_target, "../index.html", "../publication.html", "../assets/logos/cltr.png")
    _ensure_primary_menu(atlas_index_target)
    _ensure_atlas_footer_style(atlas_index_target)
    _ensure_hide_index_html(atlas_index_target)
    if target_dir == docs_atlas_dir:
        sessions_report_target = target_dir / "sessions.html"
        if sessions_report_target.exists():
            _finalize_published_html(sessions_report_target, target_dir, docs_atlas_dir)
        for html_path in (target_dir / "cohort").rglob("*.html"):
            _finalize_published_html(html_path, target_dir, docs_atlas_dir)
        for html_path in (target_dir / "sessions").rglob("*.html"):
            _finalize_published_html(html_path, target_dir, docs_atlas_dir)
    else:
        sessions_report_target = target_dir / "sessions.html"
        if sessions_report_target.exists():
            _finalize_published_html(sessions_report_target, target_dir, docs_atlas_dir)
        for html_path in (target_dir / "cohort").rglob("*.html"):
            _finalize_published_html(html_path, target_dir, docs_atlas_dir)
        for html_path in (target_dir / "sessions").rglob("*.html"):
            _finalize_published_html(html_path, target_dir, docs_atlas_dir)
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
