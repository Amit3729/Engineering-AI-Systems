"""Turning corpus files into indexable text.

The knowledge base holds two very different kinds of file: a handful of
hand-written Markdown notes, and the CPython 3.14 documentation as a Sphinx
HTML build (579 pages, 74 MB). Feeding raw HTML to the chunker would index
navigation chrome, sidebars and the pilcrow links that Sphinx sprinkles after
every heading, so pages are parsed into *sections* first.

A section is the unit Sphinx already uses to organise a page (``<section
id="basic-usage">``), which makes it a far better chunk boundary than a blind
180-word window: ``library/datetime.html`` is 16k words, and a window that
straddles two unrelated APIs retrieves badly for both. Carrying the section id
also means a citation can point at ``library/json.html#json.dumps`` rather than
at the page as a whole.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

SUPPORTED_SUFFIXES = (".md", ".txt", ".html")

# Bump whenever extraction changes: it feeds the index fingerprint, so an index
# built by an older parser is rebuilt instead of silently kept.
EXTRACTOR_VERSION = "1"

# Sphinx build artefacts that carry no prose: asset directories, the generated
# A-Z index, the JS search page, and the boilerplate legal pages.
DEFAULT_EXCLUDES = (
    "_static/*",
    "_downloads/*",
    "_images/*",
    "_sources/*",
    "genindex*.html",
    "py-modindex.html",
    "search.html",
    "404.html",
    "license.html",
    "copyright.html",
    "improve-page*.html",
)

# Chrome that lives inside the main content area of a Sphinx page.
_STRIP_TAGS = ("script", "style", "nav", "header", "footer", "form")
_STRIP_CLASSES = ("headerlink", "sphinxsidebar", "related", "clearer", "admonition-todo")

# This build ships stubs where a page moved (library/stdtypes.html -> builtins/).
# They hold one sentence of boilerplate and would only ever match noise.
_REDIRECT_MARKER = "should have been redirected"

_BLOCK_TAGS = ("p", "pre", "dl", "ul", "ol", "table", "blockquote", "h1", "h2", "h3", "h4", "h5", "h6")
# Layout-only containers: descend into them instead of rendering them whole.
_WRAPPER_TAGS = ("div", "aside", "figure", "article", "main")


@dataclass(frozen=True)
class Section:
    """One indexable slice of a document."""

    heading: str  # breadcrumb, e.g. "json - JSON encoder and decoder > Basic Usage"
    anchor: str  # in-page id, e.g. "basic-usage"; empty for non-HTML files
    text: str


@dataclass(frozen=True)
class Document:
    title: str
    sections: list[Section]


def is_indexable(path: Path, root: Path, excludes: tuple[str, ...] = DEFAULT_EXCLUDES) -> bool:
    """Should this file go into the index?"""
    if not path.is_file() or path.suffix not in SUPPORTED_SUFFIXES:
        return False
    relative = path.relative_to(root).as_posix()
    return not any(fnmatch(relative, pattern) or fnmatch(path.name, pattern) for pattern in excludes)


def _clean(text: str) -> str:
    # Sphinx leaves a pilcrow on every heading and non-breaking spaces in signatures.
    text = text.replace("¶", " ").replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _flatten_verbatim(main) -> None:
    """Collapse signatures and code blocks into single text nodes.

    ``get_text(" ")`` would render ``<span>json</span><span>.</span>
    <span>dump</span>(`` as ``json . dump (``, which no longer matches the
    symbol a reader would search for. Rendering those subtrees with no
    separator first keeps ``json.dump(obj, fp, *, skipkeys=False)`` intact,
    while ordinary prose still gets the space-joined treatment.
    """
    from bs4 import NavigableString

    for tag in main.find_all(class_=re.compile(r"^sig(-|$)")):
        if tag.find_parent(class_=re.compile(r"^sig(-|$)")) is None:
            # A signature often spans source lines; fold it back onto one.
            tag.replace_with(NavigableString(re.sub(r"\s+", " ", tag.get_text("")).strip()))
    for tag in main.find_all("pre"):
        # Keep the <pre> element (it is a block the section walker renders) but
        # collapse its markup to one fenced string so indentation survives.
        tag.string = "```\n" + tag.get_text("").strip() + "\n```"


def _heading_of(section):
    for tag in section.find_all(["h1", "h2", "h3", "h4", "h5", "h6"], recursive=False):
        return tag
    return None


def _own_text(container, skip=None) -> str:
    """Text belonging to this section, excluding nested sections.

    Nested sections are indexed in their own right, so including them here
    would store the whole page again under every ancestor heading. Layout
    ``<div>`` wrappers (Sphinx nests code blocks two divs deep) are descended
    through rather than rendered, so their contents are not lost.
    """
    blocks: list[str] = []
    for child in container.find_all(True, recursive=False):
        if child is skip or child.name == "section":
            continue
        if child.name in _WRAPPER_TAGS and child.find(_BLOCK_TAGS):
            nested = _own_text(child)
            if nested:
                blocks.append(nested)
            continue
        if child.name not in _BLOCK_TAGS:
            continue
        block = _clean(child.get_text(" ", strip=True))
        if block:
            blocks.append(block)
    return "\n\n".join(blocks)


def _walk(section, trail: list[str], out: list[Section]) -> None:
    heading_tag = _heading_of(section)
    heading = _clean(heading_tag.get_text(" ", strip=True)) if heading_tag else ""
    breadcrumb = trail + [heading] if heading else trail
    # The heading is carried by the breadcrumb, so don't repeat it in the body.
    text = _own_text(section, skip=heading_tag)
    if text:
        out.append(
            Section(
                heading=" > ".join(breadcrumb),
                anchor=str(section.get("id") or ""),
                text=f"{' > '.join(breadcrumb)}\n\n{text}" if breadcrumb else text,
            )
        )
    for child in section.find_all("section", recursive=False):
        _walk(child, breadcrumb, out)


def parse_html(html: str) -> Document:
    """Extract the prose of a Sphinx page as a list of sections."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    if soup.find("meta", attrs={"http-equiv": re.compile("refresh", re.I)}) or _REDIRECT_MARKER in html:
        return Document(title="", sections=[])

    raw_title = soup.title.get_text(strip=True) if soup.title else ""
    # "json - JSON encoder and decoder - Python 3.14.7 documentation"
    title = _clean(re.sub(r"\s*[-—]\s*Python 3\.[\d.]+\w*\s+documentation\s*$", "", raw_title))

    main = soup.find("div", {"role": "main"}) or soup.find("div", class_="body") or soup.body
    if main is None:
        return Document(title=title, sections=[])

    for tag in main.find_all(_STRIP_TAGS):
        tag.decompose()
    for klass in _STRIP_CLASSES:
        for tag in main.find_all(class_=klass):
            tag.decompose()
    _flatten_verbatim(main)

    sections: list[Section] = []
    top = main.find_all("section", recursive=False) or main.find_all("section")
    for section in top:
        # find_all() without recursive=False can return nested ones too; _walk covers those.
        if section.find_parent("section") is None:
            _walk(section, [], sections)

    if not sections:
        text = _clean(main.get_text(" ", strip=True))
        if text:
            sections = [Section(heading=title, anchor="", text=text)]
    return Document(title=title, sections=sections)


def load(path: Path) -> Document:
    """Read a corpus file as a title plus sections.

    Markdown and plain text are passed through as a single section; the
    existing word-window chunker already handles them well at their size.
    """
    if path.suffix == ".html":
        return parse_html(path.read_text(encoding="utf-8", errors="ignore"))
    text = path.read_text(encoding="utf-8", errors="ignore")
    return Document(title=path.stem, sections=[Section(heading=path.stem, anchor="", text=text)] if text.strip() else [])
