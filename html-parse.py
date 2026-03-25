from bs4 import BeautifulSoup
import os
import json
import re
from collections import Counter

SAVE_DIR = "data/raw_html/ecfr"
RAW_HTML_PATH = f"{SAVE_DIR}/section_668_32.html"
JSON_OUTPUT_PATH = f"{SAVE_DIR}/section_668_32.json"
TEXT_OUTPUT_PATH = f"{SAVE_DIR}/section_668_32.txt"
SECTION_ID_RE = re.compile(r"^p-(?P<section>\d+(?:\.\d+)+)(?P<suffix>(?:\([^)]+\))*)$")
HEADING_SECTION_RE = re.compile(r"§\s*(\d+(?:\.\d+)+)")


def clean_text(text):
    text = " ".join(text.split())

    # Fix marker spacing: ( a ) ( 1 ) ( i ) -> (a) (1) (i)
    text = re.sub(r"\(\s*([^)]+?)\s*\)", lambda m: f"({m.group(1).strip()})", text)

    # Remove spaces before punctuation: "part ." -> "part."
    text = re.sub(r"\s+([,;:.])", r"\1", text)

    return text


def parse_id_parts(node_id):
    if not node_id:
        return None, []

    match = SECTION_ID_RE.match(node_id)
    if not match:
        return None, []

    section = match.group("section")
    suffix = match.group("suffix")
    parts = re.findall(r"\(([^)]+)\)", suffix)
    return section, parts


def build_subsection_tree(paragraphs):
    root = {"id": None, "label": None, "text": "", "children": []}
    path_index = {(): root}

    for item in paragraphs:
        node_id = item.get("id")
        text = item.get("text", "")
        _, parts = parse_id_parts(node_id)

        path = ()
        for part in parts:
            parent = path_index[path]
            path = (*path, part)
            if path not in path_index:
                node = {"id": None, "label": part, "text": "", "children": []}
                parent["children"].append(node)
                path_index[path] = node

        target = path_index[path]
        target["id"] = node_id
        target["text"] = text

    return root["children"]


def extract_section_content(html):
    soup = BeautifulSoup(html, "lxml")

    # Remove junk
    for tag in soup(["script", "style", "nav"]):
        tag.decompose()

    heading = ""
    heading_section = None
    for h in soup.find_all(["h1", "h2", "h3", "h4", "h5"]):
        h_text = clean_text(h.get_text(" ", strip=True))
        match = HEADING_SECTION_RE.search(h_text)
        if match:
            heading = h_text
            heading_section = match.group(1)
            break

    id_nodes = []
    section_counter = Counter()
    for node in soup.select('[id^="p-"]'):
        node_id = node.get("id")
        section, _ = parse_id_parts(node_id)
        if section:
            id_nodes.append(node)
            section_counter[section] += 1

    target_section = heading_section
    if not target_section and section_counter:
        target_section = section_counter.most_common(1)[0][0]

    paragraphs = []
    for node in id_nodes:
        node_id = node.get("id")
        section, _ = parse_id_parts(node_id)
        if target_section and section != target_section:
            continue

        text = clean_text(node.get_text(" ", strip=True))
        if text:
            paragraphs.append(
                {
                    "id": node_id,
                    "text": text,
                }
            )

    # Fallback if the page structure changes and IDs are not found
    if not paragraphs:
        article = soup.find("article") or soup
        for p in article.find_all("p"):
            text = clean_text(p.get_text(" ", strip=True))
            if text and "ECFR CONTENT" not in text:
                paragraphs.append({"id": None, "text": text})

    subsections = build_subsection_tree(paragraphs)

    return {
        "section": target_section or "",
        "heading": heading,
        "paragraphs": paragraphs,
        "subsections": subsections,
    }


def parse_saved_html():
    if not os.path.exists(RAW_HTML_PATH):
        raise FileNotFoundError(
            f"Raw HTML file not found: {RAW_HTML_PATH}. Run html-scrap.py first."
        )

    with open(RAW_HTML_PATH, "r", encoding="utf-8") as f:
        html = f.read()

    data = extract_section_content(html)

    with open(JSON_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    with open(TEXT_OUTPUT_PATH, "w", encoding="utf-8") as f:
        if data["heading"]:
            f.write(data["heading"] + "\n\n")
        for item in data["paragraphs"]:
            if item["id"]:
                f.write(f"[{item['id']}]\n")
            f.write(item["text"] + "\n\n")

    print(f"Saved parsed JSON to: {JSON_OUTPUT_PATH}")
    print(f"Saved parsed text to: {TEXT_OUTPUT_PATH}")


if __name__ == "__main__":
    parse_saved_html()
