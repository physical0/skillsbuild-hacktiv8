"""Render a validated LintasTeks visual payload as an inert, local SVG.

No dependencies, network, embedded images, HTML, or executable content.
Accepts JSON or a saved tool response containing one lintasteks-visual fence.
"""
import argparse
import json
import re
import textwrap
from html import escape
from pathlib import Path


def parse_payload(text):
    if len(text) > 500000:
        raise ValueError("Visual input exceeds 500,000 characters")
    blocks = re.findall(r"^```lintasteks-visual\s*\n(.*?)^```\s*$", text, re.M | re.S)
    if len(blocks) > 1:
        raise ValueError("Choose one visual payload per file")
    return json.loads(blocks[0] if blocks else text)


def validate(data):
    if not isinstance(data, dict) or data.get("schema_version") != 1 or data.get("status") != "READY" or data.get("kind") != "resource_relationship":
        raise ValueError("Unsupported visual contract")
    nodes, edges, sources = (data.get(k) for k in ("nodes", "edges", "sources"))
    if not isinstance(nodes, list) or not 1 <= len(nodes) <= 8 or not isinstance(edges, list) or len(edges) > 12 or not isinstance(sources, list) or len(sources) > 200:
        raise ValueError("Visual exceeds supported bounds")
    def string(value, limit=200):
        return isinstance(value, str) and bool(value.strip()) and len(value) <= limit
    if any(not isinstance(s, dict) or not string(s.get("id")) or not string(s.get("passage"), 8000) for s in sources):
        raise ValueError("Missing source identity or passage")
    source_map = {s["id"]: s for s in sources}
    if len(source_map) != len(sources):
        raise ValueError("Duplicate source IDs")
    def citations(record):
        ids = record.get("evidence_ids")
        if not isinstance(ids, list) or not ids or len(ids) > 20 or any(not string(s) or s not in source_map for s in ids):
            raise ValueError("Missing evidence citation")
        return ids
    for node in nodes:
        if not isinstance(node, dict) or not string(node.get("id")) or not string(node.get("label")) or not isinstance(node.get("owner", ""), str) or len(node.get("owner", "")) > 200:
            raise ValueError("Invalid node")
        if any(node["id"] not in source_map[s].get("resource_ids", []) for s in citations(node)):
            raise ValueError("Source does not identify this node")
    node_map = {n["id"]: n for n in nodes}
    if len(node_map) != len(nodes):
        raise ValueError("Duplicate node IDs")
    for edge in edges:
        if not isinstance(edge, dict) or edge.get("from") not in node_map or edge.get("to") not in node_map or edge.get("from") == edge.get("to") or edge.get("status") != "SUPPORTED" or edge.get("predicate") not in ("feeds", "uses", "owned_by"):
            raise ValueError("Unsupported or dangling edge")
        for source_id in citations(edge):
            if not any(isinstance(claim, dict) and claim.get("subject_id") == edge["from"] and
                       claim.get("object_id") == edge["to"] and claim.get("predicate") == edge["predicate"] and
                       claim.get("polarity", "supports") == "supports" and claim.get("scope") == data.get("scope")
                       for claim in source_map[source_id].get("assertions", [])):
                raise ValueError("Source does not support this directed edge and scope")
    if not isinstance(data.get("text_equivalent"), str) or len(data["text_equivalent"]) > 8000:
        raise ValueError("Missing text equivalent")
    return node_map


def render_svg(data):
    nodes = validate(data)
    edges = data["edges"]
    source_map = {s["id"]: s for s in data["sources"]}
    source_order = list(dict.fromkeys([s for n in data["nodes"] for s in n["evidence_ids"]] +
                                     [s for e in edges for s in e["evidence_ids"]] + list(source_map)))
    source_numbers = {s: i + 1 for i, s in enumerate(source_order)}
    def refs(ids):
        return ", ".join("[" + str(source_numbers[s]) + "]" for s in ids)
    connected = {i for e in edges for i in (e["from"], e["to"])}
    rows = [(nodes[e["from"]], nodes[e["to"]], e) for e in edges]
    isolated = [n for n in data["nodes"] if n["id"] not in connected]
    rows += [(isolated[i], isolated[i + 1] if i + 1 < len(isolated) else None, None) for i in range(0, len(isolated), 2)]
    # Each documented edge gets its own row: arrows never cross a node card.
    # A resource may repeat across rows, with the same exact identity each time.
    height = 225 + 220 * len(rows) + 55 * len(data["sources"])
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="{height}" viewBox="0 0 1280 {height}" role="img">',
           '<title>LintasTeks resource evidence map</title>', '<desc>' + escape(data["text_equivalent"] + " Resource IDs: " + ", ".join(nodes)) + '</desc>',
           '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto"><path d="M0,0 L10,5 L0,10" fill="#147d79"/></marker></defs>',
           f'<rect width="1280" height="{height}" fill="#f6f9fc"/>']
    def text(x, y, value, size=18, color="#23364b", weight="400"):
        svg.append(f'<text x="{x}" y="{y}" font-family="Segoe UI, Arial, sans-serif" font-size="{size}" font-weight="{weight}" fill="{color}">{escape(str(value))}</text>')
    text(64, 65, "LintasTeks / Resource evidence", 32, weight="700")
    text(64, 100, "Synthetic example — not company data" if data.get("synthetic_example") else "Documented resource relationships", 18, "#63758a")
    text(64, 130, "Scope: " + str(data.get("scope", "unspecified")) + "  |  Checked: " + str(data.get("checked_at", "unspecified")), 16, "#63758a")
    def card(node, x, y):
        svg.append(f'<rect x="{x}" y="{y}" width="440" height="162" rx="14" fill="#ffffff" stroke="#cbd8e5" stroke-width="2"/>')
        text(x + 22, y + 32, node.get("owner") or "Owner unspecified", 15, "#147d79", "600")
        for j, line in enumerate(textwrap.wrap(node["label"], 33)[:2]):
            text(x + 22, y + 65 + 27 * j, line, 22, weight="600")
        text(x + 22, y + 133, "Registry source " + refs(node["evidence_ids"]), 14, "#63758a")
    for i, (left, right, edge) in enumerate(rows):
        y = 170 + i * 220
        card(left, 64, y)
        if right:
            card(right, 776, y)
        if edge:
            svg.append(f'<path d="M518,{y + 85} L756,{y + 85}" stroke="#147d79" stroke-width="3" marker-end="url(#arrow)" fill="none"/>')
            text(555, y + 62, edge["predicate"], 20, "#147d79", "600")
            text(555, y + 118, "Source " + refs(edge["evidence_ids"]), 14, "#63758a")
        text(64, y + 194, "Documented relationship; operational impact remains conditional." if edge else "Resource identities only; no relationship is asserted.", 15, "#63758a")
    bottom = 175 + len(rows) * 220
    text(64, bottom, "Evidence references", 20, weight="600")
    for i, source_id in enumerate(source_order):
        source = source_map[source_id]
        y = bottom + 30 + i * 55
        shared_demo_status = data.get("synthetic_example") and all(s.get("approval_status") == "demo_only" for s in data["sources"])
        text(64, y, refs([source_id]) + " " + source["id"] + ("" if shared_demo_status else " · " + str(source.get("approval_status", "status unspecified"))), 14, "#147d79", "600")
        # Full source passages stay verbatim in JSON; this is a compact index.
        linked_edges = [e for e in edges if source_id in e["evidence_ids"]]
        if linked_edges:
            caption = "; ".join(nodes[e["from"]]["label"] + " " + e["predicate"] + " " + nodes[e["to"]]["label"] for e in linked_edges)
        else:
            caption = "Identity record: " + ", ".join(nodes[n]["label"] for n in source.get("resource_ids", []) if n in nodes)
        text(64, y + 20, textwrap.shorten(caption, width=150, placeholder="…"), 13, "#63758a")
    svg.append('</svg>')
    return "\n".join(svg) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        svg = render_svg(parse_payload(args.input.read_text(encoding="utf-8-sig")))
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        parser.exit(2, "Cannot render visual: " + str(exc) + "\n")
    if args.output.suffix.lower() != ".svg":
        parser.exit(2, "Output must be an SVG file\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg, encoding="utf-8")
    print("Saved " + str(args.output.resolve()))


if __name__ == "__main__":
    main()
