"""Produces a bounded graph from resolved resource records, never model guesses."""
import json
from lfx.custom.custom_component.component import Component
from lfx.io import MessageTextInput, Output
from lfx.schema.message import Message


class EvidenceVisualSpec(Component):
    display_name = "Visual Specification — Evidence-backed Resource Map"
    description = "Builds source-linked nodes and supported directed edges for a requested visual."
    icon = "workflow"
    name = "EvidenceVisualSpec"
    inputs = [MessageTextInput(name="evidence", display_name="Resource Evidence", input_types=["Message"], required=True)]
    outputs = [Output(name="visual", display_name="Validated Visual Specification", method="build_visual")]

    def build_visual(self) -> Message:
        result = {"status": "SKIPPED"}
        try:
            data = json.loads(str(getattr(self.evidence, "text", self.evidence) or ""))
            if data.get("request", {}).get("visual_requested") and data.get("status") in ("RESOLVED", "SUPPORTED"):
                resources = data["resources"]
                if not isinstance(resources, list) or not 1 <= len(resources) <= 8:
                    raise ValueError("Invalid resources")
                ids = {r["id"] for r in resources}
                if len(ids) != len(resources):
                    raise ValueError("Duplicate resources")
                source_map = {s["id"]: s for s in data.get("sources", [])}
                for resource in resources:
                    if not resource.get("evidence_ids") or any(s not in source_map or
                            resource["id"] not in source_map[s].get("resource_ids", []) or
                            not source_map[s].get("passage") for s in resource["evidence_ids"]):
                        raise ValueError("Missing resource identity evidence")
                edges = []
                for edge in data.get("relationships", []):
                    if edge["subject_id"] not in ids or edge["object_id"] not in ids or edge["subject_id"] == edge["object_id"] or not edge.get("evidence_ids") or edge.get("polarity", "supports") != "supports" or edge.get("predicate") not in ("feeds", "uses", "owned_by"):
                        raise ValueError("Invalid edge")
                    for source_id in edge["evidence_ids"]:
                        if source_id not in source_map or not source_map[source_id].get("passage") or not any(
                                isinstance(claim, dict) and all(claim.get(k) == edge.get(k) for k in
                                ("subject_id", "predicate", "object_id", "scope")) and
                                claim.get("polarity", "supports") == "supports"
                                for claim in source_map[source_id].get("assertions", [])):
                            raise ValueError("Missing supported relationship assertion")
                    edges.append({"from": edge["subject_id"], "to": edge["object_id"], "predicate": edge["predicate"],
                                  "status": "SUPPORTED", "evidence_ids": edge["evidence_ids"]})
                if len(edges) > 12:
                    raise ValueError("Too many edges")
                names = {r["id"]: r["name"] for r in resources}
                result = {"status": "READY", "schema_version": 1, "kind": "resource_relationship",
                    "synthetic_example": data.get("synthetic_example") is True,
                    "nodes": [{"id": r["id"], "label": r["name"], "owner": r.get("owner", ""),
                               "evidence_ids": r["evidence_ids"]} for r in resources],
                    "edges": edges, "sources": data.get("sources", []),
                    "scope": data.get("scope", resources[0].get("scope", "")), "checked_at": data.get("checked_at", ""),
                    "text_equivalent": "; ".join(names[e["from"]] + " — " + e["predicate"] + " → " + names[e["to"]]
                        for e in edges) or "; ".join(names.values())}
        except (ValueError, TypeError, KeyError, AttributeError):
            result = {"status": "ERROR", "diagnostic": "Visual evidence could not be validated. Text explanation remains available."}
        return Message(text=json.dumps(result, ensure_ascii=False), session_id=getattr(self.evidence, "session_id", None) or "")
