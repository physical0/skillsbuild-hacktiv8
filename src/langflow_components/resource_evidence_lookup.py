"""Exact resource resolution and bounded, source-backed relationship checks."""
import json
import re
from datetime import date, datetime, timezone

from lfx.custom.custom_component.component import Component
from lfx.io import MessageTextInput, MultilineInput, Output
from lfx.schema.message import Message


class ResourceEvidenceLookup(Component):
    display_name = "Resource Evidence — Registry and Relationship Lookup"
    description = "Resolves resource IDs and checks one documented directed relationship."
    icon = "network"
    name = "ResourceEvidenceLookup"
    inputs = [
        MessageTextInput(name="request", display_name="Resolved Request", input_types=["Message"], required=True),
        MultilineInput(name="catalog_json", display_name="Resource Registry JSON", required=True),
    ]
    outputs = [Output(name="evidence", display_name="Resource Evidence and Status", method="lookup")]

    def _norm(self, value):
        return " ".join(value.casefold().split())

    def _parse(self, value):
        return json.loads(str(getattr(value, "text", value) or ""))

    def _out(self, status, request, **fields):
        self.status = {"status": status, "resources": len(fields.get("resources", []))}
        return Message(text=json.dumps({"status": status, "request": request, "resources": [],
                       "relationships": [], "sources": [], **fields}, ensure_ascii=False),
                       session_id=getattr(self.request, "session_id", None) or "")

    def _catalog(self):
        catalog = self._parse(self.catalog_json)
        if not isinstance(catalog, dict) or catalog.get("version") != 1:
            raise ValueError("Invalid catalog version")
        resources, sources, relations = (catalog.get(k) for k in ("resources", "sources", "relationships"))
        if not all(isinstance(x, list) and len(x) <= 200 for x in (resources, sources, relations)):
            raise ValueError("Invalid registry lists")
        if any(not isinstance(x, dict) or not isinstance(x.get("id"), str) or not x["id"] for x in resources + sources + relations):
            raise ValueError("Missing record IDs")
        if any(len({x["id"] for x in values}) != len(values) for values in (resources, sources, relations)):
            raise ValueError("Duplicate record IDs")
        source_map = {x["id"]: x for x in sources}
        for item in resources:
            if not isinstance(item.get("name"), str) or not item["name"].strip() or len(item["name"]) > 200 or len(item["id"]) > 200:
                raise ValueError("Invalid resource name")
            if not isinstance(item.get("aliases", []), list) or any(not isinstance(x, str) or not x.strip() for x in item.get("aliases", [])):
                raise ValueError("Invalid aliases")
            evidence = item.get("evidence_ids")
            if not isinstance(evidence, list) or not evidence or any(x not in source_map for x in evidence):
                raise ValueError("Unreferenced resource evidence")
            if any(item["id"] not in source_map[s].get("resource_ids", []) or not source_map[s].get("passage") for s in evidence):
                raise ValueError("Resource source does not identify it")
        return catalog, resources, source_map, relations

    def _scan(self, question, resources):
        hits = []
        for item in resources:
            for alias in [item["id"], item["name"], *item.get("aliases", [])]:
                hits.extend((m.start(), m.end(), m.group()) for m in re.finditer(
                    r"(?<!\w)" + re.escape(alias) + r"(?!\w)", question, re.I))
        hits.sort(key=lambda x: (x[0], -(x[1] - x[0])))
        mentions, end = [], -1
        for start, finish, text in hits:
            if start >= end:
                if self._norm(text) not in {self._norm(x) for x in mentions}:
                    mentions.append(text)
                end = finish
        return mentions

    def lookup(self) -> Message:
        request = {}
        try:
            request = self._parse(self.request)
            if not isinstance(request, dict) or request.get("status") != "READY":
                raise ValueError("Request must be READY")
            if request.get("request_type", "TERMINOLOGY") == "TERMINOLOGY":
                return self._out("NOT_REQUESTED", request)
            if len(str(self.catalog_json or "")) > 500000:
                raise ValueError("Catalog too large")
            catalog, resources, source_map, relations = self._catalog()
            question = request["original_question"]
            if not isinstance(question, str) or not question.strip() or len(question) > 12000:
                raise ValueError("Invalid original request")
            mentions = request.get("resource_mentions") or self._scan(question, resources)
            if not isinstance(mentions, list) or len(mentions) > 12 or any(not isinstance(x, str) or not x.strip() for x in mentions):
                raise ValueError("Invalid resource mentions")
            if not mentions:
                return self._out("NOT_FOUND", request, diagnostic="No identifiable resource was supplied.")
            choices = request.get("resource_choices") or {}
            if not isinstance(choices, dict):
                raise ValueError("Invalid resource choices")
            resolved, by_mention, missing = [], {}, []
            for mention in mentions:
                candidates = [r for r in resources if self._norm(mention) in {
                    self._norm(x) for x in [r["id"], r["name"], *r.get("aliases", [])]}]
                if mention in choices:
                    candidates = [r for r in candidates if r["id"] == choices[mention]]
                if len(candidates) > 1:
                    return self._out("AMBIGUOUS", request, ambiguous_mention=mention,
                        candidates=[{k: r.get(k, "") for k in ("id", "name", "owner", "scope")} for r in candidates])
                if not candidates:
                    missing.append(mention)
                    continue
                resource = candidates[0]
                by_mention[self._norm(mention)] = resource
                if resource["id"] not in {r["id"] for r in resolved}:
                    resolved.append(resource)
            used = {s for r in resolved for s in r["evidence_ids"]}
            base = {"resources": resolved, "sources": [source_map[s] for s in sorted(used)],
                    "synthetic_example": catalog.get("synthetic_example") is True,
                    "missing_resource_mentions": missing}
            if missing:
                return self._out("NOT_FOUND", request, **base, diagnostic="Some resource mentions could not be resolved.")
            if request.get("request_type") != "RELATIONSHIP":
                return self._out("RESOLVED", request, **base)
            if len(resolved) < 2:
                return self._out("NEEDS_TARGET", request, **base, diagnostic="Specify the source and target resources to compare.")
            def endpoint(field, fallback):
                mention = request.get(field, "")
                if not mention:
                    return fallback
                item = by_mention.get(self._norm(mention))
                if item is None:
                    item = next((r for r in resolved if self._norm(mention) in {
                        self._norm(x) for x in [r["id"], r["name"], *r.get("aliases", [])]}), None)
                if item is None:
                    raise ValueError("Relation endpoint is not resolved")
                return item
            subject = endpoint("relation_subject", resolved[0])
            target = endpoint("relation_object", resolved[1])
            predicate = request.get("relation_predicate", "")
            if predicate not in ("", "feeds", "uses", "owned_by"):
                raise ValueError("Unsupported relationship predicate")
            checked_at = date.fromisoformat(request["as_of"]) if request.get("as_of") else datetime.now(timezone.utc).date()
            scope = request.get("resource_scope") or subject.get("scope")
            matches = [r for r in relations if r.get("subject_id") == subject["id"] and
                       r.get("object_id") == target["id"] and (not predicate or r.get("predicate") == predicate)]
            supported, denied, excluded = [], [], []
            for relation in matches:
                if relation.get("predicate") not in ("feeds", "uses", "owned_by"):
                    raise ValueError("Invalid relationship type")
                ids = relation.get("evidence_ids")
                polarity = relation.get("polarity", "supports")
                if polarity not in ("supports", "denies") or not isinstance(ids, list) or not ids:
                    raise ValueError("Missing relationship evidence")
                for source_id in ids:
                    source = source_map.get(source_id)
                    if not source or not source.get("passage") or not any(
                        all(claim.get(k) == relation.get(k) for k in ("subject_id", "predicate", "object_id")) and
                        claim.get("polarity", "supports") == polarity and claim.get("scope") == relation.get("scope")
                        for claim in source.get("assertions", []) if isinstance(claim, dict)):
                        raise ValueError("Source assertion does not support the relationship")
                    used.add(source_id)
                if relation.get("scope") != scope or target.get("scope") != scope:
                    excluded.append("OUT_OF_SCOPE")
                    continue
                if (relation.get("valid_from") and checked_at < date.fromisoformat(relation["valid_from"])) or (
                        relation.get("valid_until") and checked_at > date.fromisoformat(relation["valid_until"])):
                    excluded.append("STALE")
                    continue
                (supported if polarity == "supports" else denied).append(relation)
            base.update(sources=[source_map[s] for s in sorted(used)], checked_at=checked_at.isoformat(), scope=scope,
                        checked_subject_id=subject["id"], checked_object_id=target["id"],
                        excluded_statuses=list(dict.fromkeys(excluded)))
            status = "CONFLICTING" if supported and denied else "SUPPORTED" if supported else "CONTRADICTED" if denied else (
                     excluded[0] if excluded else "NOT_ESTABLISHED")
            return self._out(status, request, **base, relationships=supported if status == "SUPPORTED" else [],
                             conflicting_records=supported + denied if status == "CONFLICTING" else [],
                             diagnostic="Checked documented records only; future operational impact is not verified.")
        except (ValueError, TypeError, KeyError, AttributeError):
            return self._out("ERROR", request if isinstance(request, dict) else {},
                             diagnostic="Invalid request, catalog, or evidence contract. No relationship is established.")
