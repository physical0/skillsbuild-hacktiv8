"""Offline registry, evidence, conversation-resume, export and SVG checks."""
import copy
import importlib.util
import json
import unittest
import xml.etree.ElementTree as ET
from unittest.mock import AsyncMock, patch

from test_context_flow import ROOT, MODULES, Message, OfflineAsyncCase, OfflineLoop, extraction, intake, model_client

spec = importlib.util.spec_from_file_location("render_resource_map", ROOT / "src/visualization/render_resource_map.py")
RENDER = importlib.util.module_from_spec(spec)
spec.loader.exec_module(RENDER)
CATALOG = json.loads((ROOT / "demo/data/resource-catalog.json").read_text(encoding="utf-8"))


def request(**changes):
    return {"status": "READY", "original_question": "I’m from Finance. If we change the billing pipeline, could it affect the invoice preparation report? Show a diagram.",
        "context": {"recipient_department": "Finance"}, "language": "en", "context_selected": True,
        "context_selection": {"recipient_department": "Finance", "recipient_departments": ["Finance"]},
        "request_type": "RELATIONSHIP", "terms": [], "resource_mentions": ["billing pipeline", "invoice preparation report"],
        "relation_subject": "billing pipeline", "relation_object": "invoice preparation report", "relation_predicate": "feeds",
        "visual_requested": True, "as_of": "2026-10-11", **changes}


def lookup(state=None, catalog=None):
    return MODULES["resource_evidence_lookup"].ResourceEvidenceLookup(
        request=Message(text=json.dumps(state if state is not None else request()), session_id="resource-session"),
        catalog_json=json.dumps(catalog if catalog is not None else CATALOG)).lookup()


def visual(evidence):
    return MODULES["evidence_visual_spec"].EvidenceVisualSpec(evidence=evidence).build_visual()


def ambiguous_catalog():
    catalog = copy.deepcopy(CATALOG)
    item = copy.deepcopy(catalog["resources"][0])
    item.update(id="DEMO-RESOURCE-BILLING-SECOND", name="Demo second billing pipeline", owner="Operations", evidence_ids=["DEMO-RES-SECOND"])
    catalog["resources"].append(item)
    catalog["sources"].append({"id": "DEMO-RES-SECOND", "resource_ids": [item["id"]], "passage": "Synthetic second resource."})
    return catalog


def compose(evidence, answer="Documented demo relationship [DEMO-REL-SOURCE-001]. Possible impact remains conditional."):
    return MODULES["response_composer"].ResponseComposer(
        answer=Message(text=answer, session_id="resource-session", session_metadata={"lintasteks": {"context_selected": True}}),
        evidence=evidence, visual=visual(evidence)).compose()


class ResourceTests(unittest.TestCase):
    def test_supported_edge_preserves_identity_scope_date_and_source(self):
        result = json.loads(lookup().text)
        self.assertEqual(result["status"], "SUPPORTED")
        self.assertEqual(result["scope"], "demo")
        self.assertEqual(result["checked_at"], "2026-10-11")
        self.assertEqual(result["relationships"][0]["id"], "DEMO-REL-001")
        self.assertEqual(len(result["sources"]), 3)

    def test_exact_names_aliases_and_ids_resolve_same_object(self):
        for mention in ("billing pipeline", "BILLING PIPELINE", "Demo billing data pipeline", "DEMO-RESOURCE-BILLING"):
            result = json.loads(lookup(request(request_type="RESOURCE", resource_mentions=[mention])).text)
            self.assertEqual(result["status"], "RESOLVED")
            self.assertEqual(result["resources"][0]["id"], "DEMO-RESOURCE-BILLING")

    def test_scan_fallback_uses_whole_names(self):
        result = json.loads(lookup(request(resource_mentions=[])).text)
        self.assertEqual(result["status"], "SUPPORTED")
        self.assertEqual(len(result["resources"]), 2)

    def test_terminology_skips_registry_even_if_invalid(self):
        result = json.loads(lookup(request(request_type="TERMINOLOGY"), {}).text)
        self.assertEqual(result["status"], "NOT_REQUESTED")

    def test_ambiguity_lists_candidates_without_guessing(self):
        result = json.loads(lookup(catalog=ambiguous_catalog()).text)
        self.assertEqual(result["status"], "AMBIGUOUS")
        self.assertEqual(len(result["candidates"]), 2)
        self.assertEqual(result["relationships"], [])

    def test_choice_must_belong_to_alias_candidates(self):
        for choice, expected in (("DEMO-RESOURCE-BILLING", "SUPPORTED"), ("DEMO-RESOURCE-INVOICE-REPORT", "NOT_FOUND")):
            result = json.loads(lookup(request(resource_choices={"billing pipeline": choice}), ambiguous_catalog()).text)
            self.assertEqual(result["status"], expected)

    def test_missing_resource_and_target_have_different_statuses(self):
        self.assertEqual(json.loads(lookup(request(resource_mentions=["unknown resource"])).text)["status"], "NOT_FOUND")
        self.assertEqual(json.loads(lookup(request(resource_mentions=["billing pipeline"], relation_object="")).text)["status"], "NEEDS_TARGET")

    def test_reverse_or_absent_edge_is_not_established(self):
        reverse = request(relation_subject="invoice preparation report", relation_object="billing pipeline")
        empty = copy.deepcopy(CATALOG)
        empty["relationships"] = []
        for state, catalog in ((reverse, CATALOG), (request(), empty)):
            result = json.loads(lookup(state, catalog).text)
            self.assertEqual(result["status"], "NOT_ESTABLISHED")
            self.assertEqual(result["relationships"], [])

    def test_stale_and_out_of_scope_do_not_draw_edge(self):
        expired = copy.deepcopy(CATALOG)
        expired["relationships"][0]["valid_until"] = "2026-10-10"
        for state, catalog, expected in ((request(), expired, "STALE"), (request(resource_scope="production"), CATALOG, "OUT_OF_SCOPE")):
            message = lookup(state, catalog)
            self.assertEqual(json.loads(message.text)["status"], expected)
            self.assertEqual(json.loads(visual(message).text)["status"], "SKIPPED")

    def test_denied_and_conflicting_assertions_are_not_positive_edges(self):
        denied = copy.deepcopy(CATALOG)
        denied["relationships"][0]["polarity"] = "denies"
        denied["sources"][-1]["assertions"][0]["polarity"] = "denies"
        self.assertEqual(json.loads(lookup(catalog=denied).text)["status"], "CONTRADICTED")
        conflict = copy.deepcopy(CATALOG)
        deny = copy.deepcopy(denied["relationships"][0])
        deny.update(id="DENIAL", evidence_ids=["DENIAL-SOURCE"])
        source = copy.deepcopy(denied["sources"][-1])
        source["id"] = "DENIAL-SOURCE"
        conflict["relationships"].append(deny)
        conflict["sources"].append(source)
        result = json.loads(lookup(catalog=conflict).text)
        self.assertEqual(result["status"], "CONFLICTING")
        self.assertEqual(result["relationships"], [])

    def test_missing_or_wrong_source_assertion_is_error(self):
        for field, value in (("assertions", []), ("passage", "")):
            bad = copy.deepcopy(CATALOG)
            bad["sources"][-1][field] = value
            self.assertEqual(json.loads(lookup(catalog=bad).text)["status"], "ERROR")

    def test_duplicate_registry_ids_are_error(self):
        bad = copy.deepcopy(CATALOG)
        bad["resources"].append(bad["resources"][0])
        self.assertEqual(json.loads(lookup(catalog=bad).text)["status"], "ERROR")

    def test_resource_only_request_does_not_query_glossary(self):
        c = MODULES["term_router"].TermRouter(user_query=Message(text=json.dumps(request())))
        loop = OfflineLoop()
        try:
            self.assertEqual(loop.run_until_complete(c._prepare())["target_terms"], [])
        finally:
            loop.close()


class VisualTests(unittest.TestCase):
    def test_tool_text_round_trip_render_has_same_ids_and_edge_sources(self):
        reply = compose(lookup())
        data = RENDER.parse_payload(reply.text)
        svg = RENDER.render_svg(data)
        ET.fromstring(svg)
        for node in data["nodes"]:
            self.assertIn(node["id"], svg)
        self.assertIn("DEMO-REL-SOURCE-001", svg)
        self.assertIn("feeds", svg)
        self.assertIn("Synthetic example — not company data", svg)
        self.assertNotIn("Demo billing data pipeline", svg)
        self.assertNotIn("· demo_only", svg)
        self.assertIn("Possible impact remains conditional", reply.text)

    def test_no_visual_requested_preserves_text(self):
        reply = compose(lookup(request(visual_requested=False)), "Existing explanation.")
        self.assertEqual(reply.text, "Existing explanation.")

    def test_invalid_visual_retains_explanation(self):
        c = MODULES["response_composer"].ResponseComposer(answer=Message(text="Useful explanation."),
            evidence=lookup(), visual=Message(text='{"status":"ERROR"}'))
        self.assertIn("Useful explanation.", c.compose().text)

    def test_renderer_rejects_unsupported_dangling_or_uncited_edge(self):
        original = json.loads(visual(lookup()).text)
        for field, value in (("status", "NOT_ESTABLISHED"), ("to", "UNKNOWN"), ("evidence_ids", []), ("from", original["edges"][0]["to"])):
            bad = copy.deepcopy(original)
            bad["edges"][0][field] = value
            with self.assertRaises(ValueError):
                RENDER.render_svg(bad)

    def test_renderer_checks_assertion_not_just_source_id(self):
        bad = json.loads(visual(lookup()).text)
        next(s for s in bad["sources"] if s["id"] == "DEMO-REL-SOURCE-001")["assertions"][0]["predicate"] = "uses"
        with self.assertRaises(ValueError):
            RENDER.render_svg(bad)

    def test_renderer_escapes_untrusted_labels(self):
        data = json.loads(visual(lookup()).text)
        data["nodes"][0]["label"] = '<script>alert("x")</script>'
        svg = RENDER.render_svg(data)
        self.assertNotIn("<script>", svg)
        self.assertIn("&lt;script&gt;", svg)

    def test_single_resource_is_identity_visual_without_edge(self):
        data = json.loads(visual(lookup(request(request_type="RESOURCE", resource_mentions=["billing pipeline"]))).text)
        self.assertEqual(data["edges"], [])
        self.assertIn("no relationship is asserted", RENDER.render_svg(data))

    def test_visual_spec_rejects_forged_source_assertion(self):
        data = json.loads(lookup().text)
        next(s for s in data["sources"] if s["id"] == "DEMO-REL-SOURCE-001")["assertions"] = []
        self.assertEqual(json.loads(visual(Message(text=json.dumps(data))).text)["status"], "ERROR")


class ResourceConversationTests(OfflineAsyncCase):
    async def test_numeric_choice_resumes_original_request_without_model(self):
        reply = compose(lookup(catalog=ambiguous_catalog()))
        c = intake("1", "resource-session")
        c._memory = reply.session_metadata["lintasteks"]
        c._conversation = AsyncMock(return_value=[{"role": "user", "text": "1"}])
        c._extract = AsyncMock()
        result = json.loads((await c.build_ready()).text)
        self.assertEqual(result["original_question"], request()["original_question"])
        self.assertEqual(result["context"]["recipient_department"], "Finance")
        self.assertEqual(result["language"], "en")
        self.assertEqual(result["resource_choices"]["billing pipeline"], "DEMO-RESOURCE-BILLING")
        c._extract.assert_not_awaited()
        self.assertEqual(json.loads(lookup(result, ambiguous_catalog()).text)["status"], "SUPPORTED")

    async def test_bad_choice_and_greeting_keep_pending_resource(self):
        reply = compose(lookup(catalog=ambiguous_catalog()))
        for text, expected in (("99", "RESOURCE_CLARIFY"), ("hello", "CONVERSATION")):
            c = intake(text)
            c._memory = reply.session_metadata["lintasteks"]
            c._conversation = AsyncMock(return_value=[{"role": "user", "text": text}])
            response = await c.build_clarification()
            state = response.session_metadata["lintasteks"]
            self.assertEqual(state["status"], expected)
            self.assertTrue(state["pending_resource_request"])

    async def test_exact_id_choice_and_new_question_have_different_paths(self):
        reply = compose(lookup(catalog=ambiguous_catalog()))
        c = intake("DEMO-RESOURCE-BILLING")
        c._memory = reply.session_metadata["lintasteks"]
        c._conversation = AsyncMock(return_value=[{"role": "user", "text": "DEMO-RESOURCE-BILLING"}])
        c._extract = AsyncMock()
        result = json.loads((await c.build_ready()).text)
        self.assertEqual(result["resource_choices"]["billing pipeline"], "DEMO-RESOURCE-BILLING")
        c._extract.assert_not_awaited()
        fresh = intake("What is API?")
        fresh._memory = reply.session_metadata["lintasteks"]
        self.assertIsNone(fresh._resource_resume([{"role": "user", "text": "What is API?"}]))
        result = fresh._resolve(extraction(terms=["API"]), [{"role": "user", "text": "What is API?"}])
        self.assertEqual(result["original_question"], "What is API?")
        self.assertIsNone(result["pending_resource_request"])

    async def test_extractor_cannot_invent_resource_mentions_or_scope(self):
        original = request()["original_question"]
        for changes in ({"resource_mentions": ["imagined resource"]}, {"resource_scope": "production"}, {"as_of": "2030-01-01"}):
            c = intake(original)
            with self.assertRaises(ValueError):
                c._resolve(extraction(request_type="RELATIONSHIP", **changes), [{"role": "user", "text": original}])

    async def test_change_context_clears_pending_resource(self):
        reply = compose(lookup(catalog=ambiguous_catalog()))
        c = intake("change context")
        c._memory = reply.session_metadata["lintasteks"]
        c._conversation = AsyncMock(return_value=[{"role": "user", "text": "change context"}])
        state = (await c.build_clarification()).session_metadata["lintasteks"]
        self.assertIsNone(state["pending_resource_request"])

    async def test_supervisor_separates_documented_relationship_and_conditional_effect(self):
        module = MODULES["context_supervisor"]
        state = request()
        supervisor = module.ContextSupervisor(original_query=Message(text=json.dumps(state)), resource_evidence=lookup(),
            engineering=Message(text='{"status":"SKIPPED"}'), finance=Message(text='{"status":"SKIPPED"}'),
            marketing=Message(text='{"status":"SKIPPED"}'), google_api_key="mock-key")
        client = model_client("Source-backed explanation; possible effect is conditional.")
        with patch.object(module.httpx, "AsyncClient", return_value=client):
            result = await supervisor.run_supervisor()
        self.assertIn("Source-backed", result.text)
        payload = client.post.call_args.kwargs["json"]
        prompt = payload["contents"][0]["parts"][0]["text"]
        self.assertIn("DEMO-REL-SOURCE-001", prompt)
        self.assertIn("never claim the change happened", prompt)
        self.assertIn("NOT_ESTABLISHED means no supporting record", prompt)

    async def test_missing_resource_does_not_discard_found_terminology(self):
        module = MODULES["context_supervisor"]
        state = request(terms=["API"], resource_mentions=["unknown resource"])
        supervisor = module.ContextSupervisor(original_query=Message(text=json.dumps(state)), resource_evidence=lookup(state),
            engineering=Message(text='{"status":"FOUND","query":"API","answer":"Supported API definition","sources":[{"id":"ENGINEERING-001"}]}'),
            finance=Message(text='{"status":"SKIPPED"}'), marketing=Message(text='{"status":"SKIPPED"}'), google_api_key="mock-key")
        client = model_client("API explanation [ENGINEERING-001]; resource unresolved.")
        with patch.object(module.httpx, "AsyncClient", return_value=client):
            result = await supervisor.run_supervisor()
        self.assertIn("ENGINEERING-001", result.text)
        prompt = client.post.call_args.kwargs["json"]["contents"][0]["parts"][0]["text"]
        self.assertIn("NOT_FOUND", prompt)
        self.assertIn("Supported API definition", prompt)

    async def test_missing_resource_wire_has_configuration_status(self):
        module = MODULES["context_supervisor"]
        supervisor = module.ContextSupervisor(original_query=Message(text=json.dumps(request())),
            engineering=Message(text='{"status":"SKIPPED"}'), finance=Message(text='{"status":"SKIPPED"}'),
            marketing=Message(text='{"status":"SKIPPED"}'))
        self.assertIn("MISSING_INPUT", (await supervisor.run_supervisor()).text)

    async def test_resource_extraction_keeps_verbatim_names_and_role(self):
        original = request()["original_question"]
        c = intake(original)
        result = c._resolve(extraction(recipient_department="Finance", request_type="RELATIONSHIP",
            resource_mentions=["billing pipeline", "invoice preparation report"], relation_subject="billing pipeline",
            relation_object="invoice preparation report", relation_predicate="feeds", visual_requested=True),
            [{"role": "user", "text": original}])
        self.assertEqual(result["status"], "READY")
        self.assertEqual(result["context"]["recipient_department"], "Finance")
        self.assertTrue(result["visual_requested"])
        self.assertEqual(result["resource_mentions"], request()["resource_mentions"])


class ResourceExportTests(unittest.TestCase):
    def test_new_nodes_registry_and_final_composition_are_in_complete_export(self):
        flow = json.loads((ROOT / "flows/terminology-agent.json").read_text(encoding="utf-8"))
        nodes, edges = flow["data"]["nodes"], flow["data"]["edges"]
        self.assertEqual((len(nodes), len(edges)), (20, 35))
        by_kind = {n["data"]["type"]: n for n in nodes}
        embedded = by_kind["ResourceEvidenceLookup"]["data"]["node"]["template"]["catalog_json"]["value"]
        self.assertEqual(json.loads(embedded), CATALOG)
        composer = by_kind["ResponseComposer"]["id"]
        self.assertEqual({e["data"]["targetHandle"]["fieldName"] for e in edges if e["target"] == composer}, {"answer", "evidence", "visual"})
        self.assertTrue(any(e["source"] == composer and e["data"]["targetHandle"]["fieldName"] == "input_value" for e in edges))


if __name__ == "__main__":
    unittest.main()
