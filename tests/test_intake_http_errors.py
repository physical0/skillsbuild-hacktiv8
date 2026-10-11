"""Safe diagnostics for rejected Gemini requests; all responses are mocked."""
import json
import types
import unittest
from unittest.mock import AsyncMock, patch

from test_context_flow import ROOT, INTAKE, HTTPStatusError, OfflineAsyncCase, intake, model_client


def error(code=400, body=None):
    exc = HTTPStatusError(code)
    if body is not None:
        exc.response.json = lambda: body
    return exc


class IntakeHttpTests(OfflineAsyncCase):
    async def test_schema_failure_in_playground_identifies_field_and_stops_retrieval(self):
        path = "generation_config.response_schema.properties[relation_predicate].enum"
        exc = error(body={"error": {"status": "INVALID_ARGUMENT", "message": "Schema rejected; sensitive request details omitted",
            "details": [{"fieldViolations": [{"field": path, "description": "private document text"}]}]}})
        c = intake("What does ROI mean?")
        client = model_client(error=exc)
        with patch.object(INTAKE, "aget_messages", AsyncMock(return_value=[])), patch.object(INTAKE.httpx, "AsyncClient", return_value=client):
            self.assertEqual((await c.build_ready()).text, "")
            reply = await c.build_clarification()
            state = await c._state()
        self.assertEqual(state["http_status"], 400)
        self.assertEqual(state["provider_status"], "INVALID_ARGUMENT")
        self.assertEqual(state["diagnostic"], "REQUEST_SCHEMA_REJECTED")
        self.assertEqual(state["invalid_fields"], [path])
        self.assertIn(path, reply.text)
        self.assertIn("not evidence of exhausted quota", reply.text)
        self.assertNotIn("private document text", reply.text)

    async def test_direct_gemini_payload_omits_empty_enum_without_connected_model(self):
        from test_context_flow import extraction
        c = intake("What is API?")
        self.assertIsNone(c.language_model)
        client = model_client(json.dumps(extraction(terms=["API"])))
        with patch.object(INTAKE.httpx, "AsyncClient", return_value=client):
            await c._extract([{"role": "user", "text": "What is API?"}])
        payload = client.post.call_args.kwargs["json"]
        predicate = payload["generationConfig"]["responseSchema"]["properties"]["relation_predicate"]
        self.assertEqual(predicate, {"type": "STRING"})
        self.assertIn("relation_predicate", payload["generationConfig"]["responseSchema"]["required"])
        self.assertEqual(payload["generationConfig"]["responseMimeType"], "application/json")


class IntakeDiagnosticTests(unittest.TestCase):
    def test_api_key_error_does_not_mislabel_schema_or_expose_key(self):
        c = intake()
        c.google_api_key = "private-key-value"
        exc = error(body={"error": {"status": "INVALID_ARGUMENT", "message": "API key not valid: private-key-value",
            "details": [{"reason": "API_KEY_INVALID", "metadata": {"key": "private-key-value"}},
                        {"fieldViolations": [{"field": "private-key-value"}]}]}})
        state = c._gemini_http_failure(exc)
        self.assertEqual(state["diagnostic"], "API_KEY_INVALID")
        self.assertIn("API_KEY_INVALID", state["question"])
        self.assertNotIn("private-key-value", json.dumps(state))

    def test_429_reports_quota_and_400_does_not_assert_it(self):
        for code, expected in ((429, "reported a rate/quota limit"), (400, "does not establish a quota problem")):
            state = intake()._gemini_http_failure(error(code))
            self.assertIn(expected, state["question"])
            self.assertEqual(state["status"], "API_ERROR")

    def test_failed_precondition_is_project_eligibility_not_exhausted_quota(self):
        state = intake()._gemini_http_failure(error(body={"error": {"status": "FAILED_PRECONDITION"}}))
        self.assertIn("eligibility", state["question"])
        self.assertEqual(state["provider_status"], "FAILED_PRECONDITION")

    def test_unknown_or_unreadable_provider_body_keeps_safe_http_diagnostic(self):
        for body in ("private arbitrary text", {"error": "private arbitrary text"}, {"error": {"status": "PRIVATE_STATUS_VALUE", "message": "private arbitrary text"}}):
            state = intake()._gemini_http_failure(error(body=body))
            self.assertEqual(state["http_status"], 400)
            self.assertNotIn("private arbitrary text", json.dumps(state))
            self.assertNotIn("PRIVATE_STATUS_VALUE", json.dumps(state))

    def test_field_paths_are_bounded_filtered_and_not_raw_error_descriptions(self):
        fields = [{"field": "generation_config.response_schema.properties[field" + str(i) + "]"} for i in range(9)]
        fields += [{"field": "<script>private text</script>"}, {"field": "AIza" + "z" * 35}]
        state = intake()._gemini_http_failure(error(body={"error": {"details": [{"fieldViolations": fields}]}}))
        self.assertEqual(len(state["invalid_fields"]), 5)
        self.assertNotIn("<script>", state["question"])
        self.assertNotIn("AIza", state["question"])

    def test_schema_compatibility_change_keeps_predicate_allow_list_enforced(self):
        c = intake()
        with self.assertRaises(ValueError):
            c._resource_plan({"request_type": "RELATIONSHIP", "relation_predicate": "invented"}, "What is API?", {})
        self.assertEqual(c._resource_plan({}, "What is API?", {})["relation_predicate"], "")

    def test_complete_export_contains_diagnostic_and_current_schema(self):
        flow = json.loads((ROOT / "flows/terminology-agent.json").read_text(encoding="utf-8"))
        node = next(n for n in flow["data"]["nodes"] if n["data"]["type"] == "ContextIntake")
        code = node["data"]["node"]["template"]["code"]["value"]
        self.assertIn("def _gemini_http_failure", code)
        self.assertIn('"relation_predicate": {"type": "STRING"}', code)


if __name__ == "__main__":
    unittest.main()
