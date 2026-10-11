"""Provider-neutral interface checks; no provider SDK or live model is called."""
import json
import types
import unittest
from unittest.mock import Mock, patch

from test_context_flow import ROOT, MODULES, Message, OfflineAsyncCase, extraction, intake


class ConnectedModel:
    def __init__(self, text="Supported explanation [ENGINEERING-001].", structured=None, error=None, supports_schema=True):
        self.text, self.structured, self.error, self.supports_schema = text, structured, error, supports_schema
        self.calls, self.schemas = [], []

    def with_structured_output(self, schema):
        if not self.supports_schema:
            raise NotImplementedError("Unsupported provider details must not be exposed")
        self.schemas.append(schema)
        async def invoke(messages):
            self.calls.append(messages)
            if self.error:
                raise self.error
            return self.structured
        return types.SimpleNamespace(ainvoke=invoke)

    async def ainvoke(self, messages):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return types.SimpleNamespace(content=self.text, response_metadata={})


def specialist(model, batch=False):
    documents = [{"text": "API definition.", "metadata": {"source_id": "ENGINEERING-001", "term": "API"}}]
    payload = {"status": "FOUND", "documents": documents}
    if batch:
        payload = {"status": "BATCH", "term_results": [{"status": "FOUND", "query": "api", "documents": documents}]}
    return MODULES["domain_term_agent"].DomainTermAgent(query=Message(text="api"), department="Engineering", language_model=model,
        request_context=Message(text=json.dumps({"original_question": "What is API?", "language": "en", "context": {"recipient_department": "Finance"}})),
        retrieved=Message(text=json.dumps(payload)), google_api_key="")


class ModelConnectionTests(OfflineAsyncCase):
    async def test_intake_uses_connected_schema_model_without_gemini_key(self):
        module = MODULES["context_intake"]
        text = "I’m from Engineering. What is ROI?"
        c = intake(text)
        c.google_api_key = ""
        c.language_model = ConnectedModel(structured=extraction(terms=["ROI"], recipient_department="Engineering"))
        with patch.object(module.httpx, "AsyncClient", Mock(side_effect=AssertionError("Direct Gemini must not run"))):
            result = await c._extract([{"role": "user", "text": text}])
        self.assertEqual(result["status"], "READY")
        self.assertEqual(result["context"]["recipient_department"], "Engineering")
        self.assertEqual(result["original_question"], text)
        self.assertEqual(result["terms"], ["ROI"])
        self.assertEqual(c.language_model.calls[0][0][0], "system")
        self.assertIn(text, c.language_model.calls[0][1][1])

    async def test_schema_conversion_changes_types_but_not_enums(self):
        c = intake("What is API?")
        c.language_model = ConnectedModel(structured=extraction(terms=["API"]))
        await c._extract([{"role": "user", "text": "What is API?"}])
        schema = c.language_model.schemas[0]
        self.assertEqual(schema["type"], "object")
        self.assertEqual(schema["properties"]["terms"]["type"], "array")
        self.assertEqual(schema["properties"]["terms"]["items"]["type"], "string")
        self.assertIn("TERMINOLOGY", schema["properties"]["message_kind"]["enum"])
        self.assertEqual(schema["required"], c.schema["required"])
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(c.schema["type"], "OBJECT")

    async def test_single_term_native_model_preserves_sources_and_language(self):
        c = specialist(ConnectedModel())
        with patch.object(MODULES["domain_term_agent"].httpx, "AsyncClient", Mock(side_effect=AssertionError("No Gemini request"))):
            result = json.loads((await c.run_specialist()).text)
        self.assertEqual(result["status"], "FOUND")
        self.assertEqual(result["sources"][0]["id"], "ENGINEERING-001")
        self.assertIn("English", c.language_model.calls[0][0][1])
        self.assertIn("Finance", c.language_model.calls[0][1][1])

    async def test_batch_native_model_still_validates_each_term(self):
        model = ConnectedModel(structured={"findings": [{"term": "API", "status": "FOUND", "answer": "Definition [ENGINEERING-001]."}]})
        c = specialist(model, batch=True)
        with patch.object(MODULES["domain_term_agent"].httpx, "AsyncClient", Mock(side_effect=AssertionError("No Gemini request"))):
            result = json.loads((await c.run_specialist()).text)
        self.assertEqual(result["status"], "BATCH")
        self.assertEqual(result["term_findings"][0]["status"], "FOUND")
        self.assertEqual(result["term_findings"][0]["query"], "api")
        self.assertEqual(model.schemas[0]["properties"]["findings"]["items"]["properties"]["term"]["enum"], ["api"])

    async def test_batch_native_model_invalid_findings_are_not_accepted(self):
        c = specialist(ConnectedModel(structured={"findings": [{"term": "api", "status": "FOUND", "answer": ""}]}), batch=True)
        result = json.loads((await c.run_specialist()).text)
        self.assertEqual(result["term_findings"][0]["status"], "MODEL_MALFORMED")
        self.assertEqual(result["term_findings"][0]["validation_error"], "EMPTY_DEFINITION")

    async def test_supervisor_native_model_preserves_instructions_and_context(self):
        module = MODULES["context_supervisor"]
        model = ConnectedModel(text="Final explanation [ENGINEERING-001].")
        c = module.ContextSupervisor(language_model=model, google_api_key="", original_query=Message(text=json.dumps({
            "status": "READY", "language": "en", "context": {"recipient_department": "Finance"}, "original_question": "What is API?"})),
            engineering=Message(text='{"status":"FOUND","query":"API","answer":"API","sources":[{"id":"ENGINEERING-001"}]}'),
            finance=Message(text='{"status":"SKIPPED"}'), marketing=Message(text='{"status":"SKIPPED"}'))
        with patch.object(module.httpx, "AsyncClient", Mock(side_effect=AssertionError("No Gemini request"))):
            result = await c.run_supervisor()
        self.assertEqual(result.text, "Final explanation [ENGINEERING-001].")
        self.assertIn("Definition:", model.calls[0][0][1])
        self.assertIn("English", model.calls[0][0][1])
        self.assertIn("Finance", model.calls[0][1][1])

    async def test_native_failure_does_not_use_existing_gemini_key_or_leak_details(self):
        c = specialist(ConnectedModel(error=RuntimeError("private-key-secret and private request text")))
        c.google_api_key = "configured-legacy-key"
        with patch.object(MODULES["domain_term_agent"].httpx, "AsyncClient", Mock(side_effect=AssertionError("Must not switch providers"))):
            result = json.loads((await c.run_specialist()).text)
        self.assertEqual(result["status"], "API_ERROR")
        self.assertIn("RuntimeError", result["reason"])
        self.assertNotIn("private-key-secret", result["reason"])
        self.assertNotIn("configured-legacy-key", result["reason"])

    async def test_unsupported_structured_model_has_clear_configuration_status(self):
        c = intake("What is API?")
        c.language_model = ConnectedModel(supports_schema=False)
        with patch.object(MODULES["context_intake"].httpx, "AsyncClient", Mock(side_effect=AssertionError("No fallback"))):
            result = await c._extract([{"role": "user", "text": "What is API?"}])
        self.assertEqual(result["status"], "CONFIGURATION_ERROR")
        self.assertIn("structured-output", result["question"])
        self.assertNotIn("provider details", result["question"])
        self.assertEqual(c.language_model.calls, [])

    async def test_wrong_text_output_connection_is_not_treated_as_model(self):
        c = specialist(Message(text="Text output is not a model object"))
        result = json.loads((await c.run_specialist()).text)
        self.assertEqual(result["status"], "CONFIGURATION_ERROR")
        self.assertIn("LanguageModel output", result["reason"])

    async def test_native_content_blocks_ignore_thinking_and_keep_answer(self):
        c = specialist(ConnectedModel(text=[{"type": "thinking", "text": "PRIVATE_THINKING"}, {"type": "text", "text": "Supported explanation."}]))
        result = json.loads((await c.run_specialist()).text)
        self.assertEqual(result["answer"], "Supported explanation.")
        self.assertNotIn("PRIVATE_THINKING", result["answer"])

    async def test_native_empty_text_is_not_claimed_as_success(self):
        result = json.loads((await specialist(ConnectedModel(text="")).run_specialist()).text)
        self.assertEqual(result["status"], "MODEL_EMPTY")
        self.assertNotIn("Gemini", result["reason"])

    async def test_native_schema_response_requires_object(self):
        result = json.loads((await specialist(ConnectedModel(structured="unstructured text"), batch=True).run_specialist()).text)
        self.assertEqual(result["term_findings"][0]["status"], "MODEL_MALFORMED")

    async def test_shared_model_is_not_replaced_by_bound_schema_runnable(self):
        model = ConnectedModel(structured={"findings": [{"term": "api", "status": "FOUND", "answer": "Definition."}]})
        c = specialist(model, batch=True)
        await c.run_specialist()
        self.assertIs(c.language_model, model)
        self.assertEqual((await specialist(model).run_specialist()).text.count('"status": "FOUND"'), 1)


class ModelPortExportTests(unittest.TestCase):
    def test_all_five_ai_instances_have_model_ports_and_optional_direct_keys(self):
        flow = json.loads((ROOT / "flows/terminology-agent.json").read_text(encoding="utf-8"))
        ai_nodes = [n for n in flow["data"]["nodes"] if n["data"]["type"] in ("ContextIntake", "DomainTermAgent", "ContextSupervisor")]
        self.assertEqual(len(ai_nodes), 5)
        for node in ai_nodes:
            fields = node["data"]["node"]["template"]
            self.assertEqual(fields["language_model"]["input_types"], ["LanguageModel"])
            self.assertEqual(fields["language_model"]["type"], "other")
            self.assertIsNone(fields["language_model"]["value"])
            self.assertFalse(fields["google_api_key"]["required"])
            self.assertEqual(fields["google_api_key"]["value"], "")
