"""Offline behavior checks with model, memory, and Langflow interfaces mocked.

These tests do not establish Gemini quality, Langflow scheduling, or Astra access.
Run: python -m unittest discover -s tests -v
"""
import asyncio
import importlib.util
import json
import selectors
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
from uuid import UUID

ROOT = Path(__file__).resolve().parents[1]


class Message:
    def __init__(self, text="", **kwargs):
        self.text = text
        for key, value in kwargs.items():
            setattr(self, key, value)


class Input:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class Component:
    def __init__(self, **kwargs):
        for field in self.inputs:
            setattr(self, field.name, getattr(field, "value", None))
        self.__dict__.update(kwargs)
        self.stopped = set()
        self._id = "ContextIntake-LintasTeks"

    def stop(self, name):
        self.stopped.add(name)

    def log(self, text):
        pass


class HTTPStatusError(Exception):
    def __init__(self, status):
        self.response = types.SimpleNamespace(status_code=status)


class RequestError(Exception):
    pass


def load_components():
    names = ["lfx", "lfx.custom", "lfx.custom.custom_component", "lfx.custom.custom_component.component",
             "lfx.io", "lfx.memory", "lfx.schema", "lfx.schema.message", "httpx"]
    modules = {name: types.ModuleType(name) for name in names}
    modules["lfx.custom.custom_component.component"].Component = Component
    modules["lfx.schema.message"].Message = Message
    modules["lfx.memory"].aget_messages = AsyncMock(return_value=[])
    for name in ["BoolInput", "MessageTextInput", "HandleInput", "StrInput", "SecretStrInput", "IntInput", "MultilineInput", "Output"]:
        setattr(modules["lfx.io"], name, Input)
    modules["httpx"].HTTPStatusError = HTTPStatusError
    modules["httpx"].RequestError = RequestError
    modules["httpx"].AsyncClient = None
    result = {}
    with patch.dict(sys.modules, modules):
        for name in ["context_intake", "term_router", "domain_term_agent", "context_supervisor", "direct_term_retriever", "resource_evidence_lookup", "evidence_visual_spec", "response_composer"]:
            spec = importlib.util.spec_from_file_location(name, ROOT / "src/langflow_components" / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            result[name] = module
    return result


MODULES = load_components()
INTAKE = MODULES["context_intake"]


class OfflineSelector(selectors.SelectSelector):
    def select(self, timeout=None):
        # All external awaits are mocked; do not poll Windows sockets with no FDs.
        return super().select(timeout) if self.get_map() else []


class OfflineLoop(asyncio.SelectorEventLoop):
    def __init__(self):
        super().__init__(selector=OfflineSelector())

    def _make_self_pipe(self):
        # No thread or network I/O is used by these component behavior tests.
        pass

    def _close_self_pipe(self):
        pass

    def _write_to_self(self):
        pass


class OfflineAsyncCase(unittest.IsolatedAsyncioTestCase):
    def _setupAsyncioRunner(self):
        self._asyncioRunner = asyncio.Runner(loop_factory=OfflineLoop)


def extraction(index=0, **overrides):
    return {"request_index": index, "is_request": True, "recipient_department": "", "source_department": "",
            "surrounding_context": "", "user_goal": "", "extra_notes": "", "general_explanation": False,
            "language": "en", "message_kind": "TERMINOLOGY", "conversational_reply": "",
            "recipient_departments": [], "terms": [], "request_task": "INTERPRET", **overrides}


def intake(text="What does ETL mean?", session="session-a"):
    c = INTAKE.ContextIntake(user_query=Message(text=text, session_id=session, id="current"),
                             google_api_key="mock-private-key", graph=types.SimpleNamespace(
                                 flow_id="11111111-1111-4111-8111-111111111111", session_id=session))
    c._pre_run_setup()
    return c


def model_client(answer=None, error=None):
    response = types.SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
        "candidates": [{"content": {"parts": [{"text": answer}]}}]})
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.post = AsyncMock(return_value=response, side_effect=error)
    return client


class IntakeTests(OfflineAsyncCase):
    async def test_reported_integration_prompt_uses_explicit_role_without_interview(self):
        for introduction, expected in (("I’m an engineer", "Engineer"), ("I'm an engineer", "Engineer"),
                                       ("I am a data engineer", "Data Engineer"), ("I’m from Engineering", "Engineering")):
            original = introduction + " reviewing a proposed integration project.\nPlease explain this message in plain language:\n" + \
                "“The API integration will automate the ETL pipeline, but Finance needs to assess its ROI before approving the investment.”"
            c = intake(original)
            state = c._resolve(extraction(recipient_department="Finance", language="id", terms=["API", "ETL", "ROI"]),
                               [{"role": "user", "text": original}])
            self.assertEqual(state["status"], "READY")
            self.assertEqual(state["context"]["recipient_department"], expected)
            self.assertEqual(state["language"], "en")
            self.assertEqual(state["request_task"], "SIMPLIFY")
            self.assertEqual(state["original_question"], original)
            self.assertTrue(state["context_selected"])
            self.assertEqual(state["terms"], ["API", "ETL", "ROI"])

    async def test_department_mentions_do_not_select_the_recipient(self):
        c = intake("Engineering says the API project needs Finance to assess ROI. Please explain.")
        state = c._resolve(extraction(terms=["API", "ROI"]), [{"role": "user", "text": c.user_query.text}])
        self.assertEqual(state["status"], "CLARIFY")
        self.assertEqual(state["context"]["recipient_department"], "")
        self.assertFalse(state["context_selected"])

    async def test_natural_conversation_preserves_context_without_retrieval(self):
        saved = {"version": 1, "status": "READY", "context_selected": True, "selection_pending": False,
                 "context_selection": {"recipient_department": "Finance", "general_explanation": False},
                 "context": {"recipient_department": "Finance"}, "pending_request": None}
        for text, kind in (("hello", "GREETING"), ("test", "SMALL_TALK"), ("asdfghjkl", "GIBBERISH"),
                           ("you're useless", "INSULT"), ("thanks", "SMALL_TALK")):
            c = intake(text)
            prior = Message(text="Previous answer", sender="Machine", sender_name="AI", session_metadata={"lintasteks": saved})
            client = model_client()
            with patch.object(INTAKE, "aget_messages", AsyncMock(return_value=[prior])), patch.object(INTAKE.httpx, "AsyncClient", return_value=client):
                ready, reply = await asyncio.gather(c.build_ready(), c.build_clarification())
            self.assertEqual(ready.text, "")
            self.assertEqual(c.stopped, {"ready"})
            self.assertTrue(reply.text)
            self.assertNotIn("gibberish", reply.text.lower())
            self.assertEqual(reply.session_metadata["lintasteks"]["message_kind"], kind)
            self.assertEqual(reply.session_metadata["lintasteks"]["context_selection"]["recipient_department"], "Finance")
            self.assertTrue(reply.session_metadata["lintasteks"]["context_selected"])
            client.post.assert_not_awaited()

    async def test_meaningful_unknown_terms_are_not_treated_as_gibberish(self):
        c = intake()
        c._memory = {"context_selected": True, "context_selection": {"recipient_department": "Finance"}, "pending_request": None}
        for text in ("What is SOP?", "What is KPI?", "SOP", "BGP", "What is SKU-123?"):
            state = c._resolve(extraction(), [{"role": "user", "text": text}])
            self.assertEqual(state["status"], "READY")
            self.assertEqual(state["context"]["recipient_department"], "Finance")
        self.assertIsNone(c._non_request("You're useless, what is ROI?"))

    async def test_semantic_conversation_classification_uses_a_natural_reply(self):
        for kind, text, reply in (("GREETING", "Good afternoon, how are you?", "Hi! I'm ready to help. What term are you exploring?"),
                                 ("GIBBERISH", "Why purple deploys the chair?", "Could you clarify what you mean by that?"),
                                 ("OUT_OF_SCOPE", "Write a recipe for pancakes", "I can help interpret workplace terms. Is there a term in that request you'd like explained?")):
            c = intake(text)
            client = model_client(json.dumps(extraction(is_request=False, message_kind=kind, conversational_reply=reply)))
            with patch.object(INTAKE, "aget_messages", AsyncMock(return_value=[])), patch.object(INTAKE.httpx, "AsyncClient", return_value=client):
                self.assertEqual((await c.build_ready()).text, "")
                self.assertEqual((await c.build_clarification()).text, reply)
            self.assertEqual((await c._state())["message_kind"], kind)

    async def test_greeting_does_not_consume_a_pending_context_choice(self):
        c = intake("hello")
        history = [Message(text="To tailor the explanation, which department?", sender="Machine", sender_name="LintasTeks Context"),
                   Message(text="What is ROI?", sender="User")]
        with patch.object(INTAKE, "aget_messages", AsyncMock(return_value=history)):
            reply = await c.build_clarification()
        saved = reply.session_metadata["lintasteks"]
        self.assertEqual(saved["pending_request"]["original_question"], "What is ROI?")
        self.assertIn("previous terminology question", reply.text)
        self.assertFalse(saved["context_selected"])
        c2 = intake("finance")
        c2._memory = saved
        conversation = [{"role": "user", "text": "What is ROI?"}, {"role": "user", "text": "finance"}]
        state = c2._resolve(extraction(99), conversation)
        self.assertEqual(state["status"], "READY")
        self.assertEqual(state["original_question"], "What is ROI?")

    async def test_context_choice_is_kept_until_explicitly_changed(self):
        c = intake()
        c._memory = {"context_selected": True, "context_selection": {"recipient_department": "Finance"}, "pending_request": None}
        for term in ("KPI", "ROI", "API"):
            state = c._resolve(extraction(recipient_department="Marketing", source_department="Marketing"),
                               [{"role": "user", "text": "What is " + term + "?"}])
            self.assertEqual(state["status"], "READY")
            self.assertEqual(state["context"]["recipient_department"], "Finance")
            self.assertEqual(state["context"]["source_department"], "")
        for command in ("change context", "reset context!", "ubah konteks", "ganti departemen"):
            reopened = c._control([{"role": "user", "text": command}])
            self.assertEqual(reopened["status"], "CONTEXT_SELECTION")
            self.assertFalse(reopened["context_selected"])
            c2 = intake("What is API?")
            c2._memory = reopened
            conversation = [{"role": "user", "text": "I'm from Finance"}, {"role": "user", "text": "What is API?"}]
            state = c2._resolve(extraction(recipient_department="Finance"), conversation)
            self.assertEqual(state["status"], "CLARIFY")
            self.assertEqual(state["context"]["recipient_department"], "")
            selected = c2._control([{"role": "user", "text": "engineering"}])
            c2._memory = selected
            ready = c2._resolve(extraction(), [{"role": "user", "text": "What is API?"}])
            self.assertEqual(ready["status"], "READY")
            self.assertEqual(ready["context"]["recipient_department"], "engineering")

    async def test_general_context_choice_persists_without_repeated_questions(self):
        c = intake("skip")
        selected = c._control([{"role": "user", "text": "skip"}])
        c._memory = selected
        for question in ("What is ROI?", "What is API?"):
            state = c._resolve(extraction(), [{"role": "user", "text": question}])
            self.assertEqual(state["status"], "READY")
            self.assertTrue(state["general_explanation"])
            self.assertEqual(state["context"]["recipient_department"], "")

    async def test_natural_language_context_change_and_free_form_role(self):
        c = intake("Could you switch my context to HR?")
        c._memory = {"context_selected": True, "context_selection": {"recipient_department": "Finance"}, "pending_request": None}
        state = c._resolve(extraction(is_request=False, message_kind="CONTEXT_CHANGE", recipient_department="HR"),
                           [{"role": "user", "text": c.user_query.text}])
        self.assertEqual(state["status"], "CONTEXT_UPDATED")
        self.assertEqual(state["context_selection"]["recipient_department"], "HR")
        c._memory = c._context_selection_prompt(True)
        state = c._resolve(extraction(is_request=False, message_kind="CONTEXT_REPLY", recipient_department="Accountant"),
                           [{"role": "user", "text": "I am an accountant"}])
        self.assertEqual(state["status"], "CONTEXT_UPDATED")
        self.assertEqual(state["context_selection"]["recipient_department"], "Accountant")
        c._memory = {"context_selected": True, "context_selection": {"recipient_department": "Marketing"}, "pending_request": None}
        state = c._resolve(extraction(is_request=False, message_kind="CONTEXT_REPLY", recipient_department="Accountant"),
                           [{"role": "user", "text": "I am an accountant"}])
        self.assertEqual(state["context_selection"]["recipient_department"], "Accountant")

    async def test_context_selection_survives_conversational_turns_beyond_history_window(self):
        saved = {"version": 1, "status": "READY", "context_selected": True, "selection_pending": False,
                 "context_selection": {"recipient_department": "Finance", "general_explanation": False}, "pending_request": None}
        latest = Message(text="Previous answer", sender="Machine", sender_name="AI", session_metadata={"lintasteks": saved})
        for number in range(18):
            c = intake("hello")
            with patch.object(INTAKE, "aget_messages", AsyncMock(return_value=[latest])):
                latest = await c.build_clarification()
            latest.sender, latest.sender_name = "Machine", "LintasTeks Context"
        c = intake("What is API?")
        client = model_client(json.dumps(extraction(99)))
        with patch.object(INTAKE, "aget_messages", AsyncMock(return_value=[latest])), patch.object(INTAKE.httpx, "AsyncClient", return_value=client):
            state = json.loads((await c.build_ready()).text)
        self.assertEqual(state["context"]["recipient_department"], "Finance")
        self.assertTrue(state["context_selected"])

    async def test_missing_context_gates_retrieval_and_shares_one_extraction(self):
        c = intake()
        c.graph.exclude_branch_conditionally = Mock()
        client = model_client(json.dumps(extraction()))
        with patch.object(INTAKE, "aget_messages", AsyncMock(return_value=[])), patch.object(INTAKE.httpx, "AsyncClient", return_value=client):
            ready, question = await asyncio.gather(c.build_ready(), c.build_clarification())
        self.assertEqual(ready.text, "")
        self.assertIn("Which department", question.text)
        self.assertIn("Where did you encounter", question.text)
        self.assertIn("skip", question.text)
        self.assertEqual(c.stopped, {"ready"})
        self.assertEqual(client.post.await_count, 1)
        c.graph.exclude_branch_conditionally.assert_called_once_with(c._id, output_name="ready")
        self.assertEqual(question.session_id, "session-a")

    async def test_complete_context_is_ready_and_preserves_original_verbatim(self):
        original = 'Engineering said "ETL is delayed". I work in Finance; explain ETL for our reporting.'
        c = intake(original)
        client = model_client(json.dumps(extraction(recipient_department="Finance", source_department="Engineering",
            surrounding_context='Engineering said "ETL is delayed"', user_goal="reporting", extra_notes="Keep it brief")))
        with patch.object(INTAKE, "aget_messages", AsyncMock(return_value=[])), patch.object(INTAKE.httpx, "AsyncClient", return_value=client):
            ready = json.loads((await c.build_ready()).text)
            self.assertEqual((await c.build_clarification()).text, "")
        self.assertEqual(ready["original_question"], original)
        self.assertEqual(ready["request_messages"], [original])
        self.assertEqual(ready["context"]["extra_notes"], "Keep it brief")
        self.assertEqual(c.stopped, {"clarification"})

    async def test_followup_uses_original_question_and_exact_scoped_history(self):
        c = intake("Marketing; Engineering mentioned a dashboard mismatch. Preparing a campaign report.")
        original = "What does ETL mean?"
        history = [c.user_query, Message(text="To tailor the explanation, which department?", sender="Machine", sender_name="LintasTeks Context"),
                   Message(text=original, sender="User")]
        c.user_query.sender = "User"
        reader = AsyncMock(return_value=history)
        client = model_client(json.dumps(extraction(recipient_department="Marketing", source_department="Engineering",
            surrounding_context="dashboard mismatch", user_goal="campaign report")))
        with patch.object(INTAKE, "aget_messages", reader), patch.object(INTAKE.httpx, "AsyncClient", return_value=client):
            payload = json.loads((await c.build_ready()).text)
        self.assertEqual(payload["original_question"], original)
        self.assertEqual(payload["request_messages"], [original, c.user_query.text])
        reader.assert_awaited_once_with(session_id="session-a", flow_id=UUID(c.graph.flow_id), limit=14, order="DESC")

    async def test_skip_continues_original_request(self):
        c = intake("skip")
        conversation = [{"role": "user", "text": "What does ETL mean?"},
                        {"role": "assistant", "text": "To tailor the explanation, what is your role?", "sender_name": "LintasTeks Context"},
                        {"role": "user", "text": "skip"}]
        state = c._resolve(extraction(general_explanation=True), conversation)
        self.assertEqual(state["status"], "READY")
        self.assertEqual(state["original_question"], "What does ETL mean?")
        self.assertTrue(state["context_incomplete"])

    async def test_partial_reply_does_not_create_an_endless_question_loop(self):
        c = intake("Finance")
        conversation = [{"role": "user", "text": "What does ETL mean?"},
                        {"role": "assistant", "text": "Agar penjelasannya sesuai kebutuhan Anda, konteks?", "sender_name": "LintasTeks Context"},
                        {"role": "user", "text": "Finance"}]
        state = c._resolve(extraction(recipient_department="Finance"), conversation)
        self.assertEqual(state["status"], "READY")
        self.assertTrue(state["context_incomplete"])

    async def test_error_response_is_not_counted_as_prior_clarification(self):
        c = intake()
        conversation = [{"role": "user", "text": c.user_query.text},
                        {"role": "assistant", "text": "API_ERROR: Try again", "sender_name": "LintasTeks Context"},
                        {"role": "user", "text": "retry"}]
        self.assertEqual(c._resolve(extraction(), conversation)["status"], "CLARIFY")

    async def test_new_request_does_not_restore_old_request_text(self):
        c = intake("What does gross profit mean? Just a general definition.")
        conversation = [{"role": "user", "text": "Explain ETL for a campaign dashboard"},
                        {"role": "assistant", "text": "To tailor the explanation, what is your role?", "sender_name": "LintasTeks Context"},
                        {"role": "user", "text": c.user_query.text}]
        state = c._resolve(extraction(2, general_explanation=True), conversation)
        self.assertEqual(state["original_question"], c.user_query.text)
        self.assertEqual(state["request_messages"], [c.user_query.text])

    async def test_department_choice_without_a_question_is_saved_not_searched(self):
        c = intake("Marketing")
        state = c._resolve(extraction(is_request=False), [{"role": "user", "text": "Marketing"}])
        self.assertEqual(state["status"], "CONTEXT_UPDATED")
        self.assertIn("Context saved", state["question"])
        self.assertTrue(state["context_selected"])

    async def test_no_session_never_reads_unscoped_history(self):
        c = intake(session="")
        reader = AsyncMock()
        with patch.object(INTAKE, "aget_messages", reader):
            messages = await c._conversation(c.user_query.text)
        reader.assert_not_awaited()
        self.assertEqual(len(messages), 1)

    async def test_prior_identical_request_is_not_mistaken_for_current_stored_message(self):
        c = intake()
        previous = Message(text=c.user_query.text, sender="User", id="older-message")
        with patch.object(INTAKE, "aget_messages", AsyncMock(return_value=[previous])):
            messages = await c._conversation(c.user_query.text)
        self.assertEqual(len(messages), 2)

    async def test_new_component_build_resets_extraction_and_branch_cache(self):
        c = intake()
        states = [{"status": "CLARIFY", "question": "Context?"},
                  {"status": "READY", "original_question": c.user_query.text, "context": {}}]
        with patch.object(c, "_collect", AsyncMock(side_effect=states)) as collect:
            self.assertEqual((await c.build_ready()).text, "")
            c._pre_run_setup()
            self.assertEqual(json.loads((await c.build_ready()).text)["status"], "READY")
        self.assertEqual(collect.await_count, 2)

    async def test_history_failure_is_explicit_and_does_not_call_model(self):
        c = intake()
        with patch.object(INTAKE, "aget_messages", AsyncMock(side_effect=RuntimeError("private provider details"))):
            state = await c._collect()
        self.assertEqual(state["status"], "MEMORY_ERROR")
        self.assertNotIn("private provider details", state["question"])

    async def test_missing_key_returns_configuration_error(self):
        c = intake()
        c.google_api_key = ""
        with patch.object(INTAKE, "aget_messages", AsyncMock(return_value=[])):
            state = await c._collect()
        self.assertEqual(state["status"], "CONFIGURATION_ERROR")

    async def test_malformed_model_output_and_http_error_gate_ready_branch(self):
        for raw, error, status in [("not JSON", None, "MALFORMED_CONTEXT"),
                                   (json.dumps(extraction(99)), None, "MALFORMED_CONTEXT"),
                                   ("", HTTPStatusError(401), "API_ERROR")]:
            c = intake("unclear reply")
            client = model_client(raw, error)
            with patch.object(INTAKE, "aget_messages", AsyncMock(return_value=[])), patch.object(INTAKE.httpx, "AsyncClient", return_value=client):
                self.assertEqual((await c.build_ready()).text, "")
                self.assertEqual((await c._state())["status"], status)

    async def test_new_roi_question_repairs_stale_or_invalid_model_index(self):
        c = intake("apa itu ROI?")
        conversation = [{"role": "user", "text": "What is SOP?"},
                        {"role": "user", "text": "I'm from the engineering department"},
                        {"role": "assistant", "text": "No matching definition", "sender_name": "AI"},
                        {"role": "user", "text": c.user_query.text}]
        for wrong_index in (1, 99):
            state = c._resolve(extraction(wrong_index, source_department="Engineering", surrounding_context="Engineering"), conversation)
            self.assertEqual(state["original_question"], c.user_query.text)
            self.assertEqual(state["context"]["recipient_department"], "engineering")
            self.assertEqual(state["context"]["source_department"], "")
            self.assertEqual(state["language"], "id")

    async def test_department_reply_resumes_pending_kpi_even_if_model_selects_reply(self):
        c = intake("I'm from the engineering department")
        conversation = [{"role": "user", "text": "How about KPI?"},
                        {"role": "assistant", "text": "To tailor the explanation, where?", "sender_name": "LintasTeks Context"},
                        {"role": "user", "text": c.user_query.text}]
        state = c._resolve(extraction(2, source_department="Engineering", surrounding_context="Engineering"), conversation)
        self.assertEqual(state["status"], "READY")
        self.assertEqual(state["original_question"], "How about KPI?")
        self.assertEqual(state["context"]["recipient_department"], "engineering")
        self.assertEqual(state["context"]["source_department"], "")
        self.assertEqual(state["context"]["surrounding_context"], "")

    async def test_profile_survives_text_history_window_via_scoped_metadata(self):
        c = intake("what is ROI?")
        saved = {"version": 1, "status": "READY", "original_question": "What is SOP?",
                 "context": {"recipient_department": "Engineering"}, "language": "en"}
        prior = Message(text="Previous answer", sender="Machine", sender_name="AI", session_metadata={"lintasteks": saved})
        client = model_client(json.dumps(extraction(99)))
        with patch.object(INTAKE, "aget_messages", AsyncMock(return_value=[prior])), patch.object(INTAKE.httpx, "AsyncClient", return_value=client):
            ready = await c.build_ready()
            question = await c.build_clarification()
        self.assertEqual(question.text, "")
        self.assertEqual(json.loads(ready.text)["status"], "READY")
        self.assertEqual(ready.session_metadata["lintasteks"]["context"]["recipient_department"], "Engineering")
        fresh = intake(session="")
        with patch.object(INTAKE, "aget_messages", AsyncMock(return_value=[prior])):
            await fresh._conversation(fresh.user_query.text)
        self.assertEqual(fresh._memory, {})

    async def test_empty_or_malformed_extraction_preserves_roi_and_known_profile(self):
        for raw in ("", "not JSON"):
            c = intake("what is ROI?")
            history = [Message(text="I'm from the engineering department", sender="User")]
            client = model_client(raw)
            with patch.object(INTAKE, "aget_messages", AsyncMock(return_value=history)), patch.object(INTAKE.httpx, "AsyncClient", return_value=client):
                state = await c._collect()
            self.assertEqual(state["status"], "READY")
            self.assertEqual(state["original_question"], "what is ROI?")
            self.assertIn("FALLBACK_", state["context_extraction"])
            self.assertTrue(state["context_selected"])

    async def test_empty_extraction_on_profile_reply_resumes_original(self):
        c = intake("I'm from the engineering department")
        conversation = [{"role": "user", "text": "What is ROI?"},
                        {"role": "assistant", "text": "To tailor the explanation, department?", "sender_name": "LintasTeks Context"},
                        {"role": "user", "text": c.user_query.text}]
        state = c._fallback(conversation, "MODEL_EMPTY")
        self.assertEqual(state["status"], "READY")
        self.assertEqual(state["original_question"], "What is ROI?")
        self.assertEqual(state["context"]["surrounding_context"], "")

    async def test_profile_with_situation_does_not_erase_source_context(self):
        c = intake("I'm from Engineering. Finance mentioned ROI in a budget review.")
        conversation = [{"role": "user", "text": "What is ROI?"},
                        {"role": "assistant", "text": "To tailor the explanation, context?", "sender_name": "LintasTeks Context"},
                        {"role": "user", "text": c.user_query.text}]
        state = c._resolve(extraction(2, source_department="Finance", surrounding_context="Finance budget review"), conversation)
        self.assertEqual(state["context"]["recipient_department"], "Engineering")
        self.assertEqual(state["context"]["source_department"], "Finance")
        self.assertEqual(state["context"]["surrounding_context"], "Finance budget review")
        self.assertEqual(c._fallback(conversation, "MODEL_EMPTY")["context"]["surrounding_context"], c.user_query.text)

    async def test_model_safety_block_is_not_bypassed_by_fallback(self):
        c = intake("What is ROI?")
        client = model_client("")
        client.post.return_value.json = lambda: {"promptFeedback": {"blockReason": "SAFETY"}}
        with patch.object(INTAKE, "aget_messages", AsyncMock(return_value=[])), patch.object(INTAKE.httpx, "AsyncClient", return_value=client):
            state = await c._collect()
        self.assertEqual(state["status"], "MODEL_BLOCKED")

    async def test_disabling_questions_still_supplies_a_context_bundle(self):
        c = intake()
        c.ask_for_context = False
        state = c._resolve(extraction(), [{"role": "user", "text": c.user_query.text}])
        self.assertEqual(state["status"], "READY")

    async def test_oversized_message_is_not_silently_truncated(self):
        self.assertEqual((await intake("x" * 12001)._collect())["status"], "INPUT_TOO_LONG")


class DownstreamTests(OfflineAsyncCase):
    async def test_term_limit_is_explicit_and_does_not_silently_drop_terms(self):
        terms = ["TERM" + str(i) for i in range(13)]
        original = "Explain " + " and ".join(terms)
        c = intake(original)
        state = c._resolve(extraction(terms=terms), [{"role": "user", "text": original}])
        self.assertEqual(state["status"], "TERM_LIMIT")
        router = MODULES["term_router"].TermRouter(user_query=Message(text=json.dumps({"status": "READY", "original_question": original, "terms": terms, "context": {}})))
        self.assertEqual((await router.build_engineering_query()).text, "")
        self.assertEqual(json.loads((await router.build_resolved_question()).text)["status"], "TERM_LIMIT")

    async def test_simplification_without_definitions_keeps_unresolved_terms_in_prompt(self):
        module = MODULES["context_supervisor"]
        request = {"status": "READY", "request_task": "SIMPLIFY", "original_question": "SOP may delay KPI reporting.",
                   "context": {"recipient_department": "Finance"}, "target_terms": ["SOP", "KPI"], "language": "en"}
        supervisor = module.ContextSupervisor(original_query=Message(text=json.dumps(request)), google_api_key="mock-private-key",
            engineering=Message(text='{"status":"NO_MATCHING_TERM","query":"SOP"}'),
            finance=Message(text='{"status":"NO_MATCHING_TERM","query":"KPI"}'), marketing=Message(text='{"status":"SKIPPED"}'))
        client = model_client("Mock sentence structure explanation with unresolved terms")
        with patch.object(module.httpx, "AsyncClient", return_value=client):
            await supervisor.run_supervisor()
        prompt = client.post.call_args.kwargs["json"]["contents"][0]["parts"][0]["text"]
        self.assertIn("SOP may delay KPI reporting.", prompt)
        self.assertIn("Keep technical terms unexpanded", prompt)
        self.assertIn("GROUNDED FINDINGS:\n[]", prompt)

    async def test_pending_sentence_terms_and_task_survive_a_department_only_reply(self):
        original = "The API supports ROI tracking while the ETL job runs."
        c = intake(original)
        pending = c._resolve(extraction(terms=["API", "ROI", "ETL"], request_task="SIMPLIFY"), [{"role": "user", "text": original}])
        self.assertEqual(pending["status"], "CLARIFY")
        c._memory = pending
        conversation = [{"role": "user", "text": original}, {"role": "user", "text": "finance"}]
        resumed = c._resolve(extraction(99, message_kind="CONTEXT_REPLY"), conversation)
        self.assertEqual(resumed["status"], "READY")
        self.assertEqual(resumed["terms"], ["API", "ROI", "ETL"])
        self.assertEqual(resumed["request_task"], "SIMPLIFY")

    async def test_new_sentence_with_terms_does_not_resume_an_old_pending_question(self):
        c = intake("The API supports ETL jobs.")
        c._memory = {"status": "CLARIFY", "pending_request": {"original_question": "What is ROI?", "terms": ["ROI"]}}
        conversation = [{"role": "user", "text": "What is ROI?"}, {"role": "user", "text": c.user_query.text}]
        state = c._resolve(extraction(0, terms=["API", "ETL"], general_explanation=True), conversation)
        self.assertEqual(state["original_question"], c.user_query.text)
        self.assertEqual(state["terms"], ["API", "ETL"])

    async def test_general_mode_searches_all_departments_for_supported_senses(self):
        state = {"status": "READY", "original_question": "What is ROI?", "interpretation_mode": "GENERAL",
                 "context": {}, "general_explanation": True}
        router = MODULES["term_router"].TermRouter(user_query=Message(text=json.dumps(state)))
        for method in (router.build_engineering_query, router.build_finance_query, router.build_marketing_query):
            self.assertEqual((await method()).text, "roi")
        self.assertEqual(json.loads((await router.build_resolved_question()).text)["routed_departments"], ["engineering", "finance", "marketing"])

    async def test_multi_department_selection_survives_and_adds_definition_owner(self):
        c = intake("Engineering and Marketing")
        choice = c._control([{"role": "user", "text": c.user_query.text}])
        self.assertEqual(choice["context_selection"]["recipient_departments"], ["Engineering", "Marketing"])
        c._memory = choice
        state = c._resolve(extraction(terms=["ROI"]), [{"role": "user", "text": "What is ROI?"}])
        self.assertEqual(state["interpretation_mode"], "MULTI")
        self.assertEqual(state["context"]["recipient_departments"], ["Engineering", "Marketing"])
        router = MODULES["term_router"].TermRouter(user_query=Message(text=json.dumps(state)))
        resolved = json.loads((await router.build_resolved_question()).text)
        self.assertEqual(resolved["department_queries"], {"engineering": ["roi"], "finance": ["roi"], "marketing": ["roi"]})

    async def test_multi_selection_reply_resumes_the_original_request(self):
        c = intake("Finance and Engineering")
        conversation = [{"role": "user", "text": "What is pipeline?"},
                        {"role": "assistant", "text": "To tailor the explanation, department?", "sender_name": "LintasTeks Context"},
                        {"role": "user", "text": c.user_query.text}]
        state = c._resolve(extraction(99, terms=["pipeline"], recipient_department="Finance"), conversation)
        self.assertEqual(state["status"], "READY")
        self.assertEqual(state["interpretation_mode"], "MULTI")
        self.assertEqual(state["context"]["recipient_departments"], ["Finance", "Engineering"])
        self.assertEqual(state["original_question"], "What is pipeline?")

    async def test_full_sentence_preserved_with_distinct_per_term_queries(self):
        original = "Simplify this: ETL and the API support a campaign measured using ROI and KPI."
        c = intake(original)
        c._memory = {"context_selected": True, "context_selection": {"recipient_department": "Finance"}, "pending_request": None}
        state = c._resolve(extraction(terms=["ETL", "API", "ROI", "KPI"], request_task="SIMPLIFY"), [{"role": "user", "text": original}])
        self.assertEqual(state["original_question"], original)
        self.assertEqual(state["request_task"], "SIMPLIFY")
        router = MODULES["term_router"].TermRouter(user_query=Message(text=json.dumps(state)))
        resolved = json.loads((await router.build_resolved_question()).text)
        self.assertEqual(resolved["target_terms"], ["etl", "api", "roi", "KPI"])
        self.assertEqual(resolved["department_queries"], {"engineering": ["etl", "api", "KPI"],
                         "finance": ["roi", "KPI"], "marketing": ["KPI"]})
        self.assertEqual(resolved["context"]["recipient_department"], "Finance")

    async def test_fallback_finds_multiple_known_and_unknown_terms_without_losing_focus(self):
        router = MODULES["term_router"].TermRouter()
        self.assertEqual(router._terms("Explain ETL and ROI for a campaign dashboard", []), ["etl", "roi"])
        self.assertEqual(router._terms("What is SOP and KPI?", []), ["SOP", "KPI"])
        self.assertEqual(router._terms("The data pipeline uses an API and tracks ROI.", []), ["data pipeline", "api", "roi"])
        c = intake("What is API?")
        with self.assertRaises(ValueError):
            c._resolve(extraction(terms=["ROI"]), [{"role": "user", "text": c.user_query.text}])

    async def test_per_term_retrieval_preserves_success_empty_and_error(self):
        module = MODULES["direct_term_retriever"]
        store = Mock()
        store.search.side_effect = [[{"text": "API definition", "metadata": {"source_id": "TEST-API"}}], [], RuntimeError("private token")]
        plan = {"department_queries": {"engineering": ["api", "etl", "KPI"]}}
        retriever = module.DirectTermRetriever(query=Message(text="api\netl\nKPI"), query_plan=Message(text=json.dumps(plan)), vector_store=store)
        result = json.loads(retriever.retrieve().text)
        self.assertEqual([x["status"] for x in result["term_results"]], ["FOUND", "NO_EVIDENCE", "SEARCH_ERROR"])
        self.assertEqual([call.kwargs["query"] for call in store.search.call_args_list], ["api", "etl", "KPI"])
        self.assertNotIn("private token", json.dumps(result))

    async def test_batch_specialist_keeps_evidence_grouped_and_reports_partial_findings(self):
        module = MODULES["domain_term_agent"]
        result = {"status": "BATCH", "term_results": [
            {"status": "FOUND", "query": "api", "documents": [{"text": "API definition", "metadata": {"source_id": "TEST-API"}}]},
            {"status": "FOUND", "query": "etl", "documents": [{"text": "Unrelated record", "metadata": {"source_id": "TEST-OTHER"}}]},
            {"status": "SEARCH_ERROR", "query": "KPI", "diagnostic": "Search unavailable"}]}
        answers = {"findings": [{"term": "api", "status": "FOUND", "answer": "API explanation [TEST-API]"},
                                {"term": "etl", "status": "NO_MATCHING_TERM", "answer": ""}]}
        agent = module.DomainTermAgent(query=Message(text="api\netl\nKPI"), request_context='{"language":"en"}', retrieved=Message(text=json.dumps(result)), google_api_key="mock-private-key")
        client = model_client(json.dumps(answers))
        with patch.object(module.httpx, "AsyncClient", return_value=client):
            findings = json.loads((await agent.run_specialist()).text)["term_findings"]
        by_term = {x["query"]: x for x in findings}
        self.assertEqual(by_term["api"]["sources"][0]["id"], "TEST-API")
        self.assertEqual(by_term["etl"]["status"], "NO_MATCHING_TERM")
        self.assertEqual(by_term["KPI"]["status"], "SEARCH_ERROR")
        self.assertEqual(client.post.await_count, 1)
        self.assertIn("English", client.post.call_args.kwargs["json"]["systemInstruction"]["parts"][0]["text"])

    async def test_partial_failure_keeps_exact_stage_and_explicit_response_language(self):
        module = MODULES["context_supervisor"]
        for language, expected in (("en", "English"), ("id", "Indonesian")):
            request = {"status": "READY", "original_question": "Explain API, ETL, and ROI.", "language": language,
                       "request_task": "SIMPLIFY", "context": {"recipient_department": "Engineer"}}
            supervisor = module.ContextSupervisor(original_query=Message(text=json.dumps(request)), google_api_key="mock-private-key",
                engineering=Message(text=json.dumps({"status": "BATCH", "term_findings": [
                    {"query": "api", "status": "MODEL_MALFORMED", "validation_error": "MISSING_FINDING"}, {"query": "etl", "status": "SEARCH_ERROR"}]})),
                finance=Message(text='{"status":"FOUND","query":"roi","answer":"ROI [TEST-ROI]"}'),
                marketing=Message(text='{"status":"SKIPPED"}'))
            client = model_client("Mock sourced explanation")
            with patch.object(module.httpx, "AsyncClient", return_value=client):
                reply = await supervisor.run_supervisor()
            self.assertIn("Engineering [api]: MODEL_MALFORMED", reply.text)
            self.assertIn("(MISSING_FINDING)", reply.text)
            self.assertIn("Engineering [etl]: SEARCH_ERROR", reply.text)
            body = client.post.call_args.kwargs["json"]
            self.assertIn(expected, body["systemInstruction"]["parts"][0]["text"])
            prompt = body["contents"][0]["parts"][0]["text"]
            self.assertNotIn("in Indonesian unless", prompt)
            self.assertIn("UNRESOLVED TERM STATUS", prompt)
            self.assertIn("not a failed database search", prompt.replace("\n", ""))
            if language == "en":
                self.assertIn("specialist response failed validation", reply.text)
                self.assertIn("database search failed", reply.text)

    async def test_batch_specialist_rejects_missing_or_extra_term_definitions(self):
        module = MODULES["domain_term_agent"]
        result = {"status": "BATCH", "term_results": [{"status": "FOUND", "query": "api",
                  "documents": [{"text": "API", "metadata": {"source_id": "TEST-API"}}]}]}
        for answers in ([], [{"term": "roi", "status": "FOUND", "answer": "Wrong term"}]):
            agent = module.DomainTermAgent(query="api", request_context="{}", retrieved=Message(text=json.dumps(result)), google_api_key="mock-private-key")
            client = model_client(json.dumps({"findings": answers}))
            with patch.object(module.httpx, "AsyncClient", return_value=client):
                findings = json.loads((await agent.run_specialist()).text)["term_findings"]
            self.assertEqual(findings[0]["status"], "MODEL_MALFORMED")

    async def test_batch_validation_keeps_api_etl_when_roi_is_invalid(self):
        module = MODULES["domain_term_agent"]
        result = {"status": "BATCH", "term_results": [
            {"status": "FOUND", "query": term, "documents": [{"text": text, "metadata": {"id": source}}]}
            for term, text, source in (("api", "API definition", "ENGINEERING-001"),
                                       ("etl", "ETL definition", "ENGINEERING-003"),
                                       ("roi", "Availability: an unrelated term", "ENGINEERING-012"))]}
        answers = {"findings": [
            {"term": "API", "status": "FOUND", "answer": "API [ENGINEERING-001]"},
            {"term": " etl ", "status": "FOUND", "answer": "ETL [ENGINEERING-003]"},
            {"term": "roi", "status": "FOUND", "answer": ""}]}
        agent = module.DomainTermAgent(query="api\netl\nroi", request_context='{"language":"en"}',
            retrieved=Message(text=json.dumps(result)), google_api_key="mock-private-key")
        client = model_client(json.dumps(answers))
        with patch.object(module.httpx, "AsyncClient", return_value=client):
            output = json.loads((await agent.run_specialist()).text)
        found = {x["query"]: x for x in output["term_findings"]}
        self.assertEqual(found["api"]["status"], "FOUND")
        self.assertEqual(found["etl"]["status"], "FOUND")
        self.assertEqual(found["api"]["sources"][0]["id"], "ENGINEERING-001")
        self.assertEqual(found["etl"]["sources"][0]["id"], "ENGINEERING-003")
        self.assertEqual(found["roi"]["validation_error"], "EMPTY_DEFINITION")
        schema = client.post.call_args.kwargs["json"]["generationConfig"]["responseSchema"]
        self.assertEqual(schema["properties"]["findings"]["items"]["properties"]["term"]["enum"], ["api", "etl", "roi"])
        self.assertEqual(schema["properties"]["findings"]["minItems"], 3)

    async def test_missing_duplicate_and_renamed_findings_only_affect_their_term(self):
        agent = MODULES["domain_term_agent"].DomainTermAgent()
        expected = {term: {"sources": [{"id": term}], "documents": [{"text": term}]} for term in ("api", "etl", "roi")}
        etl = {"term": "etl", "status": "FOUND", "answer": "ETL"}
        roi = {"term": "roi", "status": "NO_MATCHING_TERM", "answer": "Unsupported explanation is not evidence"}
        api = {"term": "api", "status": "FOUND", "answer": "API"}
        for answers, error in (([etl, roi], "MISSING_FINDING"),
                               ([api, {**api, "term": "API"}, etl, roi], "DUPLICATE_FINDING"),
                               ([{**api, "term": "Application Programming Interface"}, etl, roi], "MISSING_FINDING")):
            findings, issues = agent._validate_findings(answers, expected)
            by_term = {x["query"]: x for x in findings}
            self.assertEqual(by_term["api"]["status"], "MODEL_MALFORMED")
            self.assertEqual(by_term["api"]["validation_error"], error)
            self.assertEqual(by_term["etl"]["status"], "FOUND")
            self.assertEqual(by_term["roi"]["status"], "NO_MATCHING_TERM")
            self.assertEqual(by_term["roi"]["answer"], "")
            self.assertIn(error, issues)

    async def test_unexpected_model_terms_are_discarded_without_erasing_valid_findings(self):
        agent = MODULES["domain_term_agent"].DomainTermAgent()
        expected = {"api": {"sources": [], "documents": [{"text": "API"}]}}
        findings, issues = agent._validate_findings([
            {"term": "api", "status": "FOUND", "answer": "API"},
            {"term": "private extra term", "status": "FOUND", "answer": "unverified"}], expected)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["status"], "FOUND")
        self.assertEqual(issues, ["UNEXPECTED_TERM"])
        self.assertNotIn("private extra term", json.dumps(findings))

    async def test_invalid_batch_response_reports_parse_or_shape_failure(self):
        module = MODULES["domain_term_agent"]
        result = {"status": "BATCH", "term_results": [{"status": "FOUND", "query": "api",
                  "documents": [{"text": "API", "metadata": {"id": "TEST-API"}}]}]}
        for raw, code in (("not JSON: private content", "INVALID_RESPONSE_JSON"),
                          ('{"findings":{}}', "INVALID_FINDINGS_LIST")):
            agent = module.DomainTermAgent(query="api", request_context="{}", google_api_key="mock-private-key",
                                          retrieved=Message(text=json.dumps(result)))
            with patch.object(module.httpx, "AsyncClient", return_value=model_client(raw)):
                output = json.loads((await agent.run_specialist()).text)
            self.assertEqual(output["term_findings"][0]["status"], "MODEL_MALFORMED")
            self.assertIn(code, output["term_findings"][0]["reason"])
            self.assertNotIn("private content", json.dumps(output))

    async def test_supervisor_receives_sentence_per_term_evidence_and_unknown_terms(self):
        module = MODULES["context_supervisor"]
        request = {"status": "READY", "original_question": "API and ROI are discussed; KPI is not defined.",
                   "target_terms": ["api", "roi", "KPI"], "request_task": "SIMPLIFY", "interpretation_mode": "MULTI",
                   "context": {"recipient_departments": ["Finance", "Engineering"]}, "language": "en"}
        supervisor = module.ContextSupervisor(original_query=Message(text=json.dumps(request)), google_api_key="mock-private-key",
            engineering=Message(text=json.dumps({"status": "BATCH", "term_findings": [
                {"query": "api", "status": "FOUND", "answer": "API [TEST-API]", "sources": [{"id": "TEST-API"}]},
                {"query": "KPI", "status": "NO_MATCHING_TERM"}]})),
            finance=Message(text=json.dumps({"status": "FOUND", "query": "roi", "answer": "ROI [TEST-ROI]", "sources": [{"id": "TEST-ROI"}]})),
            marketing=Message(text='{"status":"SKIPPED"}'))
        client = model_client("Mock plain-language explanation")
        with patch.object(module.httpx, "AsyncClient", return_value=client):
            self.assertEqual((await supervisor.run_supervisor()).text, "Mock plain-language explanation")
        prompt = client.post.call_args.kwargs["json"]["contents"][0]["parts"][0]["text"]
        self.assertIn("TEST-API", prompt)
        self.assertIn("TEST-ROI", prompt)
        self.assertIn('"query": "KPI"', prompt)
        self.assertIn("Keep unsupported terms visible", prompt)
        self.assertIn("GENERAL mode", prompt)
        self.assertIn(request["original_question"], prompt)

    async def test_reported_conversation_replay_keeps_terms_and_engineering_profile(self):
        history = []
        turns = [("hello", json.dumps(extraction(is_request=False))),
                 ("What is SOP?", json.dumps(extraction())),
                 ("I'm from the engineering department", json.dumps(extraction(99, source_department="Engineering"))),
                 ("How about KPI?", json.dumps(extraction(0, source_department="Engineering"))),
                 ("I'm from the engineering department", json.dumps(extraction(99, source_department="Engineering"))),
                 ("Apa itu Return on investment?", json.dumps(extraction(0, source_department="Engineering"))),
                 ("apa itu ROI?", "malformed"), ("what is ROI?", ""),
                 ("skip", "")]
        for number, (text, raw) in enumerate(turns):
            c = intake(text)
            c.user_query.id = "turn-" + str(number)
            c.user_query.sender = "User"
            client = model_client(raw)
            with patch.object(INTAKE, "aget_messages", AsyncMock(return_value=list(reversed(history[-14:])))), patch.object(INTAKE.httpx, "AsyncClient", return_value=client):
                state = await c._state()
                reply = await c.build_clarification() if state["status"] != "READY" else await c.build_ready()
            history.append(c.user_query)
            if 2 <= number < 8:
                self.assertEqual(state["context"]["recipient_department"].lower(), "engineering")
            if state["status"] != "READY":
                self.assertTrue(reply.text)
                reply.sender, reply.sender_name = "Machine", "LintasTeks Context"
                history.append(reply)
                continue
            self.assertEqual(state["status"], "READY")
            router = MODULES["term_router"].TermRouter(user_query=reply)
            resolved = await router.build_resolved_question()
            payload = json.loads(resolved.text)
            expected = "SOP" if number == 2 else "KPI" if number == 3 else "roi"
            self.assertEqual(payload["target_term"], expected)
            if expected == "roi":
                self.assertEqual(payload["routed_departments"], ["finance"])
            supervisor = MODULES["context_supervisor"].ContextSupervisor(original_query=resolved,
                engineering=Message(text='{"status":"NO_MATCHING_TERM","department":"Engineering"}'),
                finance=Message(text='{"status":"NO_MATCHING_TERM","department":"Finance"}'),
                marketing=Message(text='{"status":"NO_MATCHING_TERM","department":"Marketing"}'))
            final = await supervisor.run_supervisor()
            final.sender, final.sender_name = "Machine", "AI"
            history.append(final)
        self.assertEqual(payload["original_question"], "what is ROI?")
        self.assertEqual(payload["context"]["recipient_department"].lower(), "engineering")

    async def test_roi_aliases_route_finance_for_an_engineering_recipient(self):
        for question in ("Apa itu Return on investment?", "apa itu ROI?", "what is ROI?", "How about ROI?"):
            c = intake(question)
            state = c._resolve(extraction(99, recipient_department="Engineering", source_department="Engineering"),
                               [{"role": "user", "text": question}])
            state["status"] = "READY"  # After optional clarification/skip, routing uses the same preserved request.
            router = MODULES["term_router"].TermRouter(user_query=Message(text=json.dumps(state)))
            self.assertEqual((await router.build_finance_query()).text, "roi")
            self.assertEqual((await router.build_engineering_query()).text, "")
            self.assertEqual((await router.build_marketing_query()).text, "")
            resolved = json.loads((await router.build_resolved_question()).text)
            self.assertEqual(resolved["context"]["recipient_department"], "Engineering")

    async def test_unknown_sop_kpi_do_not_search_for_the_recipient_reply(self):
        for question, term in (("What is SOP?", "SOP"), ("How about KPI?", "KPI")):
            state = {"status": "READY", "original_question": question,
                     "context": {"recipient_department": "Engineering", "source_department": "", "surrounding_context": ""}}
            router = MODULES["term_router"].TermRouter(user_query=Message(text=json.dumps(state)))
            for method in (router.build_engineering_query, router.build_finance_query, router.build_marketing_query):
                self.assertEqual((await method()).text, term)

    async def test_no_matching_evidence_uses_request_language_and_retains_profile(self):
        module = MODULES["context_supervisor"]
        request = {"status": "READY", "original_question": "What is SOP?", "language": "en",
                   "context": {"recipient_department": "Engineering"}}
        supervisor = module.ContextSupervisor(original_query=Message(text=json.dumps(request), session_id="session-a"),
            engineering=Message(text='{"status":"NO_MATCHING_TERM","department":"Engineering"}'),
            finance=Message(text='{"status":"SKIPPED"}'), marketing=Message(text='{"status":"SKIPPED"}'))
        reply = await supervisor.run_supervisor()
        self.assertIn("retrieved documents do not support", reply.text)
        self.assertNotIn("Hasil pemeriksaan", reply.text)
        self.assertEqual(reply.session_id, "session-a")
        self.assertEqual(reply.session_metadata["lintasteks"]["context"]["recipient_department"], "Engineering")

    async def test_recipient_department_does_not_change_source_routing(self):
        payload = {"status": "READY", "original_question": "What does ETL mean?",
                   "context": {"recipient_department": "Finance", "source_department": "Engineering",
                               "surrounding_context": "transaction reporting"}}
        router = MODULES["term_router"].TermRouter(user_query=Message(text=json.dumps(payload)))
        self.assertEqual((await router.build_engineering_query()).text, "etl")
        self.assertEqual((await router.build_finance_query()).text, "")
        resolved = json.loads((await router.build_resolved_question()).text)
        self.assertEqual(resolved["context"]["recipient_department"], "Finance")

    async def test_explicit_term_wins_over_words_in_work_context(self):
        router = MODULES["term_router"].TermRouter()
        self.assertEqual(router._identify("Explain ETL in a campaign dashboard context")[0], "etl")

    async def test_specialist_receives_original_context_and_retains_demo_source_id(self):
        module = MODULES["domain_term_agent"]
        context = '{"original_question":"What does ETL mean?","context":{"recipient_department":"Marketing"}}'
        agent = module.DomainTermAgent(query=Message(text="etl"), request_context=Message(text=context),
            google_api_key="mock-private-key", retrieved=Message(text=json.dumps({"status": "FOUND", "documents": [
                {"text": "ETL is extract, transform, load.", "metadata": {"source_id": "DEMO-ENG-ETL", "synthetic_example": True}}]})))
        client = model_client("ETL explanation [DEMO-ENG-ETL]")
        with patch.object(module.httpx, "AsyncClient", return_value=client):
            result = json.loads((await agent.run_specialist()).text)
        prompt = client.post.call_args.kwargs["json"]["contents"][0]["parts"][0]["text"]
        self.assertIn(context, prompt)
        self.assertEqual(result["sources"][0]["id"], "DEMO-ENG-ETL")
        self.assertTrue(result["sources"][0]["synthetic_example"])

    async def test_supervisor_uses_collected_context_and_general_limit(self):
        module = MODULES["context_supervisor"]
        request = {"status": "READY", "original_question": "What does ETL mean?", "context": {"recipient_department": "Finance"},
                   "context_incomplete": True, "general_explanation": True, "language": "en", "context_selected": True,
                   "context_selection": {"recipient_department": "Finance", "general_explanation": True}, "pending_request": None}
        supervisor = module.ContextSupervisor(original_query=Message(text=json.dumps(request)), google_api_key="mock-private-key",
            engineering=Message(text=json.dumps({"status": "FOUND", "answer": "ETL", "sources": []})),
            finance=Message(text='{"status":"SKIPPED"}'), marketing=Message(text='{"status":"SKIPPED"}'))
        client = model_client("General ETL explanation")
        with patch.object(module.httpx, "AsyncClient", return_value=client):
            reply = await supervisor.run_supervisor()
            self.assertEqual(reply.text, "General ETL explanation")
            self.assertTrue(reply.session_metadata["lintasteks"]["context_selected"])
            self.assertEqual(reply.session_metadata["lintasteks"]["context_selection"]["recipient_department"], "Finance")
        prompt = client.post.call_args.kwargs["json"]["contents"][0]["parts"][0]["text"]
        self.assertIn('"recipient_department": "Finance"', prompt)
        self.assertIn("Never ask the intake questions again", prompt)
        self.assertIn("Do not copy another department's", prompt)


class ExportTests(unittest.TestCase):
    def test_graph_metadata_matches_code_and_has_separate_clarification_branch(self):
        flow = json.loads((ROOT / "flows/terminology-agent.json").read_text(encoding="utf-8"))
        nodes = {n["id"]: n for n in flow["data"]["nodes"]}
        edges = flow["data"]["edges"]
        mapping = {"ContextIntake": "context_intake", "TermRouter": "term_router", "DomainTermAgent": "domain_term_agent",
                   "ContextSupervisor": "context_supervisor", "DirectTermRetriever": "direct_term_retriever",
                   "ResourceEvidenceLookup": "resource_evidence_lookup", "EvidenceVisualSpec": "evidence_visual_spec", "ResponseComposer": "response_composer"}
        for node in nodes.values():
            kind = node["data"]["type"]
            if kind in mapping:
                cfg = node["data"]["node"]
                expected = (ROOT / "src/langflow_components" / f"{mapping[kind]}.py").read_text(encoding="utf-8")
                self.assertEqual(cfg["template"]["code"]["value"], expected)
                klass = getattr(MODULES[mapping[kind]], kind)
                self.assertEqual(set(cfg["field_order"]), {x.name for x in klass.inputs})
                for port in cfg["outputs"]:
                    self.assertTrue(callable(getattr(klass, port["method"])))
        intake_id = next(k for k, n in nodes.items() if n["data"]["type"] == "ContextIntake")
        for edge in edges:
            source, target = nodes[edge["source"]], nodes[edge["target"]]
            name = edge["data"]["sourceHandle"]["name"]
            field = edge["data"]["targetHandle"]["fieldName"]
            self.assertIn(name, {o["name"] for o in source["data"]["node"]["outputs"]})
            self.assertIn(field, target["data"]["node"]["template"])
            expected_field = target["data"]["node"]["template"][field]
            self.assertEqual(edge["data"]["targetHandle"]["type"], expected_field["type"],
                             f"Wrong target handle type: {edge['target']}/{field}")
            self.assertEqual(edge["data"]["targetHandle"]["inputTypes"], expected_field.get("input_types", []),
                             f"Wrong target input types: {edge['target']}/{field}")
            self.assertEqual(json.loads(edge["targetHandle"].replace("œ", '"')), edge["data"]["targetHandle"])
            self.assertEqual(json.loads(edge["sourceHandle"].replace("œ", '"')), edge["data"]["sourceHandle"])
            if edge["source"] == intake_id and name == "clarification":
                self.assertEqual(target["data"]["type"], "ChatOutput")
                self.assertEqual(target["data"]["node"]["template"]["sender_name"]["value"], "LintasTeks Context")
        self.assertEqual(len([e for e in edges if e["data"]["targetHandle"]["fieldName"] == "request_context"]), 3)
        self.assertEqual(len([e for e in edges if e["data"]["targetHandle"]["fieldName"] == "query_plan"]), 3)
        stores = {k for k, n in nodes.items() if "AstraDBVectorStoreComponent" in n["data"]["type"]}
        gated_stores = {e["target"] for e in edges if e["target"] in stores and
                        e["data"]["targetHandle"]["fieldName"] == "search_query" and
                        nodes[e["source"]]["data"]["type"] == "TermRouter"}
        self.assertEqual(stores, gated_stores)

    def test_all_three_astra_queries_use_query_handles(self):
        flow = json.loads((ROOT / "flows/terminology-agent.json").read_text(encoding="utf-8"))
        nodes = {n["id"]: n for n in flow["data"]["nodes"]}
        collections = set()
        for edge in flow["data"]["edges"]:
            target = nodes[edge["target"]]
            if "AstraDBVectorStoreComponent" in target["data"]["type"] and edge["data"]["targetHandle"]["fieldName"] == "search_query":
                self.assertEqual(edge["data"]["targetHandle"]["type"], "query")
                self.assertEqual(json.loads(edge["targetHandle"].replace("œ", '"'))["type"], "query")
                collections.add(target["data"]["node"]["template"]["collection_name"]["value"])
        self.assertEqual(collections, {"Engineering", "Finance", "Marketing_"})


if __name__ == "__main__":
    unittest.main()
