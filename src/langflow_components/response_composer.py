"""Keeps explanations and visual data intact, including resource clarification state."""
import json
from lfx.custom.custom_component.component import Component
from lfx.io import MessageTextInput, Output
from lfx.schema.message import Message


class ResponseComposer(Component):
    display_name = "Response Composer — Explanation and Resource Visual"
    description = "Preserves sources, session context, resource choices, and explicit visual JSON for MCP."
    icon = "messages-square"
    name = "ResponseComposer"
    inputs = [
        MessageTextInput(name="answer", display_name="Supervisor Answer", input_types=["Message"], required=True),
        MessageTextInput(name="evidence", display_name="Resource Evidence", input_types=["Message"], required=True),
        MessageTextInput(name="visual", display_name="Visual Specification", input_types=["Message"], required=True),
    ]
    outputs = [Output(name="response", display_name="Final Explanation and Visual Data", method="compose")]

    def compose(self) -> Message:
        text = str(getattr(self.answer, "text", self.answer) or "").strip()
        metadata = dict(getattr(self.answer, "session_metadata", None) or {})
        try:
            evidence = json.loads(str(getattr(self.evidence, "text", self.evidence) or ""))
            request = evidence.get("request", {})
            english = request.get("language") == "en"
            if evidence.get("status") == "AMBIGUOUS":
                candidates = evidence["candidates"]
                text = ("Which resource do you mean by " if english else "Sumber daya mana yang Anda maksud dengan ") + evidence["ambiguous_mention"] + "?\n"
                text += "\n".join(str(i + 1) + ". " + r["name"] + " — " + r["id"] for i, r in enumerate(candidates))
                text += ("\nReply with its number or exact ID. Your role context is retained." if english else
                         "\nBalas dengan nomor atau ID yang tepat. Konteks peran Anda tetap disimpan.")
                saved = {k: request[k] for k in ("context", "context_selection", "context_selected", "language", "interpretation_mode") if k in request}
                saved.update(version=1, status="RESOURCE_CLARIFY", selection_pending=False,
                    pending_request={"original_question": request["original_question"], "context": request.get("context", {}),
                                     "terms": request.get("terms", []), "request_task": request.get("request_task", "INTERPRET")},
                    pending_resource_request={"request": request, "mention": evidence["ambiguous_mention"], "candidates": candidates})
                metadata["lintasteks"] = saved
            else:
                saved = dict(metadata.get("lintasteks", {}))
                if saved:
                    saved["pending_resource_request"] = None
                    metadata["lintasteks"] = saved
                visual = json.loads(str(getattr(self.visual, "text", self.visual) or ""))
                if visual.get("status") == "READY":
                    text += "\n\n" + ("**Resource map — visual data**" if english else "**Peta sumber daya — data visual**")
                    # The answer's shared status note and SVG header carry provenance.
                    # Avoid repeating that notice between every presentation layer.
                    text += "\n" + visual["text_equivalent"]
                    text += "\n\n```lintasteks-visual\n" + json.dumps(visual, ensure_ascii=False, indent=2) + "\n```"
                elif visual.get("status") == "ERROR":
                    text += "\n\n" + ("The visual is unavailable; the explanation above remains usable." if english else
                                         "Visual tidak tersedia; penjelasan di atas tetap dapat digunakan.")
        except (ValueError, TypeError, KeyError, AttributeError):
            text += "\n\nVISUAL_ERROR: Visual data is unavailable."
        return Message(text=text, session_metadata=metadata,
                       session_id=getattr(self.answer, "session_id", None) or getattr(self.evidence, "session_id", None) or "")
