import json
import re
from lfx.custom.custom_component.component import Component
from lfx.io import MessageTextInput, Output
from lfx.schema.message import Message

# Keyword mapping is for routing only. Definitions come from Astra DB.
GLOSSARY = {
    "engineering": [
        "etl", "extract transform load", "api", "endpoint", "pipeline data", "data pipeline",
        "deployment", "production environment", "staging environment", "ci/cd", "latency",
        "throughput", "scalability", "availability", "sla", "database schema", "data validation", "pipeline",
        "authentication", "authorization", "webhook", "version control", "technical debt",
    ],
    "finance": [
        "gross profit", "gross margin", "revenue", "operating expense", "capital expenditure",
        "budget variance", "cash flow", "accounts receivable", "accounts payable", "invoice",
        "cost allocation", "unit economics", "break-even point", "roi", "return on investment", "forecast", "accrual",
        "reconciliation", "cost center", "working capital", "depreciation", "capex", "opex",
        "laba kotor", "margin kotor", "hpp", "harga pokok penjualan",
    ],
    "marketing": [
        "marketing qualified lead", "sales pipeline", "conversion rate", "click-through rate",
        "impression", "reach", "engagement rate", "customer acquisition cost",
        "customer lifetime value", "return on ad spend", "attribution", "marketing funnel",
        "segmentation", "a/b testing", "retention rate", "churn rate", "campaign",
        "utm parameters", "brand positioning", "lead", "roas", "cac", "ctr", "pipeline",
    ],
}
TERM_ALIASES = {"return on investment": "roi"}
DEPARTMENT_MARKERS = {
    "engineering": ["engineering", "engineer", "devops", "teknik", "data engineer", "developer", "infrastruktur", "api", "database", "etl", "server", "deployment"],
    "finance": ["finance", "finansial", "keuangan", "accounting", "akuntansi", "laba", "biaya", "anggaran", "pendapatan", "profit"],
    "marketing": ["marketing", "pemasaran", "kampanye", "campaign", "iklan", "pelanggan", "prospek", "sales"],
}

class TermRouter(Component):
    display_name = "Supervisor — Context-Aware Query Router"
    description = "Routes the preserved original request; recipient context does not redefine its meaning."
    icon = "network"
    name = "TermRouter"
    inputs = [MessageTextInput(name="user_query", display_name="User Question", input_types=["Message"], required=True)]
    outputs = [
        Output(display_name="Engineering Search", name="engineering_query", method="build_engineering_query", group_outputs=True),
        Output(display_name="Finance Search", name="finance_query", method="build_finance_query", group_outputs=True),
        Output(display_name="Marketing Search", name="marketing_query", method="build_marketing_query", group_outputs=True),
        Output(display_name="Resolved User Context", name="resolved_question", method="build_resolved_question", group_outputs=True),
    ]

    def _text(self, value):
        return str(getattr(value, "text", value) or "").strip()

    def _hit(self, text, phrase):
        return bool(re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text, flags=re.I))

    def _identify(self, query):
        patterns = [r"what does\s+(.+?)\s+mean", r"(?:apa arti(?: dari)?|apa itu|bagaimana dengan|how about|jelaskan(?: arti| istilah)?|explain|what is|define)\s+([^?.,;\n]{2,100})"]
        for pattern in patterns:
            explicit = re.search(pattern, query, re.I)
            if explicit:
                choices = []
                for department, terms in GLOSSARY.items():
                    for term in terms:
                        hit = re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", explicit.group(1), re.I)
                        if hit:
                            choices.append((hit.start(), -len(term), term, department))
                if choices:
                    choices.sort()
                    chosen = choices[0][2]
                    return TERM_ALIASES.get(chosen, chosen), {d for _, _, t, d in choices if t == chosen}
        matches = []
        for department, terms in GLOSSARY.items():
            for term in terms:
                if self._hit(query, term):
                    matches.append((len(term), term, department))
        if matches:
            # Longest match avoids classifying 'Gross Profit' merely as 'profit'.
            matches.sort(reverse=True)
            chosen = matches[0][1]
            return TERM_ALIASES.get(chosen, chosen), {department for _, term, department in matches if term == chosen}
        quoted = re.search(r'["“\']([^"”\']{2,65})["”\']', query)
        if quoted:
            return quoted.group(1).strip(), set()
        match = re.search(r"(?:apa arti(?: dari)?|apa itu|bagaimana dengan|how about|jelaskan(?: istilah)?|what is|what does|define)\s+([^?.,;\n]{2,80})", query, re.I)
        if match:
            candidate = re.split(r"\s+(?:untuk|pada|dalam|di bidang|in|for)\s+", match.group(1), maxsplit=1, flags=re.I)[0]
            candidate = re.sub(r"\s+mean$", "", candidate, flags=re.I)
            return candidate.strip(), set()
        return "", set()

    def _terms(self, question, supplied):
        if isinstance(supplied, list) and supplied:
            candidates = supplied
        else:
            focus = re.search(r"(?:what is|what are|what does|how about|apa itu|apa arti|explain|define|jelaskan)\s+(.+)", question, re.I)
            text = re.split(r"\s+(?:in|for|dalam|untuk)\s+", focus.group(1), maxsplit=1, flags=re.I)[0] if focus else question
            hits = []
            for terms in GLOSSARY.values():
                for term in terms:
                    hits.extend((m.start(), m.end(), term) for m in re.finditer(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text, re.I))
            hits.sort(key=lambda x: (x[0], -(x[1] - x[0])))
            candidates, end = [], -1
            for start, finish, term in hits:
                if start >= end:
                    candidates.append(term)
                    end = finish
            # Unfamiliar acronyms are valid retrieval candidates too.
            candidates += [m.group() for m in re.finditer(r"(?<!\w)[A-Z][A-Z0-9/-]{1,12}(?!\w)", text)
                           if not any(self._hit(term, m.group()) for term in candidates)]
            if not candidates:
                explicit, _ = self._identify(question)
                candidates = re.split(r"\s+(?:and|dan)\s+|[,;]", explicit) if explicit else [question]
        result = []
        for value in candidates:
            if not isinstance(value, str) or not value.strip():
                continue
            term = value.strip().rstrip("?.!")
            canonical = TERM_ALIASES.get(term.lower(), term)
            if any(canonical.lower() in vocabulary for vocabulary in GLOSSARY.values()):
                canonical = canonical.lower()
            if canonical.lower() not in {x.lower() for x in result}:
                result.append(canonical)
        return result

    async def _prepare(self):
        raw = self._text(self.user_query)
        try:
            state = json.loads(raw)
        except (ValueError, TypeError):
            state = None
        if not isinstance(state, dict) or state.get("status") != "READY":
            return {"status": "MISSING_CONTEXT", "term": "", "departments": set(),
                    "question": "", "context": {}}
        question = str(state.get("original_question") or "").strip()
        context = state.get("context") if isinstance(state.get("context"), dict) else {}
        if not question:
            return {"status": "MISSING_CONTEXT", "term": "", "departments": set(),
                    "question": "", "context": {}}
        terms = [] if state.get("request_type") in ("RESOURCE", "RELATIONSHIP") and not state.get("terms") else self._terms(question, state.get("terms"))
        if len(terms) > 12:
            return {**state, "status": "TERM_LIMIT", "term": "", "departments": set(),
                    "question": question, "department_queries": {d: [] for d in GLOSSARY}}
        source = str(context.get("source_department") or "") + " " + str(context.get("surrounding_context") or "")
        source_departments = set()
        for department, indicators in DEPARTMENT_MARKERS.items():
            if any(self._hit(source, marker) for marker in indicators):
                source_departments.add(department)
        recipients = context.get("recipient_departments") or ([context["recipient_department"]] if context.get("recipient_department") else [])
        mode = state.get("interpretation_mode") or ("MULTI" if len(recipients) > 1 else "SINGLE" if recipients and not state.get("general_explanation") else "GENERAL")
        comparison = {d for d, markers in DEPARTMENT_MARKERS.items() if any(self._hit(str(r), marker) for r in recipients for marker in markers)}
        queries, routes = {d: [] for d in GLOSSARY}, []
        for term in terms:
            owners = {d for d, vocabulary in GLOSSARY.items() if any(TERM_ALIASES.get(t, t).lower() == term.lower() for t in vocabulary)}
            selected = set(GLOSSARY) if mode == "GENERAL" else owners | source_departments | (comparison if mode == "MULTI" else set())
            if not selected:
                selected = set(GLOSSARY)
            for department in selected:
                queries[department].append(term)
            routes.append({"term": term, "departments": sorted(selected)})
        return {**state, "question": question, "term": terms[0] if terms else "",
                "target_terms": terms, "interpretation_mode": mode, "term_routes": routes,
                "department_queries": queries, "departments": {d for d, values in queries.items() if values}}

    async def _make(self, department):
        state = await self._prepare()
        if state["status"] != "READY" or department not in state["departments"]:
            return Message(text="")
        # Short term-only query improves recall for exact glossary entries.
        return Message(text="\n".join(state["department_queries"][department]))

    async def build_engineering_query(self) -> Message:
        return await self._make("engineering")

    async def build_finance_query(self) -> Message:
        return await self._make("finance")

    async def build_marketing_query(self) -> Message:
        return await self._make("marketing")

    async def build_resolved_question(self) -> Message:
        state = await self._prepare()
        if state["status"] == "TERM_LIMIT":
            return Message(text=json.dumps({k: v for k, v in state.items() if k != "departments"}, ensure_ascii=False))
        if state["status"] != "READY":
            return Message(text="MISSING_CONTEXT: Context Intake did not provide a valid original question.")
        payload = {k: v for k, v in state.items() if k != "departments"}
        payload["routed_departments"] = sorted(state["departments"])
        payload["target_term"] = state["term"]
        return Message(text=json.dumps(payload, ensure_ascii=False),
                       session_id=getattr(self.user_query, "session_id", None) or "")
