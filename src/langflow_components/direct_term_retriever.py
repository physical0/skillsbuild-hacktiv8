"""Direct, inspectable retrieval from an already configured Langflow Astra DB VectorStore.
No silent JSON/Data coercion between the vector store and the specialist.
"""
import json
from lfx.custom.custom_component.component import Component
from lfx.io import HandleInput, MessageTextInput, StrInput, IntInput, Output
from lfx.schema.message import Message

class DirectTermRetriever(Component):
    display_name = "Astra — Direct Term Retrieval + Diagnostics"
    description = "Queries the Astra vector-store connection directly and emits an explicit status, query and document count."
    icon = "search-check"
    name = "DirectTermRetriever"
    inputs = [
        HandleInput(name="vector_store", display_name="Astra Vector Store Connection", input_types=["VectorStore"], required=True),
        MessageTextInput(name="query", display_name="Search Term", input_types=["Message"], required=True),
        MessageTextInput(name="query_plan", display_name="Resolved Term Plan", input_types=["Message"], required=False),
        StrInput(name="department", display_name="Department", value="Engineering"),
        IntInput(name="top_k", display_name="Number of Results", value=6, required=False),
    ]
    outputs = [Output(display_name="Retrieval Status + Documents", name="retrieval", method="retrieve")]

    def _out(self, status, query, **extra):
        payload = {"status": status, "department": str(self.department), "query": query, **extra}
        self.status = {"status": status, "query": query,
                       "documents_found": payload.get("documents_found", 0),
                       "diagnostic": payload.get("diagnostic", "")}
        return Message(text=json.dumps(payload, ensure_ascii=False, default=str))

    def retrieve(self) -> Message:
        query = str(getattr(self.query, "text", self.query) or "").strip()
        raw_plan = str(getattr(getattr(self, "query_plan", ""), "text", getattr(self, "query_plan", "")) or "").strip()
        if raw_plan:
            try:
                plan = json.loads(raw_plan)
                queries = plan["department_queries"][str(self.department).lower()]
                if not isinstance(queries, list) or len(queries) > 12 or any(not isinstance(x, str) or not x.strip() for x in queries):
                    raise ValueError("Invalid plan")
            except (ValueError, KeyError, TypeError):
                return self._out("MALFORMED", query, diagnostic="Resolved term plan is invalid; no vector search executed.", documents_found=0)
            if not queries:
                return self._out("SKIPPED", "", diagnostic="Router did not select terms for this department.", documents_found=0)
            if len(queries) > 1:
                findings = [json.loads(self._retrieve_one(term).text) for term in queries]
                return self._out("BATCH", query, term_results=findings,
                                 documents_found=sum(x.get("documents_found", 0) for x in findings))
            query = queries[0]
        return self._retrieve_one(query)

    def _retrieve_one(self, query):
        if not query:
            return self._out("SKIPPED", query, diagnostic="Router did not select this department.", documents_found=0)
        store = getattr(self, "vector_store", None)
        if store is None or not callable(getattr(store, "search", None)):
            return self._out("MISSING_VECTOR_STORE", query,
                             diagnostic="Astra Vector Store Connection was not supplied as a VectorStore. Check the wire and the Astra node's output.",
                             documents_found=0)
        try:
            k = max(1, min(20, int(self.top_k or 6)))
            docs = store.search(query=query, search_type="similarity", k=k)
        except Exception as exc:
            self.log("Direct vector search failed (" + str(self.department) + "): " + type(exc).__name__)
            # Do not leak DB token, endpoint or full provider exception into chat.
            return self._out("SEARCH_ERROR", query,
                             error_type=type(exc).__name__,
                             diagnostic="VectorStore.search failed. Inspect this component's Langflow error/logs and Astra collection settings.",
                             documents_found=0)
        if docs is None:
            return self._out("UNEXPECTED_RETRIEVAL_RESULT", query,
                             diagnostic="VectorStore.search returned None, not a document list.", documents_found=0)
        if not isinstance(docs, (list, tuple)):
            return self._out("UNEXPECTED_RETRIEVAL_RESULT", query,
                             diagnostic="VectorStore.search returned " + type(docs).__name__ + ", not a document list.",
                             documents_found=0)
        if len(docs) == 0:
            return self._out("NO_EVIDENCE", query,
                             diagnostic="The vector-store search completed and returned zero documents. Check that this exact collection contains vectorized glossary entries.",
                             documents_found=0)
        results = []
        for doc in docs:
            content = getattr(doc, "page_content", None)
            metadata = getattr(doc, "metadata", None)
            if content is None and isinstance(doc, dict):
                content = doc.get("page_content") or doc.get("text") or doc.get("content")
                metadata = doc.get("metadata", doc)
            if not isinstance(content, str) or not content.strip():
                continue
            if not isinstance(metadata, dict):
                metadata = {}
            results.append({"text": content, "metadata": metadata})
        if not results:
            return self._out("UNEXPECTED_RETRIEVAL_RESULT", query,
                             diagnostic="Search returned " + str(len(docs)) + " objects, but none had readable page_content/text.",
                             documents_found=len(docs), readable_documents=0)
        return self._out("FOUND", query, documents_found=len(results),
                         diagnostic="VectorStore.search returned readable documents.", documents=results)
