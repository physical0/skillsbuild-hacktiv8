# Demonstration evidence

## Run inputs, capture actual outputs

- `inputs/01-etl.txt`: single-term journey for the current graph.
- `inputs/02-gross-profit.txt`: another current-graph check.
- `inputs/03-multi-term-limitation.txt`: known routing limitation case.
- `inputs/04-context-intake.txt`: same-session clarification/follow-up and fresh-session skip checks.
- `cases.json`: acceptance expectations, not model responses.
- `data/glossary.jsonl`: authored synthetic records for optional ingestion.
- `inputs/06-resource-impact.txt`: resource relationship and conditional-impact journey.
- `data/resource-catalog.json`: implemented registry embedded in the complete flow; identities and relationships link to source assertions.
- `data/resource-catalog.proposed.json`: historical planning fixture, superseded by the implemented registry.
- `results/resource-offline/`: recorded deterministic resource/visual output; no live model, Langflow or Bob execution.

For each run, save the unedited input and final response in `results/CASE_ID-output.md`. Include the time, environment, retrieved source IDs, and reviewer checks. Capture an actual response even if it fails. Do not fill outputs with a handwritten ideal answer.

Update `results/manifest.json` with the real file paths and evidence status. Keep credentials and private source text out of public artifacts. Unsanitized captures can go in the Git-ignored `private/` directory until reviewed.

After an actual run, use this record shape in the manifest (replace all example paths with existing files):

```json
{
  "runs": [{"case_id": "C01", "status": "success", "path": "demo/results/C01-output.md"}],
  "screenshots": [
    {"kind": "graph", "path": "demo/screenshots/01-langflow-graph.png"},
    {"kind": "playground", "path": "demo/screenshots/02-playground-answer.png"},
    {"kind": "evidence", "path": "demo/screenshots/03-retrieved-evidence.png"}
  ]
}
```

Mark a run `success` only after reviewing the actual answer and evidence. Record failures with `status: "failed"`. Set `runtime_verified` in `project.json` only once that verification exists.

## Screenshots to capture

1. **Graph:** relevant connected components and readable department labels.
2. **Playground:** full input, actual final response, and source identifiers.
3. **Evidence:** retrieved documents/metadata or diagnostics supporting the response.
4. **Bob integration, optional:** complete-message tool call and actual returned answer.

Save sanitized PNG/JPG files under `screenshots/`. Show enough detail to verify the claimed feature. A graph screenshot demonstrates wiring; it does not prove runtime success. No screenshots were available for this project's runtime at repository creation.

## Pitch update

Slide 5 is deliberately a pending-evidence slide. Replace it with a real Playground capture and a short excerpt of the actual output once a run exists. Keep the synthetic-data label if using these fixtures. Do not use unrelated screenshots from the previous Meeting Action Assistant project.

## Output record template

```text
Case:
Timestamp and time zone:
Langflow / OS / model / embedding model:
Corpus version:
Input:
Actual response, verbatim:
Retriever status and source IDs:
Supported claims and any errors:
Screenshot paths:
Reviewer:
```
