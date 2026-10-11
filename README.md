# LintasTeks

**Understand what another department means, with evidence you can inspect.**

Proposed theme: Productivity / Smart Business. 

LintasTeks is a Langflow prototype for interpreting technical terminology across departments. Its first resource extension connects named objects to documented relationships and conditional implications for the recipient's work, using a synthetic registry.

## What is included and what works now

This repository contains an existing flow export, extracted custom component code, synthetic demo data, sample inputs, Bob instructions, technical documentation, and an editable 10-slide pitch draft.

**Evidence level: static inspection and offline behavior tests.** Repository checks have not run the flow in Langflow, accessed Astra DB, called Gemini, or collected user feedback. A connected graph and passing mocked tests do not establish successful execution.

| Capability | Current evidence |
| --- | --- |
| Chat Input and Chat Output | Connected in the imported graph |
| Conversation and context intake | Welcomes conversational messages; selects context once per session, supports explicit context changes and follow-ups; offline checks only |
| Department routing | Custom keyword router implemented |
| Astra retrieval | Three retrieval branches implemented with diagnostic statuses |
| Specialist explanations | Engineering, Finance, Marketing components implemented |
| Provider/model selection | Optional native Langflow model ports on all five AI stages; shared or per-stage models, with direct Gemini compatibility; offline interface checks only |
| Source and recipient context | Separate fields passed to specialists and final supervisor; model extraction not yet evaluated |
| Final answer with source preservation | Supervisor prompt implemented; citation correctness unmeasured |
| Multiple terms / whole sentences | Per-term retrieval plans and plain-language synthesis implemented; extraction/grounding quality unmeasured |
| Multiple departmental interpretations | General/all-department and explicit multi-department modes implemented; only supported meanings may be shown |
| Exact resource lookup and one-hop relationship checks | Implemented against a source-linked synthetic registry; offline identity/ambiguity/scope/date checks |
| Resource visualization | Explicit graph JSON and local SVG renderer implemented and checked offline; inline Playground/Bob rendering unverified |
| HR and Operations specialists | Proposed, absent from this export |
| Bob calling this flow through MCP | Instructions/configuration template included; not verified |

The design problem is research-supported. The proposed workflow's effectiveness and architectural advantage remain hypotheses. See [research](docs/research.md).

## Quick start

1. Open Langflow Desktop and create a project for LintasTeks.
2. Import [the sanitized flow](flows/terminology-agent.json).
3. Configure Gemini and Astra DB privately. The setup guide maps each field, collection, and account.
4. Add approved glossary documents or seed the supplied synthetic examples into compatible vector collections.
5. Start with [a single-term input](demo/inputs/01-etl.txt) in the Playground.
6. Record the actual response and screenshots using [the demo guide](demo/README.md).

Full instructions: [setup](docs/setup.md). No server is bundled with this repository.

For the **complete updated resource flow**, import the same JSON and follow [resource setup](docs/resource-setup.md). Its registry is already embedded. [Recorded offline map](demo/results/resource-offline/README.md) shows deterministic output, not a live AI/Bob run.

To use another AI provider or model, follow [model selection](docs/model-selection.md). Connect a native Langflow Language Model to the five optional model ports; intake and batch interpretation require structured-output support.

## Reviewer guide

- [Indonesian submission text](docs/submission-id.md): title, project description, problem statement, and justification.
- [Pitch draft](pitch/LintasTeks-pitch-draft.pptx), with source notes and visible evidence limitations.
- [Slide text and presentation notes](pitch/outline.md).
- [Architecture and Bob/Langflow roles](docs/architecture.md).
- [Validation record and evaluation plan](docs/validation.md).
- [Business canvas and cost assumptions](docs/business-canvas.md).
- [Responsible AI and limitations](docs/responsible-ai.md).
- [Submission checklist](docs/submission-checklist.md).

**Actual runtime outputs and screenshots are pending.** The deck's demonstration slide must be updated with real evidence before final submission. Expected behavior in demo cases is not a recorded model response.

## Repository structure

```text
flows/                   Sanitized, importable Langflow export
src/langflow_components/ Custom code extracted from the export
bob/                     MCP example and project rule source
src/visualization/       Dependency-free SVG renderer
agent-instructions.md    Proposed Bob instructions for this project
demo/inputs/             Inputs to run in the Playground
demo/data/               Synthetic glossary and implemented resource registry
demo/results/            Evidence manifest and actual outputs once captured
demo/screenshots/        Sanitized screenshots once captured
docs/                    Setup, research, architecture, evaluation, business
pitch/                   Editable deck, outline, and authoring source
scripts/                 Local package validation and data preparation
tests/                   Offline context/routing checks with mocked services
```

## Local checks

Python 3.10+ is sufficient for repository checks. These checks do not need API accounts:

```powershell
python scripts/validate_repository.py
python scripts/prepare_demo_data.py
```

The second command prepares JSON files for Astra vectorize ingestion. It makes no network request. See the setup guide before uploading.

The local `scripts/` folder is currently Git-ignored by the team's preference. If your copy does not include it, use the prepared demo JSON files and the standalone behavior tests:

```powershell
python -m unittest discover -s tests -v
```

Those tests mock Gemini, memory, and Langflow interfaces. They do not test provider availability or the Langflow scheduler. See [context intake](docs/context-intake.md) for the new conversation and acceptance checks.

Before submission:

```powershell
python scripts/validate_repository.py --submission
```

The submission check intentionally fails while team details, public links, or real demonstration evidence are missing. It checks completeness, not quality, runtime correctness, or public access.

## Configuration and publication

[.env.example](.env.example) documents required settings. The current custom components do not automatically read `.env`. Use Langflow fields/global variables for actual execution. Keep real credentials out of the repository and screenshots.

The original private JSON remains unchanged and Git ignores it. The public export clears credentials and the original Astra endpoint. Component sources mirror that sanitized export.

A local Git repository is initialized on `main`. No remote or public site is created by this setup. Set public reviewer links in `project.json`, then open each link in a signed-out browser to confirm access without a request. A localhost URL is not a reviewer-accessible demo.

## Development priorities

1. Run and record one successful single-term journey.
2. Evaluate multi-term extraction, per-term retrieval, multiple senses, and full-message simplification in the selected recipient context.
3. Run the new resource-registry/visual journey live, then extend it with approved identities and relationship sources.
4. Compare the workflow with a contextual single-agent RAG baseline.

See [roadmap](docs/roadmap.md). There is no open-source license grant yet. Choose a license before inviting reuse of the project or redistributing third-party component code.
