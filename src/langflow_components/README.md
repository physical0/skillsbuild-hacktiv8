# Extracted custom components

The eight component classes here match the complete sanitized Langflow export. Context Intake is the conversational gate; ResourceEvidenceLookup, EvidenceVisualSpec, and ResponseComposer add exact registry checks and visual data. Department-specific configuration remains in the graph. They import Langflow/runtime libraries and are not a standalone Python application. The separate SVG renderer in `src/visualization/` runs with standard Python.

Update the Python sources and embed their code into the canonical flow together when implementation changes. Export tests check code/port consistency and connections, including Astra's native QueryInput handles. These checks do not establish runtime imports or provider compatibility.

The three AI component classes have self-contained adapters for optional Langflow `LanguageModel` connections. These normalize provider-neutral responses into the existing internal text parser format. Credentials and provider settings belong to the native model component; direct Gemini is used only when no model is connected. See [model selection](../../docs/model-selection.md).

Four original components came from the user-provided export and have been updated here; Context Intake was added in this repository. The original code's upstream authorship/license has not been independently established. No license grant is added by this repository setup.
