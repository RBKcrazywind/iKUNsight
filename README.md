# iKUNsight

**iKUNsight** is a local multimodal RAG agent for understanding scientific papers. Its name combines “iKUN” with “insight”: the project aims to help readers find useful evidence across a paper's text, figures, and tables, then trace answers back to page-level sources.

This is a research prototype for a single-user, single-paper workspace. It is intended for local experimentation, not public production deployment.

## What It Does

- Upload a local PDF and extract page-level text with `pypdf`.
- Render PDF pages with Poppler (`pdftoppm`) when installed, so readers can inspect the original page alongside retrieved evidence.
- Create page-anchored evidence records for text, figure captions, table captions, and simple table-like numeric lines.
- Route questions to text-only, multimodal, figure-focused, or table-focused retrieval, either manually or through the lightweight agent mode.
- Run a second multimodal retrieval pass when the initial evidence appears insufficient.
- Return evidence excerpts with page references when no language model is configured.
- Optionally send retrieved text and up to three relevant page images to a configured vision-capable model for answer generation.
- Search arXiv, OpenAlex, and Semantic Scholar metadata, deduplicate results, rank them with local lexical relevance scoring, and import an available open PDF.
- Keep the API key in the running server process rather than saving it in the model configuration file.

## Current Limitations

- Retrieval uses lexical matching and hand-written query aliases. It is not embedding-based semantic search and may miss relevant evidence.
- Figure and table handling is a baseline: pages are preserved as images and captions or numeric lines are heuristically detected. Precise figure cropping, OCR, and reliable table-structure recovery are not implemented.
- The project has not yet been evaluated on SPIQA or another benchmark. Recall@k, MRR, answer correctness, multimodal reasoning, and evidence faithfulness have not been measured.
- The interface does not currently display model token usage or API cost. Check usage in the model provider's own dashboard.
- Only one active paper and one local workspace are supported. There is no authentication or multi-user isolation.
- Scanned PDFs, complex layouts, and mathematical notation may not be extracted accurately.

## Requirements

- Python 3.10 or newer
- `pypdf` (installed from `requirements.txt`)
- Poppler is optional but recommended for page-image rendering. The `pdftoppm` executable must be available on `PATH`.
- A model API key is optional. Paper parsing, local retrieval, and extractive evidence answers work without one.

## Run Locally

In PowerShell:

```powershell
git clone <your-github-repository-url>
cd iKUNsight
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python server.py
```

Open <http://127.0.0.1:8766/> in your browser. The server binds to the loopback interface, so it is available only on the computer running it.

To run without cloning, download the repository ZIP, extract it, open a terminal in the extracted `iKUNsight` directory, and run the environment and startup commands above.

## Use the Agent

1. Upload or drag a scientific-paper PDF into the left panel.
2. Wait for the document title, page count, and evidence count to appear.
3. Ask a question about the paper, such as its method, experimental results, a specific figure, or a table.
4. Choose an explicit retrieval mode or leave **Agent automatic routing** selected.
5. Review the answer, execution trace, and evidence cards. Use the cited page number and page image to verify the source in the original PDF.
6. Use **Paper Search** to find papers by topic. Search metadata is fetched from the selected scholarly sources; importing an open PDF downloads that PDF to the local workspace.

Uploading a new PDF replaces the active paper. The application stores the current document index and recent runs in the local `data/` directory.

## Optional Model API

The model settings support custom OpenAI-compatible services, OpenAI, DeepSeek, and a local Ollama endpoint. The application supports Chat Completions and Responses request formats. A separate vision model can be configured when the provider supports image input.

To enable generation, open **Model Settings**, enter the provider's API endpoint, model name, and API key, then test the connection. The key is held in the server process and must be entered again after a restart. When an external model is used, retrieved evidence text and, for a vision model, up to three relevant page images are sent to that provider. Provider charges and data policies apply; paper search itself uses scholarly metadata services and does not call the configured language model.

Do not commit API keys or private papers. The project ignores local `data/`, environment files, and Python cache files by default.

## Architecture

```text
Browser UI (HTML/CSS/JavaScript)
        | local HTTP API
Python server (server.py)
        | upload / parse / retrieve / answer
Agent core (agent_core.py)
        |                         \
Local PDF and evidence index     Optional model provider
```

The agent trace exposes the main planning and retrieval steps. The current automatic router is a lightweight rule-based baseline, not a separately trained agent model.

## Evaluation Roadmap

The prototype is ready for supervisor discussion and controlled experiments. A next step is to create or select a paper-question dataset with page-level gold evidence, then compare:

- Text-only versus multimodal retrieval
- Manual retrieval modes versus automatic modality routing
- One-pass versus iterative retrieval
- Lexical retrieval versus an embedding-based retriever

Report retrieval metrics such as Recall@k and MRR, answer correctness, evidence faithfulness, latency, and (when model usage data is available) token consumption and estimated cost. Do not treat the current functional test counts as benchmark performance.

## Privacy and Deployment

By default, uploaded files, rendered page images, and run history stay in the local `data/` directory. If a cloud model is configured, retrieved evidence is sent to that provider as described above.

This server is a local prototype and is not hardened for public hosting. Before deploying it for multiple users, add authentication, per-user storage and isolation, upload controls, rate limits, HTTPS, protected secret management, and persistent storage. Do not expose the current local server directly to the public internet.
