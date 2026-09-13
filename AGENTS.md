# Repository Guidelines

## Project Structure & Module Organization

- `src/` contains the Version 1 text-based RAG pipeline: ingestion, retrieval, generation, configuration, and the CLI in `main.py`.
- `src/v2/` contains the PDF-based pipeline. Its modules read and normalize PDFs, create adaptive section chunks, ingest them into Chroma, retrieve context, rewrite follow-up questions, and run the Version 2 CLI.
- `src/v3/` contains the isolated hybrid PDF pipeline, offline checks, and retrieval evaluation cases.
- `data/documents/` is the knowledge base (`.txt` files plus PDF collections). Treat these files as source content, not generated output.
- `chroma_db/`, `chroma_db_v2/`, and `chroma_db_v3/` are local vector-store artifacts. `output/` holds captured project images.

## Build, Test, and Development Commands

Run commands from the repository root using the Python 3.11 virtual environment (PowerShell: `.venv\Scripts\Activate.ps1`). Ollama must be running with `llama3.2:3b` and `nomic-embed-text` pulled.

- `python src/ingest.py` builds the Version 1 Chroma collection from text policies.
- `python src/main.py` starts the Version 1 interactive assistant.
- `python -m src.v2.ingest_v2` ingests PDFs from `data/documents/PDF Files/` into the Version 2 store.
- `python -m src.v2.main_v2` starts the Version 2 assistant.
- `python -m src.v3.ingest_v3` incrementally synchronizes all configured PDFs into the Version 3 store.
- `python -m src.v3.evaluate_v3` runs the Version 3 retrieval regression set.
- `python -m src.v3.main_v3` starts the Version 3 assistant.
- `python -m src.v3.api_v3` serves the Version 3 FastAPI API and browser interface.
- `python src/v2/pdf_reader.py` and `python src/v2/chunk_pdf.py` provide parser and chunking smoke checks.
- `python src/v2/test_memory_v2.py` checks bounded follow-up session memory without calling Ollama.
- `python src/test_embedding.py`, `python src/test_chroma.py`, and `python src/test_retrieval.py` exercise Ollama/Chroma integration; they require the corresponding local service and collection.

There is no project build file, formatter, linter, or automated pytest configuration. Install dependencies with `pip install -r requirements.txt`. Avoid committing generated databases or `.env` files.

## Coding Style & Naming Conventions

Use four-space indentation, standard library/import grouping, and `snake_case` for modules, functions, and variables. Prefer small, single-purpose functions and type hints for new Version 2 and Version 3 code. Keep configuration in environment variables; Version 3 settings use the `V3_` prefix documented in `.env.example`.

## Testing Guidelines

Tests are executable smoke scripts named `test_*.py` and live beside the code; `tests/` is currently empty. Run a targeted script after changing retrieval, embeddings, PDF parsing, or prompts. For end-to-end changes, ingest the affected documents first, then exercise the matching CLI and verify cited sources.

## Commit & Pull Request Guidelines

Use short, imperative commit subjects consistent with history (for example, `Add PDF ingestion flow`). Pull requests should explain the user-visible or retrieval behavior change, identify affected data/models, and list validation commands. Include a screenshot or transcript when CLI output changes, and link the relevant issue when one exists.

## Security & Configuration Tips

Keep model names, Chroma paths, thresholds, and rewrite counts in `.env`; never commit credentials or private policy content. Review document changes for accidental secrets before ingesting them into a shared vector store.
