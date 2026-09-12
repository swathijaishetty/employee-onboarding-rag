# Employee Onboarding RAG Assistant

## Version 1 — Basic RAG Prototype

A Retrieval-Augmented Generation (RAG) based employee onboarding assistant that answers questions using company policy documents.

### Tech Stack

- Python 3.11
- Ollama
- Llama 3.2 3B — LLM
- nomic-embed-text — Embeddings
- ChromaDB — Vector Database
- LangChain Text Splitters

### Current Features

- Load employee policy documents
- Split documents into chunks
- Generate embeddings
- Store embeddings in ChromaDB
- Rewrite user queries for better retrieval
- Retrieve relevant document chunks
- Generate answers using Llama 3.2
- Display sources used for the answer
- Handle basic greetings and acknowledgements
- Avoid answering when information is unavailable in the knowledge base
- Interactive CLI-based chat

### Knowledge Base

- Leave Policy
- IT Onboarding Policy
- Work From Home Policy
- Benefits Policy
- Payroll Policy
- Employee Handbook

### RAG Flow

```text
Documents
    ↓
Chunking
    ↓
Embeddings
    ↓
ChromaDB
    ↓
User Query
    ↓
Query Rewriting
    ↓
Retrieval
    ↓
LLM Generation
    ↓
Answer + Sources
```

### Output

![Employee Onboarding RAG Assistant Output](output/output_v1.png)

## Version 2 - PDF RAG Pipeline

Version 2 uses six one-page PDF policies as its knowledge base:

- `benefits_policy.pdf`
- `employee_onboarding.pdf`
- `it_onboarding.pdf`
- `leave_policy.pdf`
- `payroll_policy.pdf`
- `work_from_home_policy.pdf`

The PDF pipeline is implemented in `src/v2/`. `pdf_reader.py` extracts and normalizes page text with `pypdf`; `chunk_pdf.py` splits at policy section headings and only subdivides unusually long sections at sentence boundaries, retaining source, page, and section metadata; `ingest_v2.py` embeds each chunk with Ollama and synchronizes it into the local Chroma collection `employee_policies_v2`; `retrieval_v2.py` embeds a question, applies semantic and lexical relevance checks, and returns focused policy excerpts; `generation_v2.py` asks the Llama model to answer only from those excerpts and parse source citations; `memory_v2.py` rewrites context-dependent follow-ups using a bounded recent-turn window; and `main_v2.py` provides the interactive CLI.

### Running Version 2

From the repository root, activate the Python 3.11 environment and ensure Ollama is running:

```powershell
.venv\Scripts\Activate.ps1
ollama pull llama3.2:3b
ollama pull nomic-embed-text
python -m src.v2.ingest_v2
python -m src.v2.main_v2
```

Configuration is read from `.env` (`PDF_DIRECTORY`, `LLM_MODEL`, `EMBEDDING_MODEL`, `CHROMA_PATH`, `COLLECTION_NAME`, `TOP_K`, `DISTANCE_THRESHOLD`, `MEMORY_TURNS`, `MAX_SECTION_CHARS`, `MIN_SECTION_CHUNKS`, `RETRIEVAL_CANDIDATE_MULTIPLIER`, `MAX_CHUNKS_PER_SOURCE`, `LEXICAL_FALLBACK_DISTANCE`, `LEXICAL_SCORE_WEIGHT`, `RANK_SCORE_MARGIN`, and `MAX_REWRITE_CHARS`). The assistant cites the policy file and page used for an answer and returns `I couldn't find that information in the available employee documents.` when the PDFs do not support the question. Type `clear` during a session to remove follow-up context. Parser, chunking, and memory smoke checks can be run with `python src/v2/test_pdf_reader.py`, `python src/v2/test_chunk_pdf.py`, and `python src/v2/test_memory_v2.py`.

### Version 2 Output

The example shows section-grounded retrieval, a follow-up resolved from session memory, an unsupported-question fallback, and clearing the stored conversation context.

![Employee Onboarding RAG Assistant Version 2 Output](output/output_v2.png)
