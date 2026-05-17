# RAG — Chat with your PDF

A conversational AI app that lets you upload any PDF and ask questions about it in plain English. Comes in two flavours:

| | Paid version | Free / local version |
|---|---|---|
| **File** | `RAG_PDF_paid.py` | `RAG_pdf_free.py` |
| **Embeddings** | OpenAI `text-embedding-3-small` | `all-MiniLM-L6-v2` (local) |
| **LLM** | OpenAI `gpt-4o-mini` | `google/flan-t5-base` (local) |
| **API key needed** | Yes | No |
| **Answer quality** | High | Good for factual Q&A |
| **Cost** | ~$0.0001–0.0003 / question | Free |
| **Requirements** | `requirements_paid.txt` | `requirements_free.txt` |

---

## How it works

```
PDF upload → Page extraction → Chunking → Embedding → ChromaDB
                                                            ↓
User question → Similarity search → Top-4 chunks → LLM → Answer
```

1. **Load** — PyPDF reads your PDF and extracts text page by page.
2. **Chunk** — Text is split into 500-character overlapping chunks so nothing falls between the cracks.
3. **Embed** — Each chunk is converted to a vector (1536-dim via OpenAI, or 384-dim via local model).
4. **Store** — Vectors are stored in an in-memory ChromaDB vector database.
5. **Retrieve** — Your question is embedded and the 4 most similar chunks are fetched.
6. **Answer** — The LLM reads your question + retrieved chunks + conversation history and writes a grounded answer, with source page numbers appended.

---

## Features

- Upload any PDF through a browser UI — no coding required
- Full **conversation memory** — ask follow-up questions naturally
- **Source citations** — every answer includes the page numbers it drew from
- Runs locally on `http://localhost:7860`
- **Two versions** — use OpenAI for best results, or run 100% offline for free

---

## Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/AriessGit/RAG---Chat-with-your-PDF.git
cd RAG---Chat-with-your-PDF
```

---

### Paid version (OpenAI GPT-4o-mini)

**Best answer quality. Requires an OpenAI API key.**

#### Install dependencies

```bash
pip install -r requirements_paid.txt
```

#### Add your OpenAI API key

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
```

> Get your key at [platform.openai.com/api-keys](https://platform.openai.com/api-keys)

#### Run

```bash
python RAG_PDF_paid.py
```

---

### Free / local version (runs on your PC, no API key)

**100% offline. No cost. Models download automatically on first run (~330 MB total).**

#### Install dependencies

```bash
pip install -r requirements_free.txt
```

> Requires Python 3.9+ and ~2 GB RAM. A GPU is not required — runs on CPU.

#### Run

```bash
python RAG_pdf_free.py
```

Open [http://localhost:7860](http://localhost:7860), upload a PDF, and start chatting.

---

## Tech stack

### Paid version

| Layer | Library | Role |
|---|---|---|
| UI | [Gradio](https://gradio.app) | Browser-based chat interface |
| PDF parsing | [PyPDF](https://pypdf.readthedocs.io) | Text extraction from PDF pages |
| Chunking | LangChain `RecursiveCharacterTextSplitter` | Splits pages into overlapping chunks |
| Embeddings | OpenAI `text-embedding-3-small` | Converts text to 1536-dim vectors |
| Vector store | [ChromaDB](https://www.trychroma.com) | Stores and searches embeddings |
| LLM | OpenAI `gpt-4o-mini` | Generates answers from retrieved context |
| RAG chain | LangChain `ConversationalRetrievalChain` | Wires retriever + memory + LLM |
| Memory | LangChain `ConversationBufferMemory` | Keeps full chat history across turns |

### Free / local version

| Layer | Library | Role |
|---|---|---|
| UI | [Gradio](https://gradio.app) | Browser-based chat interface |
| PDF parsing | [PyPDF](https://pypdf.readthedocs.io) | Text extraction from PDF pages |
| Chunking | LangChain `RecursiveCharacterTextSplitter` | Splits pages into overlapping chunks |
| Embeddings | `all-MiniLM-L6-v2` via HuggingFace | Local 384-dim sentence embeddings (~80 MB) |
| Vector store | [ChromaDB](https://www.trychroma.com) | Stores and searches embeddings |
| LLM | `google/flan-t5-base` via HuggingFace | Local text-to-text generation (~250 MB) |
| Inference | HuggingFace `transformers` pipeline | Runs models locally on CPU |

---

## Project structure

```
.
├── RAG_PDF_paid.py        # Paid version — OpenAI GPT-4o-mini
├── RAG_pdf_free.py        # Free version — local HuggingFace models
├── requirements_paid.txt  # Dependencies for the paid version
├── requirements_free.txt  # Dependencies for the free version
├── .env                   # Your OpenAI API key (not committed)
└── README.md
```

---

## Choosing the right version

- **Need the best answers?** Use the paid version with GPT-4o-mini.
- **Privacy-sensitive document?** Use the free version — nothing leaves your machine.
- **No internet / air-gapped?** Use the free version — models are cached locally after first download.
- **Just experimenting?** Start with the free version; switch to paid when you need richer answers.

---

## Cost (paid version)

Each PDF upload makes one API call per chunk to `text-embedding-3-small` (~$0.00002 / 1K tokens — negligible).  
Each question costs one `gpt-4o-mini` call, typically a few hundred tokens (~$0.0001–0.0003 per question).

---

## License

MIT
