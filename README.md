# RAG — Chat with your PDF

A conversational AI app that lets you upload any PDF and ask questions about it in plain English. Powered by **OpenAI GPT-4o-mini**, **LangChain**, and **ChromaDB** — with a clean browser UI built on **Gradio**.

---

## How it works

```
PDF upload → Page extraction → Chunking → Embedding → ChromaDB
                                                            ↓
User question → Similarity search → Top-4 chunks → GPT-4o-mini → Answer
```

1. **Load** — PyPDF reads your PDF and extracts text page by page.
2. **Chunk** — Text is split into 500-character overlapping chunks so nothing falls between the cracks.
3. **Embed** — Each chunk is converted to a 1536-dimension vector via `text-embedding-3-small`.
4. **Store** — Vectors are stored in an in-memory ChromaDB vector database.
5. **Retrieve** — Your question is embedded and the 4 most similar chunks are fetched.
6. **Answer** — GPT-4o-mini reads your question + retrieved chunks + conversation history and writes a grounded answer, with source page numbers appended.

---

## Features

- Upload any PDF through a browser UI — no coding required
- Full **conversation memory** — ask follow-up questions naturally
- **Source citations** — every answer includes the page numbers it drew from
- One-click example questions to get started fast
- Runs locally on `http://localhost:7860`

---

## Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/ariecold/RAG---Chat-with-your-PDF.git
cd RAG---Chat-with-your-PDF
```

### 2. Install dependencies

```bash
pip install -r requirements_paid.txt
```

### 3. Add your OpenAI API key

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
```

> Get your key at [platform.openai.com/api-keys](https://platform.openai.com/api-keys)

### 4. Run the app

```bash
python RAG_PDF_paid.py
```

Open [http://localhost:7860](http://localhost:7860) in your browser, upload a PDF, and start chatting.

---

## Tech stack

| Layer | Library | Role |
|---|---|---|
| UI | [Gradio](https://gradio.app) | Browser-based chat interface |
| PDF parsing | [PyPDF](https://pypdf.readthedocs.io) | Text extraction from PDF pages |
| Chunking | LangChain `RecursiveCharacterTextSplitter` | Splits pages into overlapping chunks |
| Embeddings | OpenAI `text-embedding-3-small` | Converts text to vectors |
| Vector store | [ChromaDB](https://www.trychroma.com) | Stores and searches embeddings |
| LLM | OpenAI `gpt-4o-mini` | Generates answers from retrieved context |
| RAG chain | LangChain `ConversationalRetrievalChain` | Wires retriever + memory + LLM together |
| Memory | LangChain `ConversationBufferMemory` | Keeps full chat history across turns |

---

## Project structure

```
.
├── RAG_PDF_paid.py        # Main app — full pipeline + Gradio UI
├── requirements_paid.txt  # Python dependencies
├── .env                   # Your OpenAI API key (not committed)
└── README.md
```

---

## Cost

Each PDF upload makes one API call per chunk to `text-embedding-3-small` (~$0.00002 / 1K tokens — negligible).  
Each question costs one `gpt-4o-mini` call, typically a few hundred tokens (~$0.0001–0.0003 per question).

---

## License

MIT
