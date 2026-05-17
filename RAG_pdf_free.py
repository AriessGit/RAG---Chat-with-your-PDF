# ─────────────────────────────────────────────────────────────────────────────
# rag_pdf_free.py  —  Chat with a PDF using only free, local models
#
# KEY DIFFERENCES vs rag_pdf.py:
#   • No OpenAI API key needed
#   • Embeddings: all-MiniLM-L6-v2 (HuggingFace, runs on CPU)
#   • Generation: flan-t5-base (HuggingFace text2text model)
#   • We build the prompt manually (no LangChain chain needed)
#   • Answers are shorter and simpler — flan-t5-base is a small model
# ─────────────────────────────────────────────────────────────────────────────

import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langchain_huggingface import HuggingFaceEmbeddings
# ↑ wraps sentence-transformers inside LangChain's embedding interface
#   so it works as a drop-in replacement for OpenAIEmbeddings

from transformers import pipeline
# ↑ HuggingFace pipelines — a simple way to run models for specific tasks
#   We use "text2text-generation" which takes text in, produces text out

# Model names — these are downloaded from HuggingFace Hub on first run
EMBEDDING_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
# ↑ compact but effective sentence embedding model (~80 MB)
#   produces 384-dimension vectors (vs 1536 for OpenAI)

GENERATION_MODEL  = "google/flan-t5-base"
# ↑ a small instruction-following model by Google (~250 MB)
#   good at question-answering but gives short, direct answers

# Global state — vectorstore holds the retriever after PDF upload
vectorstore        = None
generation_pipeline = None


# ── FUNCTION: get_generation_pipeline ────────────────────────────────────────
# Lazy-loads the generation model — downloads + loads it only on first call.
# Using a global avoids reloading the model on every question (slow!).
def get_generation_pipeline():
    global generation_pipeline
    if generation_pipeline is None:  # only load once
        generation_pipeline = pipeline(
            "text2text-generation",   # task type
            model=GENERATION_MODEL,   # model name from HuggingFace Hub
            tokenizer=GENERATION_MODEL,
        )
    return generation_pipeline


# ── FUNCTION: load_and_index ──────────────────────────────────────────────────
# Same 4-step pipeline as the paid version, but uses HuggingFace embeddings
# instead of OpenAI — no API call, no cost, runs on CPU.
def load_and_index(pdf_path: str):
        # Step 1: load PDF pages
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

        # Step 2: chunk into 500-char pieces
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(pages)

        # Step 3: embed with the local HuggingFace model (no API call!)
    # HuggingFaceEmbeddings downloads and caches the model on first run
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    store = Chroma.from_documents(chunks, embedding=embeddings)

        # Step 4: return retriever (k=4 — top 4 matching chunks per query)
    return store.as_retriever(search_kwargs={"k": 4})


# ── FUNCTION: build_prompt ────────────────────────────────────────────────────
# In the free version we don't use LangChain's ConversationalRetrievalChain.
# Instead we manually build a prompt string that flan-t5-base can understand.
#
# The prompt has three sections:
#   1. Recent chat history — so the model sees the conversation context
#   2. Retrieved chunks — the relevant parts of the PDF
#   3. The current question
def build_prompt(question: str, docs, history):
        # Format each retrieved chunk with its source page number
    context_parts = []
    for idx, doc in enumerate(docs, start=1):
        page      = doc.metadata.get("page")
        page_text = f"page {page+1}" if isinstance(page, int) else "unknown page"
        context_parts.append(f"[{idx} | {page_text}]\n{doc.page_content}")

        # Use only the last 6 messages from history to keep the prompt short
    # (flan-t5-base has a limited context window of ~512 tokens)
    recent_history = history[-6:] if history else []
    history_lines  = [f"{m['role']}: {m['content']}" for m in recent_history]

    context_text = "\n\n".join(context_parts) if context_parts else "No context found."
    history_text = "\n".join(history_lines) if history_lines else "No prior history."

        # The instruction at the top prevents hallucination:
    # flan-t5 is told to only use the provided context, not its training data
    return (
        "Answer the question using only the context below. "
        "If the answer is not present, say you do not know.\n\n"
        f"Chat history:\n{history_text}\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


# ── upload_pdf and ask_question ───────────────────────────────────────────────
# Same structure as the paid version but simpler — no LangChain chain.
def upload_pdf(pdf_file):
    global vectorstore
    if pdf_file is None:
        return "No file uploaded.", []
    retriever   = load_and_index(pdf_file.name)
    vectorstore = retriever   # store retriever globally for ask_question
    return "PDF indexed. Ask questions now.", []


def ask_question(question, history):
    history = history or []
    if not question:
        return history
    if vectorstore is None:
        return history + [{"role":"user","content":question},
                           {"role":"assistant","content":"Please upload a PDF first."}]

        # 1. Retrieve the top-4 most relevant chunks
    docs   = vectorstore.invoke(question)

        # 2. Build the prompt by combining context + history + question
    prompt = build_prompt(question, docs, history)

        # 3. Run flan-t5-base on the prompt
    # max_new_tokens=220 caps the reply length
    # do_sample=False — greedy decoding (deterministic, faster)
    generator = get_generation_pipeline()
    result    = generator(prompt, max_new_tokens=220, do_sample=False)
    answer    = result[0]["generated_text"].strip()

        # 4. Append source page references
    pages = sorted({d.metadata.get("page") for d in docs
                   if isinstance(d.metadata.get("page"), int)})
    if pages:
        answer += "\n\nSources: pages " + ", ".join(str(p+1) for p in pages)

    return history + [{"role":"user","content":question},
                      {"role":"assistant","content":answer}]


# ── GRADIO UI (same layout as the paid version) ───────────────────────────────
with gr.Blocks(title="Chat with Your PDF (Free)") as demo:
    gr.Markdown("## Chat with Your PDF (Free, local)\nUpload a PDF and ask questions.")
    with gr.Row():
        pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
        status    = gr.Textbox(label="Status", interactive=False)
    chatbot        = gr.Chatbot(label="Conversation", height=400)
    question_input = gr.Textbox(placeholder="Ask about your PDF...")
    pdf_input.change(fn=upload_pdf, inputs=pdf_input, outputs=[status, chatbot])
    question_input.submit(fn=ask_question, inputs=[question_input, chatbot], outputs=chatbot)

if __name__ == "__main__":
    demo.launch()
