# ─────────────────────────────────────────────────────────────────────────────
# rag_pdf.py  —  Chat with any PDF using OpenAI + LangChain + ChromaDB
#
# WHAT THIS APP DOES:
#   1. You upload a PDF through a Gradio web UI
#   2. The PDF is split into small chunks and embedded (converted to numbers)
#   3. Those embeddings are stored in ChromaDB (an in-memory vector database)
#   4. When you ask a question, ChromaDB finds the most relevant chunks
#   5. Those chunks + your question are sent to GPT-4o-mini to generate an answer
# ─────────────────────────────────────────────────────────────────────────────

import os       # reads environment variables (the API key)
import gradio as gr  # builds the web UI — no HTML needed

# LangChain components — each does one specific job in the pipeline:
from langchain_community.document_loaders import PyPDFLoader
# ↑ reads a PDF file and returns a list of Document objects (one per page)

from langchain_text_splitters import RecursiveCharacterTextSplitter
# ↑ splits long pages into smaller chunks that fit in the LLM context window
#   "Recursive" means it tries to split on paragraphs first, then sentences, then words

from langchain_chroma import Chroma
# ↑ ChromaDB integration — stores and searches embedding vectors

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# ↑ OpenAIEmbeddings: converts text → 1536-dimension float vectors via OpenAI API
#   ChatOpenAI: the GPT model client (here we use gpt-4o-mini)

from langchain_classic.chains import ConversationalRetrievalChain
# ↑ a pre-built LangChain chain that:
#   - reformulates the question using chat history
#   - retrieves relevant documents
#   - passes everything to the LLM

from langchain_classic.memory import ConversationBufferMemory
# ↑ stores the full conversation (all Q&A turns) and injects it into each prompt
#   so the model remembers what was said earlier in the chat

from dotenv import load_dotenv
# ↑ reads the .env file and sets environment variables

load_dotenv()  # must be called early — before any OpenAI clients are created


# ── FUNCTION: load_and_index ──────────────────────────────────────────────────
# Takes a PDF file path, runs the full indexing pipeline, returns a retriever.
# This is called once when the user uploads a PDF.
#
# PIPELINE:  PDF → pages → chunks → embeddings → ChromaDB → retriever
def load_and_index(pdf_path: str):

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is not set. Add it to .env")

        # STEP 1 — Load PDF
    # PyPDFLoader reads the file and returns one Document object per page.
    # Each Document has .page_content (text) and .metadata ({"page": 0, ...})
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"Loaded {len(pages)} pages")

        # STEP 2 — Chunk the text
    # chunk_size=500 means each chunk is at most 500 characters.
    # chunk_overlap=50 means chunks share 50 characters at their edges.
    # Overlap helps the LLM see context that might span two chunks.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(pages)
    print(f"Split into {len(chunks)} chunks")

        # STEP 3 — Embed and store in ChromaDB
    # OpenAIEmbeddings calls the OpenAI API for each chunk and gets back a vector.
    # Chroma.from_documents() stores all (chunk, vector) pairs in memory.
    # This creates a searchable vector index — the "database" of our PDF.
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(chunks, embedding=embeddings)

        # STEP 4 — Return a retriever
    # A retriever is an object with a .invoke(question) method.
    # k=4 means: return the 4 chunks most similar to the question.
    return vectorstore.as_retriever(search_kwargs={"k": 4})


# ── FUNCTION: build_chain ─────────────────────────────────────────────────────
# Wraps the retriever in a LangChain conversational RAG chain.
# This is called once after indexing — the chain is reused for every question.
def build_chain(retriever):

        # The LLM that generates the final answer.
    # temperature=0 → deterministic (same question = same answer each time)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # ConversationBufferMemory stores every (question, answer) pair.
    # memory_key="chat_history" tells LangChain where to inject it in the prompt.
    # return_messages=True keeps them as message objects (not plain strings).
    # output_key="answer" tells memory which field in the chain output to save.
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

        # ConversationalRetrievalChain combines three things:
    #   1. Memory — injects chat history so the model understands follow-up questions
    #   2. Retriever — finds relevant PDF chunks for each question
    #   3. LLM — reads the chunks + question + history and writes the answer
    # return_source_documents=True means we also get which chunks were used.
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=False
    )
    return chain


# Global variable — stores the chain between Gradio events.
# None means no PDF has been uploaded yet.
chain = None


# ── FUNCTION: upload_pdf ──────────────────────────────────────────────────────
# Triggered when the user uploads a PDF in the Gradio UI.
# pdf_file is a Gradio UploadedFile object — .name gives the temp file path.
def upload_pdf(pdf_file):
    global chain    # we modify the global chain variable
    if pdf_file is None:
        return "No file uploaded.", []

    retriever = load_and_index(pdf_file.name)  # index the PDF
    chain = build_chain(retriever)             # build the RAG chain
    return "✅ PDF indexed! Ask me anything about it.", []
        # returns: (status text, empty chat history)


# ── FUNCTION: ask_question ────────────────────────────────────────────────────
# Triggered when the user types a question and presses Enter.
# history is a list of {"role": "user"/"assistant", "content": "..."} dicts.
def ask_question(question, history):
    global chain
    history = history or []

    if chain is None:  # PDF not uploaded yet
        return history + [
            {"role": "user",      "content": question},
            {"role": "assistant", "content": "Please upload a PDF first."},
        ]

        # chain.invoke() runs the full pipeline:
    # 1. Reformulates question with memory context
    # 2. Retrieves top-4 relevant chunks from ChromaDB
    # 3. Calls GPT-4o-mini with chunks + question + history
    # Returns: {"answer": "...", "source_documents": [...]}
    result = chain.invoke({"question": question})
    answer = result["answer"]

        # Append source page numbers to the answer for transparency
    sources = result.get("source_documents", [])
    if sources:
                # .metadata["page"] is 0-indexed, so we add 1 for human-readable page numbers
        pages = set(doc.metadata.get("page", "?") for doc in sources)
        answer += f"\n\n_Sources: pages {', '.join(str(p+1) for p in sorted(pages))}_"

    return history + [
        {"role": "user",      "content": question},
        {"role": "assistant", "content": answer},
    ]


# ── GRADIO UI ─────────────────────────────────────────────────────────────────
# gr.Blocks() lets you compose custom layouts.
# Each gr.* component maps to an HTML element — no frontend code needed.
with gr.Blocks(title="Chat with Your PDF") as demo:
    gr.Markdown("## 📄 Chat with Your PDF\nUpload a PDF, then ask questions.")

    with gr.Row():  # puts components side-by-side
        pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
        status    = gr.Textbox(label="Status", interactive=False)

    chatbot        = gr.Chatbot(label="Conversation", height=400)
    question_input = gr.Textbox(placeholder="Ask about your PDF...")

        # .change() fires when the file input changes (PDF is uploaded)
    # fn=upload_pdf  → the function to call
    # inputs=pdf_input → what to pass as the argument
    # outputs=[status, chatbot] → what the function returns updates these components
    pdf_input.change(fn=upload_pdf, inputs=pdf_input, outputs=[status, chatbot])

        # .submit() fires when user presses Enter in the textbox
    question_input.submit(
        fn=ask_question,
        inputs=[question_input, chatbot],
        outputs=chatbot
    )

        # Pre-filled example questions the user can click
    gr.Examples(
        examples=[["What is this document about?"], ["Summarize the main points."]],
        inputs=question_input
    )

if __name__ == "__main__":
    demo.launch()  # opens http://localhost:7860 in your browser