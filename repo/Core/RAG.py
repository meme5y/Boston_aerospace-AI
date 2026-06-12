"""Core/RAG.py — RAG com Ollama + ChromaDB + LangChain"""
from Config.Settings import OLLAMA_URL, OLLAMA_MODEL, EMBED_MODEL, CHROMA_DIR, SYSTEM_PROMPT

_rag_ok = False
_vs = _llm = _emb = None


def init_rag() -> bool:
    global _rag_ok, _vs, _llm, _emb
    if _rag_ok:
        return True
    try:
        from langchain_ollama import ChatOllama
        from langchain_community.embeddings import OllamaEmbeddings
        from langchain_community.vectorstores import Chroma
        _emb    = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)
        _vs     = Chroma(persist_directory=CHROMA_DIR, embedding_function=_emb)
        _llm    = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_URL, temperature=0.7)
        _rag_ok = True
        print(f"[RAG] modelo={OLLAMA_MODEL} embed={EMBED_MODEL}")
        return True
    except Exception as e:
        print(f"[RAG] falhou: {e}")
        return False


def rag_chat(message: str, history: list) -> tuple:
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    context, sources = "", []
    try:
        results  = _vs.similarity_search_with_score(message, k=3)
        relevant = [(d, s) for d, s in results if s < 1.2]
        if relevant:
            context = "\n\n".join(f"[{d.metadata.get('source','')}]:\n{d.page_content}" for d, _ in relevant)
            sources = list(dict.fromkeys(d.metadata.get("source","") for d, _ in relevant if d.metadata.get("source","")))
    except Exception:
        pass

    msgs = [SystemMessage(content=SYSTEM_PROMPT)]
    if context:
        msgs.append(SystemMessage(content=f"Base de conhecimento:\n{context}"))
    for h in history[-10:]:
        if h.get("role") == "user":
            msgs.append(HumanMessage(content=h["content"]))
        elif h.get("role") == "assistant":
            msgs.append(AIMessage(content=h["content"]))
    msgs.append(HumanMessage(content=message))
    resp = _llm.invoke(msgs)
    return resp.content, sources


def add_document(path: str, fname: str) -> dict:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        ext = fname.rsplit(".", 1)[-1].lower()
        if ext == "pdf":
            from pypdf import PdfReader
            text = "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)
        else:
            with open(path, "r", errors="ignore") as f:
                text = f.read()
        if not text.strip():
            return {"error": "Documento vazio"}
        chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150).split_text(text)
        _vs.add_texts(texts=chunks, metadatas=[{"source": fname}] * len(chunks))
        return {"success": True, "chunks": len(chunks), "filename": fname}
    except Exception as e:
        return {"error": str(e)}
