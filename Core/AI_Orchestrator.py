"""Core/AI_Orchestrator.py — Assistente IA da Boston Aerospace AI (somente OpenAI)

Para o lancamento do contest, este modulo substitui o RAG antigo (Ollama-first)
por um fluxo unico, mais simples e mais confiavel:

    Pergunta do engenheiro
            |
            v
    +----------------+     +------------------+     +----------------+
    | ML prediction  |     | RAG interno       |     | Web Search     |
    | (RUL + SHAP,   |     | (manuais tecnicos |     | (OpenAI tool,  |
    |  se fornecido) |     |  indexados, se     |     |  se ativado)   |
    |                |     |  houver algum)     |     |                |
    +--------+-------+     +---------+----------+     +--------+-------+
             \\                       |                        /
              \\_______________________|_______________________/
                                       v
                              GPT-5.6 (OpenAI Responses API)
                                       v
                          Resposta estruturada (JSON) + texto

Importante: o modelo NUNCA calcula o RUL nem substitui o teu ML. Ele so
interpreta/explica o que o teu ensemble + SHAP ja calcularam, cruza com
documentacao interna (se existir) e pesquisa externa (se necessario), e
organiza tudo numa resposta que separa claramente as fontes.

Funciona com OU sem uma base de conhecimento indexada: se o ChromaDB estiver
vazio (nenhum manual foi enviado ainda via /api/kb-upload), o assistente
simplesmente responde com base no ML/SHAP fornecidos e/ou na pesquisa web.
"""
import json
from Config.Settings import (
    OPENAI_API_KEY, OPENAI_MODEL, OPENAI_EMBED_MODEL, CHROMA_DIR, SYSTEM_PROMPT,
)

_client = None
_vs = None
_ai_ok = False

ORCHESTRATOR_INSTRUCTIONS = SYSTEM_PROMPT + """

Voce e o orquestrador de inteligencia de manutencao da Boston Aerospace AI.
Voce recebe ate tres tipos de evidencia, quando disponiveis:
  1. ML_DATA: a predicao de RUL (Remaining Useful Life) e os fatores SHAP
     mais importantes, calculados pelo ensemble de machine learning do
     sistema. Isto e FATO, nao invente nem corrija estes numeros.
  2. INTERNAL_DOCS: trechos de manuais tecnicos indexados pela empresa.
     Se nao houver nenhum, ignore esta secao (pode estar vazia).
  3. Pesquisa na Web (ferramenta web_search): use apenas quando a pergunta
     precisar de contexto externo (ex: padroes conhecidos de degradacao de
     um sensor, boletins de servico, literatura tecnica). Nao pesquise para
     perguntas simples que ML_DATA ja responde.

Responda SEMPRE em formato JSON valido, seguindo exatamente este schema
(sem markdown, sem texto fora do JSON):
{
  "summary": "resposta direta e curta em linguagem natural, 1-3 frases",
  "risk_level": "low" | "medium" | "high" | "critical" | "unknown",
  "predicted_rul": numero ou null,
  "evidence": ["ponto 1", "ponto 2", ...],
  "shap_factors": ["fator 1", "fator 2", ...],
  "internal_sources": ["nome do documento", ...],
  "web_sources": ["url ou descricao da fonte", ...],
  "recommended_investigation": ["acao 1", "acao 2", ...],
  "confidence": "low" | "medium" | "high"
}

Nota: nao inclua um campo "data_sources" — isso e calculado automaticamente
pelo sistema a partir do que foi de fato usado (nao confie no que o modelo
"acha" que usou).

Regras importantes:
- Nunca misture o que veio do ML/SHAP com o que veio da pesquisa web ou dos
  manuais internos sem identificar a origem em "evidence"/"internal_sources"/
  "web_sources".
- Nunca prescreva uma acao de manutencao definitiva (ex: "troque a peca X
  agora"); sempre enquadre como recomendacao de investigacao para um
  engenheiro qualificado decidir.
- Se nao houver ML_DATA disponivel, responda normalmente ao que for
  perguntado, mas deixe "predicted_rul": null e explique no summary que a
  pergunta nao esta associada a uma predicao especifica.
- Se nao houver INTERNAL_DOCS nem necessidade de pesquisa web, deixe essas
  listas vazias — o sistema funciona igualmente bem so com ML_DATA ou so
  com conhecimento geral.
"""


def init_ai() -> bool:
    """Inicializa o cliente OpenAI e (se possivel) o indice de conhecimento interno."""
    global _client, _vs, _ai_ok
    if _ai_ok:
        return True
    if not OPENAI_API_KEY or not OPENAI_MODEL:
        print("[AI] OPENAI_API_KEY ou OPENAI_MODEL nao configurados")
        return False
    try:
        from openai import OpenAI
        _client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        print(f"[AI] falha ao criar cliente OpenAI: {e}")
        return False

    # O indice interno (RAG) e opcional: se falhar ou estiver vazio, seguimos sem ele.
    try:
        from langchain_openai import OpenAIEmbeddings
        from langchain_community.vectorstores import Chroma
        emb = OpenAIEmbeddings(model=OPENAI_EMBED_MODEL, api_key=OPENAI_API_KEY)
        _vs = Chroma(persist_directory=CHROMA_DIR, embedding_function=emb)
    except Exception as e:
        print(f"[AI] indice interno (RAG) indisponivel, seguindo so com ML/Web: {e}")
        _vs = None

    _ai_ok = True
    print(f"[AI] provider=openai modelo={OPENAI_MODEL}")
    return True


def _internal_docs(question: str, k: int = 3):
    """Busca trechos relevantes na base interna, se ela existir e tiver conteudo."""
    if _vs is None:
        return [], []
    try:
        results  = _vs.similarity_search_with_score(question, k=k)
        relevant = [(d, s) for d, s in results if s < 1.2]
        chunks   = [f"[{d.metadata.get('source','')}]:\n{d.page_content}" for d, _ in relevant]
        sources  = list(dict.fromkeys(d.metadata.get("source", "") for d, _ in relevant if d.metadata.get("source")))
        return chunks, sources
    except Exception:
        return [], []


def _web_search_was_used(response) -> bool:
    """Detecta de forma confiavel (nao pelo texto do modelo) se a ferramenta
    web_search foi de fato chamada nesta resposta."""
    try:
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) == "web_search_call":
                return True
    except Exception:
        pass
    return False


def _extract_web_sources(response) -> list:
    """Extrai URLs citados quando a ferramenta web_search foi usada."""
    sources = []
    try:
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) == "message":
                for c in getattr(item, "content", []) or []:
                    for ann in getattr(c, "annotations", []) or []:
                        url = getattr(ann, "url", None)
                        if url:
                            sources.append(url)
    except Exception:
        pass
    return list(dict.fromkeys(sources))


def _parse_structured(text: str) -> dict:
    """Tenta interpretar a resposta como JSON; se falhar, devolve um fallback seguro."""
    try:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
        return json.loads(cleaned)
    except Exception:
        return {
            "summary": text.strip(),
            "risk_level": "unknown",
            "predicted_rul": None,
            "evidence": [],
            "shap_factors": [],
            "internal_sources": [],
            "web_sources": [],
            "recommended_investigation": [],
            "confidence": "low",
        }


def ask(question: str, history: list = None, ml_context: dict = None,
        use_web_search: bool = True) -> dict:
    """Orquestra ML + RAG interno + Web Search numa unica resposta estruturada.

    ml_context: dict opcional com o resultado de Core.Predictor.predict_rul()
                (rul, status, shap, etc.) para a pergunta atual, se aplicavel.
    """
    if not init_ai():
        raise RuntimeError("OpenAI nao configurado (defina OPENAI_API_KEY e OPENAI_MODEL no .env)")

    history = history or []
    internal_chunks, internal_sources = _internal_docs(question)

    parts = []
    if ml_context:
        parts.append("ML_DATA (predicao do ensemble de ML + SHAP, ja calculada — nao recalcule):\n"
                      + json.dumps(ml_context, ensure_ascii=False))
    if internal_chunks:
        parts.append("INTERNAL_DOCS (trechos de manuais tecnicos indexados):\n"
                      + "\n\n".join(internal_chunks))
    for h in history[-10:]:
        role = "Engenheiro" if h.get("role") == "user" else "Boston AI"
        parts.append(f"{role} (historico): {h.get('content','')}")
    parts.append(f"Pergunta atual do engenheiro: {question}")

    tools = [{"type": "web_search"}] if use_web_search else []

    response = _client.responses.create(
        model=OPENAI_MODEL,
        instructions=ORCHESTRATOR_INSTRUCTIONS,
        input="\n\n---\n\n".join(parts),
        tools=tools,
        # tool_choice="auto" (padrao): disponibilizar a ferramenta NAO forca o
        # uso dela — o proprio modelo decide, por pergunta, se precisa
        # pesquisar na web ou se ML/SHAP/docs internos ja bastam. Ou seja,
        # "web_search" ligado por padrao != "toda pergunta pesquisa a web".
        tool_choice="auto" if use_web_search else "none",
    )

    structured = _parse_structured(response.output_text)
    web_was_used = _web_search_was_used(response)
    web_sources = _extract_web_sources(response) if web_was_used else []
    if web_sources and not structured.get("web_sources"):
        structured["web_sources"] = web_sources
    if not web_was_used:
        structured["web_sources"] = []  # nunca deixar o modelo "inventar" fontes web nao usadas
    if internal_sources and not structured.get("internal_sources"):
        structured["internal_sources"] = internal_sources

    # Calculado pelo sistema, nao pelo modelo — reflete o que de fato foi usado.
    structured["data_sources"] = {
        "ml": bool(ml_context),
        "internal_docs": bool(internal_chunks),
        "web": web_was_used,
    }

    return {
        "answer": structured.get("summary", response.output_text),
        "structured": structured,
        "sources": list(dict.fromkeys(
            (structured.get("internal_sources") or []) + (structured.get("web_sources") or [])
        )),
    }


def add_document(path: str, fname: str) -> dict:
    """Indexa um manual tecnico na base interna (mesma logica de antes, so com embeddings OpenAI)."""
    if not init_ai():
        return {"error": "OpenAI nao configurado"}
    if _vs is None:
        return {"error": "Indice interno indisponivel"}
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


def get_status() -> dict:
    return {"provider": "openai", "model": OPENAI_MODEL, "initialized": _ai_ok}
