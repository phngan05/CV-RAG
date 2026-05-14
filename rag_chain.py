"""
rag_chain.py
------------
RAG pipeline — OOP refactor.

Classes:
    DomainFilter   — enum-like class for type-safe domain filter values
    PromptLibrary  — centralises all prompt templates
    VectorRetriever — wraps Pinecone with domain-filtered retrieval
    RAGPipeline    — orchestrates retrieval + generation; public query interface

"""

from __future__ import annotations

import logging
from typing import Literal, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from config import Settings

logger = logging.getLogger(__name__)

DomainLiteral = Literal["BANKING", "INFORMATION-TECHNOLOGY", "ALL"]


# ─────────────────────────────────────────────
# CLASS: DomainFilter
# ─────────────────────────────────────────────

class DomainFilter:
    """
    Type-safe constants and helper for Pinecone metadata filters.
    Keeps filter construction in one place so it's easy to change.
    """

    BANKING = "BANKING"
    IT = "INFORMATION-TECHNOLOGY"
    ALL = "ALL"

    VALID = {BANKING, IT, ALL}

    @staticmethod
    def to_pinecone_filter(domain: str) -> dict | None:
        """
        Convert a domain string to a Pinecone metadata filter dict.
        Returns None for 'ALL' (meaning: no filter applied).
        """
        if domain == DomainFilter.ALL:
            return None
        if domain not in DomainFilter.VALID:
            raise ValueError(f"Invalid domain '{domain}'. Must be one of {DomainFilter.VALID}")
        return {"category": {"$eq": domain}}

    @staticmethod
    def validate(domain: str) -> str:
        if domain not in DomainFilter.VALID:
            raise ValueError(f"Invalid domain: {domain!r}. Choose from {DomainFilter.VALID}")
        return domain


# ─────────────────────────────────────────────
# CLASS: PromptLibrary
# ─────────────────────────────────────────────

class PromptLibrary:
    """
    Centralises all ChatPromptTemplates used by the RAG pipeline.
    """

    GENERAL_RAG: ChatPromptTemplate = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert HR assistant helping to analyse candidate resumes.
Use the retrieved resume context below to answer the user's question accurately and concisely.

Guidelines:
- Ground your answer strictly in the provided context.
- If the context lacks sufficient information, say so honestly.
- When mentioning candidates, include their name and relevant details.
- Format lists clearly using bullet points.

CONTEXT:
{context}
""",
        ),
        ("human", "{question}"),
    ])

    OUTLIER_BANKING_TO_IT: ChatPromptTemplate = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert HR talent analyst specialising in cross-domain candidate assessment.

    You have been provided with resume excerpts from BANKING domain candidates.
    Identify the TOP 5 DISTINCT candidates (have different source and file_path) who, despite a banking background, demonstrate strong
    TRANSFERABLE SKILLS making them ideal for training in Information Technology roles.
    
    Transferable skills to look for:
    - Quantitative / analytical ability (statistics, modelling, financial analysis, mathematics)
    - Data handling (SQL, Excel advanced, data pipelines, reporting tools)
    - Logic and structured thinking (risk frameworks, compliance logic, algorithmic design)
    - Project management (Agile, waterfall, cross-team coordination)
    - Technical aptitude (programming, scripting, automation, banking software: SWIFT, Bloomberg, etc.)
    - Problem-solving under constraints (regulatory compliance, system migrations)
    - Systems thinking (core banking systems, ERP, CRM experience)
    
    For each recommended candidate provide:
    1. **Current Banking Role / Skills** — their core banking profile
    2. **Transferable IT Skills** — specific evidence from the resume
    3. **Trainability Assessment** — 2-3 sentences of justification
    4. **Recommended IT Roles** — 2-3 specific roles they could transition into

    Rank #1 (most suitable) to #5.  Reference actual details from the resumes.

    CONTEXT (Banking Candidate Resumes):
    {context}
""",
        ),
        (
            "human",
            "Analyse the provided banking resumes and identify the Top 5 most trainable candidates for IT roles.",
        ),
    ])

    @staticmethod
    def format_docs(docs: list) -> str:
        """Format a list of retrieved Documents into a single context string."""
        parts = []
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            category = doc.metadata.get("category", "unknown")
            parts.append(
                f"[Source: {source} | Category: {category}]\n{doc.page_content}"
            )
        return "\n\n---\n\n".join(parts)


# ─────────────────────────────────────────────
# CLASS: VectorRetriever
# ─────────────────────────────────────────────

class VectorRetriever:
    """
    Wraps PineconeVectorStore with domain-filtered retrieval.
    Responsible only for fetching relevant documents; generation is elsewhere.
    """

    def __init__(self, settings: Settings):
        self._pc_cfg  = settings.pinecone
        self._api_cfg = settings.api
        self._ret_cfg = settings.retrieval
        self._vectorstore = self._connect()

    # ── Public ─────────────────────────────────────────────────────────────

    def get_retriever(self, domain: str = DomainFilter.ALL, k: int | None = None):
        """
        Return a LangChain retriever with optional domain metadata filter.

        Args:
            domain: 'BANKING', 'INFORMATION-TECHNOLOGY', or 'ALL'.
            k:      Number of chunks to retrieve (defaults to config value).
        """
        k = k or self._ret_cfg.default_k
        metadata_filter = DomainFilter.to_pinecone_filter(domain)

        search_kwargs: dict = {"k": k}
        if metadata_filter:
            search_kwargs["filter"] = metadata_filter

        return self._vectorstore.as_retriever(
            search_type=self._ret_cfg.search_type,
            search_kwargs=search_kwargs,
        )

    def retrieve(self, query: str, domain: str = DomainFilter.ALL, k: int | None = None) -> list:
        """Directly retrieve documents for a query (no generation)."""
        return self.get_retriever(domain=domain, k=k).invoke(query)

    # ── Private ────────────────────────────────────────────────────────────

    def _connect(self) -> PineconeVectorStore:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        emb_cfg = None
        # We need to access embeddings config — retrieve via stored api_cfg parent
        # This is resolved by passing full Settings to the constructor
        raise NotImplementedError("Use VectorRetriever.from_settings() instead.")

    @classmethod
    def from_settings(cls, settings: Settings) -> "VectorRetriever":
        """Factory method: builds embeddings and connects to Pinecone."""
        instance = object.__new__(cls)
        instance._pc_cfg  = settings.pinecone
        instance._api_cfg = settings.api
        instance._ret_cfg = settings.retrieval

        emb_cfg = settings.embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name=emb_cfg.model_name,
            model_kwargs={"device": emb_cfg.device},
            encode_kwargs={"normalize_embeddings": emb_cfg.normalize},
        )

        instance._vectorstore = PineconeVectorStore(
            index_name=settings.pinecone.index_name,
            embedding=embeddings,
            pinecone_api_key=settings.api.pinecone_api_key,
        )
        logger.info("✓ Connected to Pinecone index '%s'", settings.pinecone.index_name)
        return instance


# ─────────────────────────────────────────────
# CLASS: RAGPipeline
# ─────────────────────────────────────────────

class RAGPipeline:
    """
    End-to-end RAG pipeline: retrieval (Pinecone) + generation (LLAMA 3.3).

    Public interface:
        query(question, domain)            — domain-filtered Q&A
        find_outlier_banking_candidates()  — cross-domain outlier analysis
        find_similar_candidates(jd, domain)— similarity search without generation
    """

    # Broad query used to pull diverse Banking profiles for the outlier analysis
    _OUTLIER_SEED_QUERY = (
        "banking finance data analysis SQL programming skills project management "
        "quantitative analytical technical automation reporting"
    )

    def __init__(self, settings: Settings):
        self._settings   = settings
        self._llm_cfg    = settings.llm
        self._api_cfg    = settings.api
        self._ret_cfg    = settings.retrieval
        self._prompts    = PromptLibrary()
        self._retriever  = VectorRetriever.from_settings(settings)
        self._llm        = self._build_llm(temperature=self._llm_cfg.rag_temperature)
        self._output_parser = StrOutputParser()

    # ── Public ─────────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        domain: str = DomainFilter.ALL,
        k: int | None = None,
    ) -> dict:
        """
        Answer a natural-language HR question using RAG.

        Args:
            question: The HR user's question.
            domain:   Domain filter ('ALL', 'BANKING', 'INFORMATION-TECHNOLOGY').
            k:        Override default retrieval k.

        Returns:
            {"answer": str, "source_documents": list[dict]}
        """
        DomainFilter.validate(domain)
        k = k or self._ret_cfg.default_k
        logger.info("RAG Query | domain=%s | q=%s", domain, question[:80])

        retriever = self._retriever.get_retriever(domain=domain, k=k)

        chain = (
            {
                "context": retriever | RunnableLambda(PromptLibrary.format_docs),
                "question": RunnablePassthrough(),
            }
            | PromptLibrary.GENERAL_RAG
            | self._llm
            | self._output_parser
        )

        answer = chain.invoke(question)
        docs   = retriever.invoke(question)

        return {
            "answer": answer,
            "source_documents": self._serialise_docs(docs),
        }

    def find_outlier_banking_candidates(self, k: int | None = None) -> dict:
        """
        Identify top Banking candidates with transferable IT skills.

        Retrieves a broad set of Banking resumes and asks the LLM to rank
        the top 5 by transferable skills using the OUTLIER_BANKING_TO_IT prompt.

        Returns:
            {"answer": str, "source_documents": list[dict]}
        """
        k = k or self._ret_cfg.outlier_k
        logger.info("Running OUTLIER query (Banking → IT), k=%d", k)

        docs    = self._retriever.retrieve(self._OUTLIER_SEED_QUERY, domain=DomainFilter.BANKING, k=k)
        grouped_docs = {}
        for d in docs:
            source = d.metadata.get("source", "Unknown")
            if source not in grouped_docs:
                grouped_docs[source] = []
            grouped_docs[source].append(d.page_content)
            
        merged_context = ""
        for source, contents in grouped_docs.items():
            combined_text = "\n".join(contents)
            merged_context += f"\nSOURCE FILE: {source}\nCONTENT:\n{combined_text}\n"
            merged_context += "\n" + "="*30 + "\n"
        
        outlier_llm = self._build_llm(temperature=self._llm_cfg.outlier_temperature)

        prompt_value = PromptLibrary.OUTLIER_BANKING_TO_IT.invoke({"context": merged_context})
        answer = self._output_parser.invoke(outlier_llm.invoke(prompt_value))

        return {
            "answer": answer,
            "source_documents": self._serialise_docs(docs),
        }

    def find_similar_candidates(
        self,
        job_description: str,
        domain: str = DomainFilter.ALL,
        k: int | None = None,
    ) -> List[dict]:
        """
        Semantic search: return the k most similar candidates for a job description.
        No generation step — purely vector similarity.

        Returns:
            List of ranked candidate dicts with snippets.
        """
        k = k or self._ret_cfg.similarity_k
        docs = self._retriever.retrieve(job_description, domain=domain, k=k)
        return [
            {
                "rank": i + 1,
                "source": d.metadata.get("source", "N/A"),
                "category": d.metadata.get("category", "N/A"),
                "relevance_snippet": d.page_content[:500],
            }
            for i, d in enumerate(docs)
        ]

    # ── Private ────────────────────────────────────────────────────────────

    def _build_llm(self, temperature: float) -> ChatGroq:
        return ChatGroq(
            model=self._llm_cfg.model,
            api_key=self._api_cfg.groq_api_key,
            temperature=temperature,
        )

    @staticmethod
    def _serialise_docs(docs: list) -> List[dict]:
        return [
            {
                "source":   d.metadata.get("source", "N/A"),
                "category": d.metadata.get("category", "N/A"),
                "snippet":  d.page_content[:300] + "...",
            }
            for d in docs
        ]


# ─────────────────────────────────────────────
# CLI DEMO
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from config import Settings

    cfg = Settings().validate()
    rag = RAGPipeline(cfg)

    r1 = rag.query("Which candidates have SQL and data analysis skills?", domain="ALL")
    print("=== GENERAL QUERY ===\n", r1["answer"])

    r2 = rag.find_outlier_banking_candidates()
    print("\n=== OUTLIER: BANKING → IT ===\n", r2["answer"])