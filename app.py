"""
app.py
------
Streamlit UI
"""

from __future__ import annotations

import os
import json
import logging
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

import streamlit as st

# Local module imports
from config import Settings, get_settings
from pdf_loader import SmartPDFLoader
from extractor import CVEntityExtractor
from rag_chain import RAGPipeline

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# APP CONTEXT
# ─────────────────────────────────────────────

class AppContext:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._extractor: CVEntityExtractor | None = None
        self._rag: RAGPipeline | None = None

    @property
    def settings(self) -> Settings: return self._settings
    @property
    def extractor(self) -> CVEntityExtractor:
        if self._extractor is None: self._extractor = CVEntityExtractor(self._settings)
        return self._extractor
    @property
    def rag(self) -> RAGPipeline:
        if self._rag is None: self._rag = RAGPipeline(self._settings)
        return self._rag

def get_app_context() -> AppContext:
    if "app_context" not in st.session_state:
        st.session_state.app_context = AppContext(get_settings().validate())
    return st.session_state.app_context


 

# ─────────────────────────────────────────────
# PAGES
# ─────────────────────────────────────────────

class BasePage(ABC):
    @abstractmethod
    def render(self, ctx: AppContext) -> None: pass

class CVAnalyserPage(BasePage):
    def render(self, ctx: AppContext) -> None:
        st.header(":clipboard: CV Extractor")
        st.info("Upload a resume PDF (digital or scanned). "
                "The system will extract its text and let you export structured JSON.")
        uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
        
        if uploaded_file:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Text Extraction")
                with st.spinner("Processing..."):
                    st.write("Image-based PDF may take longer to extract")
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                    
                    loader = SmartPDFLoader(tmp_path)
                    docs = loader.load()
                    raw_text = "\n\n".join([doc.page_content for doc in docs])
                    
                    st.text_area("Markdown Output", raw_text, height=400)
                    os.unlink(tmp_path)

            with col2:
                st.subheader("Structured Entities (JSON)")
                if st.button("Extract into JSON"):
                    with st.spinner("Analyzing and Extracting..."):
                        entity = ctx.extractor.extract(raw_text)
                        if entity:
                            st.json(entity.model_dump())
                            st.download_button(
                                label=":arrow_down: Download JSON",
                                data=entity.model_dump_json(indent=2),
                                file_name=f"{uploaded_file.name}_extracted.json",
                                mime="application/json"
                            )

class HRChatPage(BasePage):
    def render(self, ctx: AppContext) -> None:
        st.header(":speech_balloon: HR RAG Chat")
        domain_choice = st.radio("Search within domain:", ["ALL", "BANKING", "INFORMATION-TECHNOLOGY"], horizontal=True)

        if "messages" not in st.session_state: st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "sources" in message and message["sources"]:
                    with st.expander("Reference Sources"):
                        for src in message["sources"]:
                            st.markdown(f"**File:** {src.get('source', 'N/A')} | **Category:** {src.get('category', 'N/A')}")
                            st.caption(f"...{src.get('snippet', '')}...")

        if prompt := st.chat_input("Ask about candidates..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Querying vector database..."):
                    result = ctx.rag.query(prompt, domain=domain_choice)
                    st.markdown(result["answer"])
                    st.session_state.messages.append({"role": "assistant", "content": result["answer"], "sources": result.get("sources", [])})

class OutlierPage(BasePage): # Giả sử bạn đang kế thừa BasePage
    def render(self, ctx: AppContext) -> None:
        st.header(":mag: Cross-Domain Outlier Finder")
        st.info(
            "Identifies the **Top 5 Banking candidates** who have transferable skills "
            "and can be trained for Information Technology roles.\n\n"
            "After analysis, you can download each candidate's original PDF resume."
        )
        if st.button(":rocket: Run Outlier Analysis"):
            with st.spinner("Scanning for potential candidates..."):
                result = ctx.rag.find_outlier_banking_candidates()
                
                answer_text = result.get('answer', "No result found.")
                st.markdown(answer_text)
                
                source_docs = result.get('source_documents', [])
                if source_docs:
                    st.divider()
                    st.subheader("📚 Source Resumes Analyzed")
                    
                    unique_sources = {}
                    for doc in source_docs:
                        src_file = doc.get('source', 'Unknown')
                        if src_file in answer_text:
                            if src_file not in unique_sources:
                                unique_sources[src_file] = doc
                    
                    for src_file, doc_info in unique_sources.items():
                        category = doc_info.get('category', 'N/A')
                        snippet = doc_info.get('snippet', '')
                        
                        with st.expander(f"📄 File: {src_file} | Category: {category}"):
                            st.write(f"_{snippet}_")

# ─────────────────────────────────────────────
# MAIN APP ORCHESTRATOR
# ─────────────────────────────────────────────

class App:

    @classmethod
    def run(cls) -> None:
        st.set_page_config(page_title="CV Assistant", page_icon=":bookmark_tabs:", layout="wide")
        
        ctx = get_app_context()

            

        tab_analyser, tab_chat, tab_outlier = st.tabs([
            ":clipboard: CV Extractor", 
            ":speech_balloon: HR Chat", 
            ":mag: Outlier Candidates"
        ])

        with tab_analyser:
            CVAnalyserPage().render(ctx)
        
        with tab_chat:
            HRChatPage().render(ctx)
            
        with tab_outlier:
            OutlierPage().render(ctx)


if __name__ == "__main__":
    App.run()