"""
eval.py
-------
Evaluation script using Cosine Similarity to compare RAG answers against Ground Truth.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import Settings
from rag_chain import RAGPipeline, DomainFilter

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# DATACLASS: TestCase
# -----------------------------------------------------------------------------

@dataclass
class TestCase:
    """A single evaluation sample with question, expected answer, and domain filter."""
    question: str
    ground_truth: str
    domain: str = DomainFilter.ALL
    use_outlier_logic: bool = False

# -----------------------------------------------------------------------------
# CLASS: TestSuite
# -----------------------------------------------------------------------------

class TestSuite:
    """
    Manages the collection of synthetic test cases used for evaluation.
    Covers Banking, IT, cross-domain, and outlier detection logic.
    """
    @staticmethod
    def get_all_cases() -> List[TestCase]:
        return [
        # ── INFORMATION-TECHNOLOGY ─────────────────────────────────────────
        TestCase(
            question="Which IT candidates have experience with Python programming?",
            ground_truth=(
                "IT candidates with Python experience typically have skills in scripting, "
                "data analysis, web development (Django/Flask), or machine learning "
                "frameworks such as TensorFlow or scikit-learn."
            ),
            domain=DomainFilter.IT,
        ),
        TestCase(
            question="What cloud platform certifications do IT candidates hold?",
            ground_truth=(
                "Several IT candidates hold certifications in AWS (e.g. Solutions Architect), "
                "Microsoft Azure, and/or Google Cloud, demonstrating cloud infrastructure expertise."
            ),
            domain=DomainFilter.IT,
        ),
        TestCase(
            question="Which IT candidates have led software development teams?",
            ground_truth=(
                "IT candidates with team leadership experience hold roles such as Senior Engineer, "
                "Tech Lead, or Engineering Manager, responsible for sprint planning, code review, "
                "and mentoring junior developers."
            ),
            domain=DomainFilter.IT,
        ),
        # ── BANKING ────────────────────────────────────────────────────────
        TestCase(
            question="Which banking candidates have experience in risk management or compliance?",
            ground_truth=(
                "Banking candidates with risk/compliance experience hold backgrounds in credit "
                "risk, market risk, regulatory compliance (Basel III, KYC/AML), internal audit, "
                "or financial controls, often as Risk Analyst or Compliance Officer."
            ),
            domain=DomainFilter.BANKING,
        ),
        TestCase(
            question="Do any banking candidates have SQL or Excel data analysis skills?",
            ground_truth=(
                "Many banking candidates possess SQL for querying financial databases and advanced "
                "Excel skills for financial modelling, pivot tables, VLOOKUP, and dashboards."
            ),
            domain=DomainFilter.BANKING,
        ),
        TestCase(
            question="Which banking candidates have investment banking or capital markets experience?",
            ground_truth=(
                "Banking candidates with investment banking backgrounds may have worked in M&A, "
                "equity research, fixed income, or capital markets, with DCF modelling, Bloomberg, "
                "and client relationship management skills."
            ),
            domain=DomainFilter.BANKING,
        ),
        # ── CROSS-DOMAIN ───────────────────────────────────────────────────
        TestCase(
            question="Which candidates across both domains have project management experience?",
            ground_truth=(
                "Candidates from both domains may have PM experience: IT candidates use "
                "Agile/Scrum; Banking candidates manage regulatory or system-implementation "
                "projects. PMP or PRINCE2 certifications may be present."
            ),
            domain=DomainFilter.ALL,
        ),
        TestCase(
            question="Which candidates are multilingual or have international work experience?",
            ground_truth=(
                "Some candidates mention proficiency in multiple languages or international "
                "work across different countries or multinational organisations."
            ),
            domain=DomainFilter.ALL,
        ),
        # ── OUTLIER ────────────────────────────────────────────────────────
        TestCase(
            question=(
                "Which banking candidates have the most transferable skills for an IT role, particularly in data engineering or analytics?"
            ),
            ground_truth=(
                """Banking candidates best suited for IT data roles demonstrate quantitative 
                analysis, SQL/database skills, experience with financial data pipelines, 
                automation via Excel macros or Python scripting, and systems thinking from 
                core banking or ERP platforms."""
            ),
            domain=DomainFilter.BANKING,
            use_outlier_logic=True,
        ),
    ]

# -----------------------------------------------------------------------------
# CLASS: RAGEvaluator
# -----------------------------------------------------------------------------

class RAGEvaluator:
    """
    Orchestrates evaluation using Cosine Similarity on embeddings.
    Compares the generated answer from the RAG pipeline with the predefined ground truth.
    """
    def __init__(self, settings: Settings):
        self._settings = settings
        self._eval_cfg = settings.evaluation
        
        # Initialize RAG Pipeline to generate answers
        self._pipeline = RAGPipeline(settings)
        
        # Initialize local Embedding Model to calculate vectors
        logger.info(f"Loading local embedding model: {settings.embeddings.model_name}")
        self._embeddings = HuggingFaceEmbeddings(
            model_name=settings.embeddings.model_name,
            model_kwargs={"device": settings.embeddings.device},
            encode_kwargs={"normalize_embeddings": settings.embeddings.normalize},
        )

    def evaluate_with_cosine(self, test_suite: List[TestCase]) -> Dict[str, Any]:
        """
        Evaluates the RAG system by calculating Cosine Similarity between 
        the generated answers and the ground truth.
        """
        results = []
        logger.info(f"Starting Cosine Similarity evaluation for {len(test_suite)} test cases...")

        for i, case in enumerate(test_suite, 1):
            logger.info(f"Processing Test Case {i}/{len(test_suite)}: {case.question[:50]}...")
            
            try:
                # 1. Get Answer from RAG Pipeline
                if case.use_outlier_logic:
                    rag_res = self._pipeline.find_outlier_banking_candidates()
                    answer = str(rag_res) # Convert list/dict result to string for embedding
                else:
                    rag_res = self._pipeline.query(case.question, domain=case.domain)
                    answer = rag_res.get("answer", "")
                    
                # Handle empty answers safely
                if not answer or answer.isspace():
                    answer = "No relevant information found."

                # 2. Generate Embeddings (Vectors)
                vec_answer = np.array(self._embeddings.embed_query(answer)).reshape(1, -1)
                vec_truth = np.array(self._embeddings.embed_query(case.ground_truth)).reshape(1, -1)

                # 3. Calculate Cosine Similarity Score (0.0 to 1.0)
                score = float(cosine_similarity(vec_answer, vec_truth)[0][0])

                results.append({
                    "question": case.question,
                    "domain": case.domain,
                    "is_outlier_logic": case.use_outlier_logic,
                    "cosine_similarity_score": score,
                    "generated_answer": answer[:500] + "..." if len(answer) > 500 else answer,
                    "ground_truth": case.ground_truth
                })
                
                logger.info(f"Result Score: {score:.4f}")

            except Exception as e:
                logger.error(f"Error processing test case '{case.question}': {str(e)}")
                results.append({
                    "question": case.question,
                    "error": str(e),
                    "cosine_similarity_score": 0.0
                })

        # Calculate Aggregate Metrics
        valid_scores = [r["cosine_similarity_score"] for r in results if "cosine_similarity_score" in r]
        avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

        output_report = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "methodology": "Cosine Similarity (Local Embeddings)",
            "embedding_model": self._settings.embeddings.model_name,
            "total_test_cases": len(test_suite),
            "average_similarity_score": avg_score,
            "detailed_results": results
        }

        self._save_results(output_report)
        self._log_summary(output_report)
        
        return output_report

    def _save_results(self, output: Dict[str, Any]) -> None:
        """Saves the evaluation results to a JSON file specified in config."""
        path = Path(self._eval_cfg.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        logger.info(f"Results successfully saved to: {path}")

    @staticmethod
    def _log_summary(report: Dict[str, Any]) -> None:
        """Prints a concise summary of the evaluation to the console."""
        logger.info("=" * 60)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Methodology : {report['methodology']}")
        logger.info(f"Model Used  : {report['embedding_model']}")
        logger.info(f"Total Cases : {report['total_test_cases']}")
        logger.info(f"Avg Score   : {report['average_similarity_score']:.4f} (Max: 1.0)")
        logger.info("=" * 60)


# -----------------------------------------------------------------------------
# CLI ENTRYPOINT
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    
    # Suppress overly verbose logs from libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="CV RAG Evaluation via Cosine Similarity")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    args = parser.parse_args()

    # Load configuration
    cfg = Settings(config_path=args.config).validate()
    
    # Run Evaluation
    evaluator = RAGEvaluator(cfg)
    test_cases = TestSuite.get_all_cases()
    
    # Execute Cosine Similarity Evaluation
    evaluator.evaluate_with_cosine(test_cases)