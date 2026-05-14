"""
extractor.py
------------
CV entity extraction — OOP refactor.

Classes:
    CVEntity          — Pydantic v2 schema for a structured resume
    CVEntityExtractor — LangChain + LLAMA extraction chain, config-injected

All configuration is injected via a Settings instance; no os.getenv() here.

Usage:
    from config import Settings
    from extractor import CVEntityExtractor

    cfg = Settings().validate()
    extractor = CVEntityExtractor(cfg)
    entity = extractor.extract(raw_text)
    print(entity.model_dump_json(indent=2))
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional

from pydantic import BaseModel, Field

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException

from config import Settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# PYDANTIC SCHEMA
# ─────────────────────────────────────────────

class EducationEntry(BaseModel):
    degree: Optional[str] = Field(None, description="Degree title, e.g. 'B.Sc. Computer Science'")
    institution: Optional[str] = Field(None, description="University or school name")
    graduation_year: Optional[str] = Field(None, description="Year of graduation or expected graduation")
    gpa: Optional[str] = Field(None, description="GPA or grade, if mentioned")


class WorkExperienceEntry(BaseModel):
    company: Optional[str] = Field(None, description="Company or organisation name")
    title: Optional[str] = Field(None, description="Job title / role")
    start_date: Optional[str] = Field(None, description="Start date (month/year or year)")
    end_date: Optional[str] = Field(None, description="End date or 'Present'")
    responsibilities: Optional[List[str]] = Field(
        default_factory=list,
        description="Key responsibilities and achievements"
    )


class CVEntity(BaseModel):
    """Strict schema for a parsed CV / resume. All fields are Optional for incomplete resumes."""

    full_name: Optional[str] = Field(None, description="Candidate's full name")
    email: Optional[str] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    location: Optional[str] = Field(None, description="City, country, or address")
    linkedin_url: Optional[str] = Field(None, description="LinkedIn profile URL")
    github_url: Optional[str] = Field(None, description="GitHub profile URL")
    summary: Optional[str] = Field(None, description="Professional summary or objective statement")
    education: Optional[List[EducationEntry]] = Field(
        default_factory=list, description="Educational background"
    )
    work_experience: Optional[List[WorkExperienceEntry]] = Field(
        default_factory=list, description="Work history"
    )
    technical_skills: Optional[List[str]] = Field(
        default_factory=list,
        description="Technical / hard skills (languages, tools, frameworks, financial software)"
    )
    soft_skills: Optional[List[str]] = Field(
        default_factory=list,
        description="Soft skills (communication, leadership, etc.)"
    )
    certifications: Optional[List[str]] = Field(
        default_factory=list, description="Certifications and licences"
    )
    languages: Optional[List[str]] = Field(
        default_factory=list, description="Spoken/written languages"
    )
    projects: Optional[List[str]] = Field(
        default_factory=list, description="Notable personal or academic projects"
    )
    domain_category: Optional[str] = Field(
        None,
        description="Inferred domain"
    )


# ─────────────────────────────────────────────
# PROMPT TEMPLATE
# ─────────────────────────────────────────────

_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert HR data extraction assistant. Parse the given resume and
        extract all information into a strict JSON format.

        RULES:
        - Return ONLY valid JSON. No markdown fences, no commentary.
        - Use null for any field not present in the CV.
        - For lists, return an empty array [] if nothing is found.
        - domain_category: infer "BANKING" or "INFORMATION-TECHNOLOGY" from experience/skills.
        - technical_skills: programming languages, frameworks, databases, cloud tools, financial software.
        - soft_skills: communication, teamwork, leadership, analytical thinking, etc.
        - Be thorough — do not omit any information that is present.

        JSON schema to conform to:
        {schema}
        """,
    ),
    (
        "human",
        "Extract all CV information from the following resume text:\n\n{resume_text}",
    ),
])

# Max characters of resume text sent to the LLM (to stay within token limits)
_MAX_TEXT_CHARS = 12_000


# ─────────────────────────────────────────────
# CLASS: CVEntityExtractor
# ─────────────────────────────────────────────

class CVEntityExtractor:
    """
    Extracts structured CV entities from raw resume text using LangChain + LLAMA.

    The extraction chain:
        Prompt → ChatGroq → JsonOutputParser(CVEntity) → CVEntity

    The JSON schema is injected into the system prompt so the model
    always targets the correct structure.
    """

    def __init__(self, settings: Settings):
        self._llm_cfg = settings.llm
        self._api_cfg = settings.api
        self._schema_str = json.dumps(CVEntity.model_json_schema(), indent=2)
        self._chain = self._build_chain()

    # ── Public ─────────────────────────────────────────────────────────────

    def extract(self, resume_text: str) -> CVEntity:
        """
        Parse raw resume text into a validated CVEntity.

        Args:
            resume_text: Plain text content of a resume/CV.

        Returns:
            CVEntity: Fully validated Pydantic model.

        Raises:
            ValueError: If the LLM response cannot be parsed as JSON.
        """
        try:
            raw_output: dict = self._chain.invoke({
                "resume_text": resume_text[:_MAX_TEXT_CHARS],
                "schema": self._schema_str,
            })
            entity = CVEntity(**raw_output)
            logger.info("✓ Extracted entity for: %s", entity.full_name or "Unknown")
            return entity

        except OutputParserException as exc:
            logger.error("JSON parse error during extraction: %s", exc)
            raise ValueError(f"LLM output could not be parsed: {exc}") from exc

    def extract_as_dict(self, resume_text: str) -> dict:
        """Convenience wrapper returning a plain dict (JSON-serialisable)."""
        return self.extract(resume_text).model_dump()

    # ── Private ────────────────────────────────────────────────────────────

    def _build_chain(self):
        """Construct the LCEL chain: Prompt → LLM → JsonOutputParser."""
        llm = ChatGroq(
            model=self._llm_cfg.model,
            api_key=self._api_cfg.groq_api_key,
            temperature=self._llm_cfg.extraction_temperature,
            max_tokens=self._llm_cfg.max_tokens,
        )
        parser = JsonOutputParser(pydantic_object=CVEntity)
        return _EXTRACTION_PROMPT | llm | parser


# ─────────────────────────────────────────────
# CLI DEMO
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from config import Settings

    _SAMPLE = """
    John Smith | john.smith@example.com | +1-555-0101
    San Francisco, CA | linkedin.com/in/johnsmith

    SUMMARY
    Results-driven software engineer with 5 years building scalable web applications.

    EDUCATION
    B.Sc. Computer Science, UC Berkeley, 2017, GPA 3.8

    TECHNICAL SKILLS: Python, JavaScript, React, Node.js, PostgreSQL, AWS, Docker

    SOFT SKILLS: Communication, Team leadership, Agile/Scrum

    EXPERIENCE
    Senior Software Engineer | TechCorp Inc. | Jan 2020 - Present
    - Led team of 5 engineers; built microservices platform for 2M daily users
    - Reduced API latency by 40% via caching optimisation

    CERTIFICATIONS: AWS Certified Solutions Architect - Associate (2021)
    """

    cfg = Settings().validate()
    extractor = CVEntityExtractor(cfg)
    result = extractor.extract(_SAMPLE)
    print(result.model_dump_json(indent=2))