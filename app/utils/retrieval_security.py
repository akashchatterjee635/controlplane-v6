"""Retrieval security — indirect prompt injection defense.

Sanitizes retrieved documents before they enter the generation prompt.
Assigns trust scores based on document source.
"""

import re

from langchain_core.documents import Document

# Patterns that indicate indirect prompt injection in retrieved documents
INDIRECT_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|rules|prompts|context)",
    r"disregard\s+(all\s+)?(previous|prior|above)",
    r"you\s+are\s+now\s+(an?\s+)?(unrestricted|unfiltered|evil)",
    r"new\s+instructions?:?",
    r"system\s*:\s*you\s+are",
    r"<\s*/?\s*system\s*>",
    r"\[\s*INST\s*\]",
    r"act\s+as\s+(root|admin|god|sudo)",
    r"reveal\s+(the|your)\s+(user|system|api|secret)",
    r"output\s+(the|your)\s+(system|initial)\s+prompt",
]

# Compiled patterns for performance
_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in INDIRECT_INJECTION_PATTERNS]

# Trust levels by source
SOURCE_TRUST = {
    "internal_kb": 1.0,
    "knowledge_base": 0.9,
    "web_search": 0.5,
    "unknown": 0.3,
}


def detect_indirect_injection(text: str) -> list[str]:
    """Scan text for indirect prompt injection patterns.

    Returns list of matched pattern descriptions.
    """
    matches = []
    for pattern in _COMPILED_PATTERNS:
        if pattern.search(text):
            matches.append(pattern.pattern)
    return matches


def sanitize_retrieved_doc(doc: Document) -> Document:
    """Sanitize a retrieved document by stripping injection patterns.

    Returns a new Document with dangerous content replaced.
    """
    text = doc.page_content
    sanitized = text
    injection_found = False

    for pattern in _COMPILED_PATTERNS:
        if pattern.search(sanitized):
            injection_found = True
            sanitized = pattern.sub("[REDACTED-INJECTION]", sanitized)

    new_metadata = dict(doc.metadata)
    if injection_found:
        new_metadata["injection_sanitized"] = True
        new_metadata["trust_score"] = max(0.1, get_trust_score(doc) - 0.3)
    else:
        new_metadata["trust_score"] = get_trust_score(doc)

    return Document(page_content=sanitized, metadata=new_metadata)


def get_trust_score(doc: Document) -> float:
    """Assign a trust score based on document source."""
    source = doc.metadata.get("source", "unknown")
    for key, score in SOURCE_TRUST.items():
        if key in source.lower():
            return score
    return SOURCE_TRUST["unknown"]


def sanitize_documents(docs: list[Document]) -> list[Document]:
    """Sanitize a list of retrieved documents.

    Returns sanitized documents with trust scores.
    """
    return [sanitize_retrieved_doc(doc) for doc in docs]
