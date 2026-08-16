"""Security utilities for Layer 1 deterministic validation."""

import re
from typing import Any

def detect_pii(text: str, patterns: dict[str, str]) -> list[dict[str, Any]]:
    """Detect PII in text using regex patterns.
    
    Args:
        text: The text to analyze.
        patterns: A dictionary of PII type to regex string.
        
    Returns:
        A list of matches with 'type' and 'span'.
    """
    matches = []
    for pii_type, pattern_str in patterns.items():
        try:
            pattern = re.compile(pattern_str, re.IGNORECASE)
            for m in pattern.finditer(text):
                matches.append({
                    "type": pii_type,
                    "span": m.span(),
                    "match": m.group(),
                })
        except re.error:
            # Skip invalid patterns
            pass
    return matches

def check_prompt_injection(text: str, prohibited: list[str]) -> bool:
    """Check if the text contains known prompt injection patterns.
    
    Args:
        text: The text to analyze.
        prohibited: A list of prohibited keywords/phrases.
        
    Returns:
        True if any prohibited keyword is found, False otherwise.
    """
    text_lower = text.lower()
    for phrase in prohibited:
        if phrase.lower() in text_lower:
            return True
    return False

def sanitize_output(text: str, pii_matches: list[dict[str, Any]]) -> str:
    """Redact detected PII from text.
    
    Args:
        text: The original text.
        pii_matches: The list of detected PII matches from detect_pii().
        
    Returns:
        The text with PII redacted (replaced with [REDACTED <TYPE>]).
    """
    # Sort matches in reverse order so we don't mess up indices during replacement
    pii_matches = sorted(pii_matches, key=lambda m: m["span"][0], reverse=True)
    
    sanitized = text
    for match in pii_matches:
        start, end = match["span"]
        pii_type = match["type"].upper()
        sanitized = sanitized[:start] + f"[REDACTED {pii_type}]" + sanitized[end:]
        
    return sanitized
