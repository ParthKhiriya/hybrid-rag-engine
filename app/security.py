"""Content-safety layer for a RAG pipeline.

The main injection surface in a RAG system is not the user's question — it's
the *retrieved content*, which an attacker can plant in a document that later
gets ingested and served back to the LLM as "trusted" context. This module
provides:

1. Heuristic scanning for injected instructions in ingested/retrieved text.
2. A trust-scoring model so untrusted content is labeled, down-ranked, and
   explicitly fenced off from the instruction channel of the prompt.
3. A lightweight output check to catch obvious signs that an injection
   succeeded (e.g. the model echoing a fake "system" role or leaking the
   prompt scaffold back to the caller).

None of this is a substitute for a dedicated classifier in a real deployment
— see docs/production-hardening.md for what a hardened version would add
(trained classifiers, allowlisted output schemas, sandboxed tool execution).
"""

import re
from dataclasses import dataclass, field

# Heuristics for content trying to hijack the model's instruction-following.
# Deliberately broad / high-recall: false positives just lower trust score
# and get surfaced to a human, they don't hard-block ingestion by default.
INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore (all |any )?(the )?(previous|prior|above) instructions", re.I),
    re.compile(r"disregard (the )?(system|previous|above) (prompt|instructions)", re.I),
    re.compile(r"\byou are now\b", re.I),
    re.compile(r"reveal (your|the) (system prompt|instructions|prompt)", re.I),
    re.compile(r"\bact as (if|though)\b", re.I),
    re.compile(r"new instructions\s*:", re.I),
    re.compile(r"^\s*(system|assistant)\s*:", re.I | re.M),
    re.compile(r"</?\s*(system|assistant|user)\s*>", re.I),
    re.compile(r"do not (tell|inform|warn|notify) the user", re.I),
    re.compile(r"\bexfiltrate\b|\bsend (this|these|the) (data|credentials|secrets) to\b", re.I),
    re.compile(r"[\u200b-\u200f\u202a-\u202e\ufeff]"),  # zero-width / bidi control chars
]

MAX_RISK_SCORE = 1.0
RISK_PER_MATCH = 0.3


@dataclass(slots=True)
class ScanResult:
    flagged: bool
    matched_patterns: list[str] = field(default_factory=list)
    risk_score: float = 0.0


def scan_text(text: str) -> ScanResult:
    """Heuristic scan for prompt-injection attempts in a block of text."""
    matches = [p.pattern for p in INJECTION_PATTERNS if p.search(text)]
    risk = min(MAX_RISK_SCORE, len(matches) * RISK_PER_MATCH)
    return ScanResult(flagged=bool(matches), matched_patterns=matches, risk_score=risk)


# --- Trust scoring -----------------------------------------------------

TrustLevel = str  # "trusted" | "unverified" | "quarantined"

TRUST_WEIGHT = {
    "trusted": 1.0,
    "unverified": 0.6,
    "quarantined": 0.0,  # excluded from context entirely by default
}


def classify_trust(declared_trust: str, scan: ScanResult) -> TrustLevel:
    """Combine the caller-declared trust label with scan results.

    Callers cannot self-declare their way past a positive injection scan —
    a flagged document is always at most "quarantined", regardless of what
    trust level was requested at ingest time.
    """
    if scan.flagged and scan.risk_score >= 0.6:
        return "quarantined"
    if declared_trust == "trusted" and not scan.flagged:
        return "trusted"
    return "unverified"


# --- Output validation ---------------------------------------------------


def validate_output(answer: str) -> ScanResult:
    """Check a generated answer for signs that an injection succeeded —
    e.g. the model adopting a fake role, or echoing prompt-scaffold markup
    that only appears if injected content leaked into the instruction
    channel."""
    return scan_text(answer)
