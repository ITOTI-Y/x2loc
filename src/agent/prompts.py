from __future__ import annotations

from src.agent.state import ContextResult, GlossaryMatch, SessionPattern

TRANSLATION_SYSTEM = """\
You are a professional game localization translator for XCOM 2: War of the Chosen.
Translate English game terms to target language: {target_lang}.

Rules:
- Preserve ALL tags exactly: HTML (<font>, </font>, <br/>), XML (<XGParam:.../>), \
printf (%d, %s), placeholders ({{0}}, {{Name}}), escapes (\\n, \\t)
- Base glossary translations are authoritative — always prefer established terms
- Fix source typos: translate the correct meaning (e.g., "Reistance" → 抗性)
- For <font color='...'>text</font>: translate only inner text, keep tag structure
- Output ONLY the target language translation, no explanations or quotes"""

SCORING_SYSTEM = """\
You are a quality assessor for XCOM 2 game term translations (EN → {target_lang}).
Score translations on a 0-100 scale using the rubric below.
Return ONLY valid JSON, no other text."""

TAG_FIX_TEMPLATE = """\
Your previous translation has tag errors. Fix them.

Source: {source}
Your translation: {translation}

Missing tags (must be added): {missing}
Extra tags (must be removed): {extra}

All tags from the source must appear in the translation exactly as-is.
Output ONLY the corrected translation, nothing else."""


def format_translation_prompt(
    source: str,
    category: str | None,
    base_matches: list[GlossaryMatch],
    mods_matches: list[GlossaryMatch],
    context_result: ContextResult,
    patterns: list[SessionPattern],
) -> str:
    parts = [f"Source: {source}", f"Category: {category or 'unknown'}", ""]

    if base_matches:
        parts.append("Base Glossary Reference:")
        for m in base_matches[:10]:
            parts.append(f"  {m['source']} → {m['target']}")
        parts.append("")

    if mods_matches:
        parts.append("Mods Glossary Reference:")
        for m in mods_matches[:10]:
            parts.append(f"  {m['source']} → {m['target']}")
        parts.append("")

    nearby = context_result.get("nearby", [])
    mod_comp = context_result.get("mod_component")
    if nearby and mod_comp:
        pct = context_result.get("translated_percent", "?")
        parts.append(f"Nearby Strings from {mod_comp} ({pct}% translated):")
        for n in nearby:
            tgt = n["tgt"] or "(untranslated)"
            parts.append(f"  {n['src']} → {tgt}")
        parts.append("")

    if patterns:
        parts.append("Established Session Patterns:")
        for p in patterns:
            parts.append(
                f"  {p['src_pattern']} → {p['tgt_pattern']} ({p['approved_count']} approved)"
            )
        parts.append("")

    parts.append("Translation:")
    return "\n".join(parts)


def format_scoring_prompt(
    source: str,
    translation: str,
    category: str | None,
    base_matches: list[GlossaryMatch],
    mods_matches: list[GlossaryMatch],
    context_result: ContextResult,
    patterns: list[SessionPattern],
) -> str:
    parts = [
        f"Source: {source}",
        f"Translation: {translation}",
        f"Category: {category or 'unknown'}",
        "",
    ]

    if base_matches:
        parts.append("Base Glossary Matches:")
        for m in base_matches[:10]:
            parts.append(f"  {m['source']} → {m['target']}")
        parts.append("")

    nearby = context_result.get("nearby", [])
    mod_comp = context_result.get("mod_component")
    if nearby and mod_comp:
        parts.append(f"Nearby Strings ({mod_comp}):")
        for n in nearby:
            tgt = n["tgt"] or "(untranslated)"
            parts.append(f"  {n['src']} → {tgt}")
        parts.append("")

    if mods_matches:
        parts.append("Similar Translated Terms (glossary-mods):")
        for m in mods_matches[:10]:
            parts.append(f"  {m['source']} → {m['target']}")
        parts.append("")

    if patterns:
        parts.append("Established Session Patterns:")
        for p in patterns:
            parts.append(
                f"  {p['src_pattern']} → {p['tgt_pattern']} ({p['approved_count']} approved)"
            )
        parts.append("")

    parts.append("""\
Scoring rubric — start at 100, deduct for issues:

1. Glossary Consistency (max -30)
   Must use established base glossary terms.

2. Semantic Accuracy (max -30)
   Faithful to source meaning in XCOM 2 context.

3. Style Consistency (max -20)
   Matches naming patterns of same-category terms.

4. Context Fit (max -20)
   Makes sense alongside nearby strings.

Return JSON:
{"score": <int>, "deductions": [{"dim": "<name>", "pts": <neg_int>, "reason": "<brief>"}], \
"suggested_alternative": "<str_or_null>", "notes": "<optional>"}
REQUIRED: if score < 95, suggested_alternative must be non-null.""")
    return "\n".join(parts)
