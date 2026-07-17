from collections import Counter
from typing import Final, Literal

import questionary
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.agent.tools import validate_tags
from src.models.agent import (
    ReviewDecisionSchema,
    ScoreResultSchema,
    TranslationUnitSchema,
)

_SCORE_GOOD: Final = 95
_SCORE_WARN: Final = 80
_CONSOLE: Final = Console()


def _score_text(score_result: ScoreResultSchema | None) -> Text:
    if score_result is None:
        return Text("No score", style="dim")
    if score_result.score >= _SCORE_GOOD:
        style = "bold green"
    elif score_result.score >= _SCORE_WARN:
        style = "yellow"
    else:
        style = "bold red"
    return Text(f"{score_result.score}/100", style=style)


def _format_tag_problems(missing: dict[str, int], extra: dict[str, int]) -> str:
    parts: list[str] = []
    if missing:
        parts.append("Missing " + "、".join(f"{t}x{n}" for t, n in missing.items()))
    if extra:
        parts.append("Extra " + "、".join(f"{t}x{n}" for t, n in extra.items()))
    return "; ".join(parts)


def _render_overview(scores: list[TranslationUnitSchema]) -> Table:
    table = Table(
        title=f"Pending manual review ({len(scores)})",
        title_justify="left",
        box=box.SIMPLE_HEAVY,
    )
    table.add_column("#", justify="right", style="dim")
    table.add_column("Score", justify="right")
    table.add_column("Tags", justify="center")
    table.add_column("Source", max_width=36, no_wrap=True, overflow="ellipsis")
    table.add_column("Translation", max_width=36, no_wrap=True, overflow="ellipsis")
    for i, unit in enumerate(scores, 1):
        tag_mark = (
            Text("✓", style="green") if unit.tag_valid else Text("✗", style="red")
        )
        table.add_row(
            str(i),
            _score_text(unit.score_result),
            tag_mark,
            Text(unit.source),
            Text(unit.translated),
        )
    return table


def _render_unit_panel(unit: TranslationUnitSchema, index: int, total: int) -> Panel:
    body = Table.grid(padding=(0, 3, 1, 0))
    body.add_column(style="bold cyan", no_wrap=True)
    body.add_column(overflow="fold")

    body.add_row("Source", Text(unit.source))
    body.add_row(
        "Translation",
        Text(unit.translated) if unit.translated else Text("(empty)", style="dim"),
    )

    suggestion = unit.suggested_translation
    if suggestion and suggestion != unit.translated:
        body.add_row("Suggested Translation", Text(suggestion))

    if not unit.tag_valid:
        _, missing, extra = validate_tags(unit.source, unit.translated)
        body.add_row(
            "Tags", Text(_format_tag_problems(missing, extra), style="bold red")
        )

    if unit.score_result and unit.score_result.deductions:
        lines = [
            Text.assemble(
                (f"-{d.pts}", "bold red"), " ", (d.dim, "cyan"), "  ", d.reason
            )
            for d in unit.score_result.deductions
        ]
        body.add_row("Deductions", Text("\n").join(lines))

    for label, matches in [
        ("Base Glossary", unit.glossary_base),
        ("Mods Glossary", unit.glossary_mods),
    ]:
        if matches:
            pairs = ";".join(f"{m.source} → {m.target}" for m in matches)
            body.add_row(label, Text(pairs))

    if unit.patterns:
        pats = "; ".join(
            f"{p.src_pattern} → {p.tgt_pattern} (x{p.approved_count})"
            for p in unit.patterns
        )
        body.add_row("Patterns", Text(pats, style="dim"))

    if unit.score_result and unit.score_result.notes:
        body.add_row("Notes", Text(unit.score_result.notes, style="dim"))

    title = Text.assemble(
        f"[{index}/{total}] #{unit.id} · {unit.key} · {unit.category} · ",
        _score_text(unit.score_result),
    )
    return Panel(
        body, title=title, title_align="left", border_style="blue", padding=(1, 1, 0, 1)
    )


def _render_context(unit: TranslationUnitSchema) -> Panel:
    if not unit.context:
        return Panel(
            Text("No context found", style="dim"),
            title="Context",
            title_align="left",
            border_style="magenta",
        )
    table = Table(box=box.SIMPLE)
    table.add_column("Component", style="magenta", no_wrap=True)
    table.add_column("Key", style="dim", overflow="fold")
    table.add_column("Source → Translation", overflow="fold")
    for comp in unit.context:
        table.add_row(
            comp.slug, comp.key, Text(f"{comp.unit.source} → {comp.unit.target}")
        )
        for nb in comp.nearby:
            table.add_row(
                "",
                Text(f"↳ {nb.context}", style="dim"),
                Text(f"{nb.source} → {nb.target}", style="dim"),
            )
    return Panel(table, title="Context", title_align="left", border_style="magenta")


def _prompt_translation_edit(source: str, initial: str) -> str | None:
    draft = initial
    while True:
        edited = questionary.text("译文：", default=draft).ask()
        if edited is None or not edited.strip():
            return None
        passed, missing, extra = validate_tags(source, edited)
        if passed:
            return edited
        _CONSOLE.print(
            Text.assemble(
                ("Tags mismatch: ", "bold red"),
                _format_tag_problems(missing, extra),
                "，Please correct and resubmit",
            )
        )
        draft = edited


def _prompt_unit_decision(
    unit: TranslationUnitSchema, index: int, total: int
) -> ReviewDecisionSchema | Literal["approve_rest", "skip_rest"]:
    _CONSOLE.print(_render_unit_panel(unit, index, total))
    suggestion = unit.suggested_translation

    choices = [
        questionary.Choice(
            f"Accept current translation: {unit.translated}",
            value="approve",
            shortcut_key="a",
        )
    ]
    if suggestion and suggestion != unit.translated:
        choices.append(
            questionary.Choice(
                f"Use suggested translation: {suggestion}",
                value="suggest",
                shortcut_key="u",
            )
        )
    choices.append(
        questionary.Choice("Edit translation", value="edit", shortcut_key="e")
    )
    choices.append(questionary.Choice("Skip this unit", value="skip", shortcut_key="s"))

    if unit.context:
        choices.append(
            questionary.Choice("View context", value="context", shortcut_key="c")
        )
    choices.append(
        questionary.Choice("Approve All", value="approve_rest", shortcut_key="y")
    )
    choices.append(questionary.Choice("Skip All", value="skip_rest", shortcut_key="x"))

    while True:
        answer = questionary.select(
            f"[{index}/{total}] Choose action", choices=choices, use_shortcuts=True
        ).ask()

        if answer is None:
            _CONSOLE.print("[yellow]已取消，本条起全部跳过[/]")
            return "skip_rest"

        if answer in ("approve_rest", "skip_rest"):
            return answer

        if answer == "context":
            _CONSOLE.print(_render_context(unit))
            continue

        if answer == "approve":
            return ReviewDecisionSchema(unit_id=unit.id, action="approve")
        if answer == "skip":
            return ReviewDecisionSchema(unit_id=unit.id, action="skip")
        if answer == "suggest":
            return ReviewDecisionSchema(
                unit_id=unit.id, action="modify", translation=suggestion
            )
        edited = _prompt_translation_edit(unit.source, suggestion or unit.translated)
        if edited is not None:
            return ReviewDecisionSchema(
                unit_id=unit.id, action="modify", translation=edited
            )


def prompt_user_review(
    scores: list[TranslationUnitSchema],
    auto_accept: bool = False,
    accept_threshold: int = _SCORE_GOOD,
) -> list[ReviewDecisionSchema]:
    if not scores:
        return []

    auto_skip_count = 0
    manual_scores: list[TranslationUnitSchema] = []
    decisions: list[ReviewDecisionSchema] = []

    for u in scores:
        if u.tag_valid and u.score_result and u.score_result.score >= accept_threshold:
            decisions.append(ReviewDecisionSchema(unit_id=u.id, action="approve"))
        elif auto_accept:
            auto_skip_count += 1
            decisions.append(ReviewDecisionSchema(unit_id=u.id, action="skip"))
        else:
            manual_scores.append(u)

    if not manual_scores:
        _CONSOLE.print(
            f"[bold]Review completed:[/] approved {len(decisions)} units"
            f", skipped {auto_skip_count} units with invalid tags or score below {accept_threshold}"
        )
        return decisions

    _CONSOLE.print(_render_overview(manual_scores))

    decisions: list[ReviewDecisionSchema] = []
    batch_action: Literal["approve", "skip"] | None = None

    for index, unit in enumerate(manual_scores, 1):
        if batch_action is None:
            result = _prompt_unit_decision(unit, index, len(manual_scores))
            if isinstance(result, ReviewDecisionSchema):
                decisions.append(result)
                continue
            batch_action = "approve" if result == "approve_rest" else "skip"
        decisions.append(ReviewDecisionSchema(unit_id=unit.id, action=batch_action))

    counts = Counter(d.action for d in decisions)
    _CONSOLE.print(
        f"[bold]Review completed:[/] approved {counts['approve']} units"
        f", modified {counts['modify']} units"
        f", skipped {counts['skip']} units"
    )
    return decisions
