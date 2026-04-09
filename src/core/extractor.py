import re
from typing import Final

from src.models.corpus import BilingualCorpus, BilingualEntry
from src.models.glossary import Glossary, GlossaryTerm, TermContext

TEMPLATE_RULES: Final[dict[str, tuple[str, str]]] = {
    "X2AbilityTemplate": ("LocFriendlyName", "ability"),
    "X2WeaponTemplate": ("FriendlyName", "weapon"),
    "X2ArmorTemplate": ("FriendlyName", "armor"),
    "X2AmmoTemplate": ("FriendlyName", "ammo"),
    "X2GrenadeTemplate": ("FriendlyName", "grenade"),
    "X2PairedWeaponTemplate": ("FriendlyName", "paired_weapon"),
    "X2WeaponUpgradeTemplate": ("FriendlyName", "weapon_upgrade"),
    "X2SitRepTemplate": ("FriendlyName", "sitrep"),
    "X2CharacterTemplate": ("strCharacterName", "character"),
    "X2SoldierClassTemplate": ("DisplayName", "soldier_class"),
    "X2MPCharacterTemplate": ("DisplayName", "mp_character"),
    "X2SoldierAbilityUnlockTemplate": ("DisplayName", "ability_unlock"),
    "X2SoldierUnlockTemplate": ("DisplayName", "soldier_unlock"),
    "X2SoldierPersonalityTemplate": ("FriendlyName", "soldier_personality"),
    "X2TraitTemplate": ("TraitFriendlyName", "trait"),
    "X2TechTemplate": ("DisplayName", "tech"),
    "X2ItemTemplate": ("FriendlyName", "item"),
    "X2EquipmentTemplate": ("FriendlyName", "equipment"),
    "X2QuestItemTemplate": ("FriendlyName", "quest_item"),
    "X2DarkEventTemplate": ("DisplayName", "dark_event"),
    "X2StrategyCardTemplate": ("DisplayName", "strategy_card"),
    "X2ResistanceActivityTemplate": ("DisplayName", "resistance_activity"),
    "X2ObjectiveTemplate": ("Title", "objective"),
    "X2MissionTemplate": ("DisplayName", "mission"),
    "X2RewardTemplate": ("DisplayName", "reward"),
    "X2SabotageTemplate": ("DisplayName", "sabotage"),
    "X2GameplayMutatorTemplate": ("DisplayName", "mutator"),
    "X2ChosenActionTemplate": ("DisplayName", "chosen_action"),
    "X2AbilityPointTemplate": ("ActionFriendlyName", "ability_point"),
    "X2CityTemplate": ("DisplayName", "city"),
    "X2CountryTemplate": ("DisplayName", "country"),
    "X2WorldRegionTemplate": ("DisplayName", "world_region"),
    "X2ContinentTemplate": ("DisplayName", "continent"),
    "X2FacilityTemplate": ("DisplayName", "facility"),
    "X2FacilityUpgradeTemplate": ("DisplayName", "facility_upgrade"),
    "X2BastionTemplate": ("DisplayName", "bastion"),
    "X2EncyclopediaTemplate": ("ListTitle", "encyclopedia"),
    "X2HackRewardTemplate": ("FriendlyName", "hack_reward"),
    "X2BodyPartTemplate": ("DisplayName", "cosmetic"),
}

KEY_PATTERN_RULES: Final[list[tuple[re.Pattern[str], str]]] = [
    (re.compile(r"^m_strGeneric\w+$"), "ui_generic"),
    (re.compile(r"^m_strDefaultHelp_\w+$"), "ui_nav"),
    (re.compile(r"^CharStatLabels\[.+\]$"), "stat"),
    (re.compile(r"^RankNames\[\d+\]$"), "rank"),
    (re.compile(r"^ShortNames\[\d+\]$"), "rank_short"),
    (re.compile(r"^PsiRankNames\[\d+\]$"), "psi_rank"),
    (re.compile(r"^m_MedalTypes\[\d+\]$"), "medal"),
    (re.compile(r"^m_arrDifficultyTypeStrings\[\d+\]$"), "difficulty"),
]

NAME_KEY_HINTS: Final[set[str]] = {"name", "title", "label", "header", "button"}

_NOISE_RE: Final[re.Pattern[str]] = re.compile(r"[<>#]")
_ARRAY_INDEX_RE: Final[re.Pattern[str]] = re.compile(r"\[\d+\]")


class TermExtractor:
    def extract(
        self,
        corpora: list[BilingualCorpus],
    ) -> Glossary:
        """Extract glossary terms from aligned bilingual corpora.

        Args:
            corpora: Aligned corpora, ordered by priority (index 0 = highest).
                     Entries from earlier corpora sort first in output.

        Returns:
            Glossary with deduplicated terms and placeholder entries.
        """
        if not corpora:
            return Glossary(source_lang="", target_lang="", terms=[])

        source_lang = corpora[0].source_lang
        target_lang = corpora[0].target_lang

        # Key: (source_text, target_text) → (category, priority, contexts, same_as_source)
        merged: dict[tuple[str, str], tuple[str, int, list[TermContext], bool]] = {}

        for corpus in corpora:
            for entry in corpus.entries:
                result = self._classify(entry)
                if result is None:
                    continue

                category, priority = result
                source_text = entry.source.value
                target_text = entry.target.value if entry.target is not None else ""
                same = entry.target is not None and source_text == target_text

                ctx = TermContext(
                    compound_key=entry.compound_key,
                    section_raw=entry.section_header.raw,
                    key=entry.source.key,
                    source_path=corpus.source_path,
                )

                dedup_key = (source_text, target_text)
                if dedup_key in merged:
                    existing_cat, existing_pri, existing_ctxs, existing_same = merged[
                        dedup_key
                    ]
                    if priority < existing_pri:
                        merged[dedup_key] = (
                            category,
                            priority,
                            [*existing_ctxs, ctx],
                            same,
                        )
                    else:
                        merged[dedup_key] = (
                            existing_cat,
                            existing_pri,
                            [*existing_ctxs, ctx],
                            existing_same,
                        )
                else:
                    merged[dedup_key] = (category, priority, [ctx], same)

        placeholder_patterns: set[str] = set()
        for corpus in corpora:
            for entry in corpus.entries:
                for ph in entry.source.placeholders:
                    placeholder_patterns.add(ph.pattern)

        terms: list[GlossaryTerm] = []

        for (source_text, target_text), (
            category,
            _priority,
            contexts,
            same,
        ) in merged.items():
            terms.append(
                GlossaryTerm(
                    source=source_text,
                    target=target_text,
                    category=category,
                    do_not_translate=False,
                    same_as_source=same,
                    contexts=contexts,
                )
            )

        terms.sort(key=lambda t: (t.category, t.source))

        for pattern in sorted(placeholder_patterns):
            terms.append(
                GlossaryTerm(
                    source=pattern,
                    target=pattern,
                    category="placeholder",
                    do_not_translate=True,
                    same_as_source=True,
                    contexts=[],
                )
            )

        return Glossary(
            source_lang=source_lang,
            target_lang=target_lang,
            terms=terms,
        )

    def _classify(self, entry: BilingualEntry) -> tuple[str, int] | None:
        """Classify a BilingualEntry into a term category.

        Returns:
            (category, priority) where priority 0=Rule A, 1=Rule B, 2=Rule C.
            None if entry does not qualify as a term.
        """
        source = entry.source

        if source.is_append and source.struct_fields is not None:
            return None
        if not source.value:
            return None

        # Rule A: ClassName + Key match
        class_name = entry.section_header.class_name
        if class_name and class_name in TEMPLATE_RULES:
            target_key, category = TEMPLATE_RULES[class_name]
            if source.key == target_key and not source.is_append:
                return (category, 0)

        # Rule B: Key pattern match
        for pattern, category in KEY_PATTERN_RULES:
            if pattern.match(source.key):
                return (category, 1)

        # Rule C: Short text fallback
        if source.placeholders:
            return None
        text = source.value
        if text and text[-1] in ".!?":
            return None
        if _NOISE_RE.search(text):
            return None
        if _ARRAY_INDEX_RE.search(source.key):
            return None
        word_count = len(text.split())
        if word_count <= 3:
            return ("short_text", 2)
        if word_count <= 5 and any(
            hint in source.key.lower() for hint in NAME_KEY_HINTS
        ):
            return ("short_text", 2)

        return None
