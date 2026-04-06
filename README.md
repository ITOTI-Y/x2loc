# x2loc

A parser for XCOM 2 (Unreal Engine 3) localization files. Parses `.int`/`.chn`/`.cht` and other UE3 localization files into structured Pydantic data models.

## Features

- Parse UE3 localization section headers (4 formats: CLASS_ONLY / OBJECT_CLASS / PACKAGE_CLASS / ARCHETYPE_CLASS)
- Extract entry key-value pairs, array indices, and `+Key=(...)` append struct syntax
- Detect 9 placeholder types: XGParam, Ability, Bullet, Heal, BR, HTML font, Percent, Percent Wrapped, Newline
- Auto encoding detection: UTF-16-LE/BE, UTF-8-sig, UTF-8, latin1 fallback
- Language inference from file extension (`.int` → en, `.chn` → zh_Hans, 11 languages supported)

## Project Structure

```
src/
├── _share.py              # Language-extension mappings
├── core/
│   ├── _share.py          # Regex patterns, ParseError
│   └── parser.py          # LocFileParser
└── models/
    ├── _share.py          # BaseSchema, enums
    ├── entry.py           # PlaceholderSchema, StructFieldSchema, EntrySchema
    ├── section.py         # SectionHeader, SectionSchema
    └── file.py            # LocalizationFile
```

## Quick Start

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest tests/ -v

# Lint + Format
uv run ruff check --fix . && uv run ruff format .
```

## Usage

```python
from pathlib import Path
from src.core.parser import LocFileParser

parser = LocFileParser()
result = parser.parse(Path("XComGame.int"))

print(result.lang)          # "en"
print(result.encoding)      # "utf-16-le"
print(result.entry_count)   # 20131

for section in result.sections:
    print(f"[{section.header.raw}] ({len(section.entries)} entries)")
    for entry in section.entries:
        if entry.placeholders:
            types = [p.type.value for p in entry.placeholders]
            print(f"  {entry.key} -> placeholders: {types}")
```

## Supported Language Extensions

| Extension | Language Code | Language |
|-----------|--------------|----------|
| `.int` | en | English |
| `.chn` | zh_Hans | Simplified Chinese |
| `.cht` | zh_Hant | Traditional Chinese |
| `.jpn` | ja | Japanese |
| `.kor` | ko | Korean |
| `.deu` | de | German |
| `.fra` | fr | French |
| `.esn` | es | Spanish |
| `.ita` | it | Italian |
| `.pol` | pl | Polish |
| `.rus` | ru | Russian |

## Dependencies

- Python >= 3.12
- pydantic >= 2.12
- loguru >= 0.7
