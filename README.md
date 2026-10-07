# x2loc

XCOM 2 War of the Chosen localization toolkit. It parses UE3 `.int`/`.chn` files, aligns source and target languages, and translates mods through Weblate with an LLM agent, either interactively from the CLI or as an HTTP service that takes a Steam Workshop URL and returns a translated overlay package.

## Requirements

- Python 3.13 and [uv](https://docs.astral.sh/uv/)
- A Weblate project holding three glossary components (base game, mods, custom)
- An OpenAI-compatible LLM endpoint (OpenRouter by default)
- SteamCMD and a Steam account, for the service's Workshop downloads only

## Setup

```bash
uv sync
cp configs/weblate.toml configs/weblate.local.toml   # gitignored; fill in credentials
```

`configs/weblate.local.toml` is the single configuration file for the CLI and the service:

| Table | Purpose |
|---|---|
| top level | `service_token` (API bearer token), `data_root`, `bind_port` |
| `[weblate]` | API URL, token, project slug, source language |
| `[agent]` | LLM API key, base URL, model names (`retry_translation_model_name` translates quality-gate retries; empty reuses the translation model), batch size, auto-approve threshold, `llm_timeout_seconds` per request (default 45; raise it for flex-tier endpoints, which queue for minutes) |
| `[steam]` | SteamCMD executable, download root, Steam credentials |
| `[glossary]` | slugs of the base, mods and custom glossary components |

Every key can be overridden by an `X2LOC_`-prefixed environment variable, nested with `__` (for example `X2LOC_WEBLATE__TOKEN`).

## Offline CLI

```bash
uv run x2loc parse XComGame.int -f json -o out/XComGame.json
uv run x2loc align XComGame.int XComGame.chn -f csv -o out/corpus.csv
uv run x2loc extract out/corpora/ -f csv -o out/glossary.csv
```

`parse` reads one file, `align` pairs a source file with its translation into a bilingual corpus, and `extract` mines glossary terms from directories of corpus JSON files (earlier directories take priority).

## Interactive translation agent

```bash
uv run x2loc agent run --config configs/weblate.local.toml --batch-size 20
```

The agent pulls untranslated units of the mods glossary from Weblate, translates, validates tags, scores each candidate, and asks for a human decision on every batch before uploading approved translations.

## HTTP service

```bash
uv run x2loc-api
```

All routes require `Authorization: Bearer <service_token>`.

| Route | Purpose |
|---|---|
| `POST /v1/jobs` | Submit `{"workshop_url": "https://steamcommunity.com/sharedfiles/filedetails/?id=..."}`; LLM fields are optional overrides of `[agent]` |
| `GET /v1/jobs/{id}` | Job status and stage |
| `GET /v1/jobs/{id}/events` | Server-sent progress events |
| `POST /v1/jobs/{id}/cancel` | Cancel a running job |
| `GET /v1/jobs/{id}/artifact` | Download the translated overlay zip (`X-Artifact-SHA256` header) |

A job downloads the mod with SteamCMD, syncs its localization into Weblate components, translates missing units with automatic threshold review, writes Chinese overlay files, adds newly mined terms to the custom glossary, and packages the result. Units that still fail the quality gate after the last attempt stay empty in Weblate and keep their source text in the overlay (`units_untranslated` in the job progress); held translations whose tags disagree with the source, such as a mod's own broken `.chn`, are cleared and retranslated. Jobs and artifacts live in memory and under `data_root`, which is reset on every start.

### Batch translation of a collection

```bash
docker compose exec x2loc x2loc batch \
  "https://steamcommunity.com/sharedfiles/filedetails/?id=<collection id>" \
  --max-size-mb 5 --limit 30
```

`x2loc batch` runs the job pipeline in-process for every item of a public Workshop collection, one at a time, and writes each overlay zip plus `summary.json` to `output/batch/<UTC timestamp>/`. It differs from the service in two ways:

- Glossaries are kept on disk (`glossary_cache_dir`, the `glossary-cache` volume in Docker): the first run reads them in full, later runs pull only units changed since the last sync, and a full re-read happens once a day.
- New custom-glossary terms are queued locally and published once the run ends; later jobs of the same run already use them. If a run stops early, the queue stays on disk and the next run publishes it.

Run it inside the container, where SteamCMD and its login cache live, and do not submit service jobs meanwhile: both would drive the same SteamCMD install.

### Translating the mods installed on this machine

```bash
uv run x2loc local                 # every mod in [local] workshop_dir
uv run x2loc local 1122974240 667104300
```

`x2loc local` reads mods straight from the local Workshop content directory and runs the same pipeline as `batch`, so it needs Weblate and the LLM endpoint but neither SteamCMD, a Steam account nor `service_token`. Set the two paths once in the `[local]` table (or pass `--workshop-dir` and `--mods-dir`). Mods without `.int` files under `Localization` are skipped.

Each translated mod is installed as a standalone mod `XComGame/Mods/x2loc_zh_<workshop id>/`, holding the generated `.chn` files and its own `.XComMod`, so a Steam update of the original mod cannot overwrite the translation; re-running replaces it. Enable the overlay mods in the mod launcher. Overlay zips and `summary.json` also go to `output/local/<UTC timestamp>/`. Run it on the host that has the game installed, not in the container.

### Docker

```bash
docker compose up -d --build
```

The container mounts `configs/weblate.local.toml` read-only, listens on `127.0.0.1:8180`, and keeps SteamCMD, its login cache and the Workshop download cache in named volumes.

## Development

```bash
uv run pytest
uv run ruff check --fix . && uv run ruff format .
uv run ty check src tests
```

## Layout

```
src/
├── config.py      # layered service/CLI configuration (env > TOML > defaults)
├── core/          # UE3 parsing, alignment, term extraction, overlay writing, packaging
├── models/        # Pydantic schemas
├── export/        # CSV/JSON serializers for the offline CLI
├── services/      # Weblate and Steam clients, custom-glossary writer
├── agent/         # LangGraph translation agent and its nodes
├── jobs/          # service job manager and Workshop pipeline
├── api/           # FastAPI app
├── cli/           # `x2loc` Typer entry point
└── ui/            # interactive review prompts
```

## License

MIT
