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
| `[agent]` | LLM API key, base URL, model names, batch size, auto-approve threshold |
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

A job downloads the mod with SteamCMD, syncs its localization into Weblate components, translates missing units with automatic threshold review, writes Chinese overlay files, adds newly mined terms to the custom glossary, and packages the result. Jobs and artifacts live in memory and under `data_root`, which is reset on every start.

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
