# x2loc

English | [简体中文](README.zh-CN.md)

XCOM 2 War of the Chosen localization toolkit. It parses UE3 `.int`/`.chn` files, aligns source and target languages, and translates mods through Weblate with an LLM agent, either interactively from the CLI or as an HTTP service that takes a Steam Workshop URL and returns a translated overlay package.

## Download

Players can download `x2loc.exe` from [Releases](https://github.com/ITOTI-Y/x2loc/releases) and run it: it opens a page in the browser that scans your subscribed mods, translates the missing text with your own OpenAI-compatible model and installs each translation as a separate mod. Without a Weblate token it only reads the shared translations and keeps its own results on your computer. See [Local app](#local-app) for details.

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
| `[agent]` | LLM API key, base URL, model names (`retry_translation_model_name` translates quality-gate retries; empty reuses the translation model), batch size, `units_per_request` (units packed into one LLM request), auto-approve threshold, `llm_timeout_seconds` per request (default 45; raise it for flex-tier endpoints, which queue for minutes) |
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

A job downloads the mod with SteamCMD, syncs its localization into Weblate components, translates missing units with automatic threshold review, writes Chinese overlay files, adds newly mined terms to the custom glossary, and packages the result. A mod's own `.chn` translations are not imported directly: each is scored as its unit's first candidate and reaches Weblate only through the quality gate, and a rejected one is shown to the translator as a reference for the retry. Units that still fail the gate after the last attempt are written as "needs editing" with the best candidate and the scorer's reasons as a Weblate comment; a unit with no candidate whose tags match the source stays empty and keeps its source text in the overlay (`units_untranslated` in the job progress). Held translations whose tags disagree with the source are cleared and retranslated. Jobs and artifacts live in memory and under `data_root`, which is reset on every start.

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

#### Without a Weblate token

Weblate is only read when `[weblate] token` is empty: translations Weblate already holds are used as they are, the rest are translated with your own LLM endpoint, and the results are kept in a local SQLite file (`store_path`, default `data/store.sqlite`) instead of being written back. `--store local` does the same with a token set; `--store weblate` forces writing to Weblate. Push what was kept locally later with a token that may write:

```bash
uv run x2loc local 1122974240 --store local
uv run x2loc upload                # every queued translation
uv run x2loc upload <component slug>
```

A unit someone changed on Weblate after it was translated here is not overwritten: `upload` lists it as a conflict.

When Weblate, this machine and the mod itself each hold a translation of a unit, the overlay takes the first available one in `priority` order: `manual` (edited here by hand), `upstream` (translated on Weblate), `machine` (translated here), `needs_editing`, `shipped` (the mod's own); the source text is the last resort.

Each translated mod is installed as a standalone mod `XComGame/Mods/x2loc_<workshop id>_zh/`, holding the generated `.chn` files and its own `.XComMod`, so a Steam update of the original mod cannot overwrite the translation; re-running replaces it. Enable the overlay mods in the mod launcher. Overlay zips and `summary.json` also go to `output/local/<UTC timestamp>/`. Run it on the host that has the game installed, not in the container.

### Docker

```bash
docker compose up -d --build
```

The container mounts `configs/weblate.local.toml` read-only, listens on `127.0.0.1:8180`, and keeps SteamCMD, its login cache and the Workshop download cache in named volumes.

## Local app

`x2loc-app` serves the same pipeline to a browser page on this machine: settings (model, Weblate, game paths, translation priority), the subscribed mods with their Weblate progress, runs, installed translations, review and the upload queue. It binds to `127.0.0.1` on a free port and opens the browser with a per-launch session token; every API call must carry it.

```bash
cd webui && pnpm install && pnpm build && cd ..   # builds into src/webui/dist
uv run x2loc-app
```

Settings live in `%LOCALAPPDATA%\x2loc\settings.json` (`X2LOC_HOME` overrides the folder), together with the local translation store; the LLM API key and the Weblate token are kept in the Windows Credential Manager. For UI work, run `uv run x2loc-app --port 8765 --token dev --no-browser` and `pnpm dev` in `webui/`, then open `http://localhost:5173/#token=dev`.

Releases ship the app as one Windows executable, `x2loc.exe`, which needs neither Python nor Node.js: run it, keep its console window open while using the page it opens, and close the window to stop it. Pushing a `v*` tag runs `.github/workflows/release.yml`, which builds the pages, runs the tests, compiles the executable with Nuitka (`packaging/build.py`) and attaches it to a GitHub release. To build it locally:

```bash
cd webui && pnpm build && cd ..
uv run --group build python packaging/build.py -- --zig   # without Visual Studio; omit --zig with MSVC
```

## Weapon category names mod

`mods/x2locWeaponCatNames` is a small script mod. [WOTCMusashiModFixes](https://steamcommunity.com/sharedfiles/filedetails/?id=2569592723) appends a "valid weapons" line to ability descriptions, where categories added by mods show as internal codes such as `combatknife`; this mod replaces each code with the localized name of the category's own weapons, in whatever language the game runs. It needs War of the Chosen, [X2WOTCCommunityHighlander](https://steamcommunity.com/sharedfiles/filedetails/?id=1134256495) and WOTCMusashiModFixes, and the local app can install it from its Installed page.

```bash
uv run python mods/x2locWeaponCatNames/build.py --sdk <WOTC SDK> --mods-dir <XComGame/Mods>
uv run python mods/x2locWeaponCatNames/publish.py --steamcmd <steamcmd.exe> --username <Steam account>
```

`build.py` compiles it with the War of the Chosen SDK and installs it; `publish.py` uploads it to the Steam Workshop with [SteamCMD](https://developer.valvesoftware.com/wiki/SteamCMD), using `workshop/description.bbcode` and `ModPreview.jpg`. Tags and required items are set on the item's Workshop page.

## Development

```bash
uv run pytest
uv run ruff check --fix . && uv run ruff format .
uv run ty check src tests
cd webui && pnpm lint && pnpm check:i18n && pnpm build
```

## Layout

```
src/
├── config.py      # layered service/CLI configuration (env > TOML > defaults)
├── core/          # UE3 parsing, alignment, term extraction, overlay writing, packaging
├── models/        # Pydantic schemas
├── export/        # CSV/JSON serializers for the offline CLI
├── services/      # Weblate and Steam clients, glossaries, translation stores
├── agent/         # LangGraph translation agent and its nodes
├── jobs/          # service job manager and Workshop pipeline
├── api/           # FastAPI service
├── app/           # local app: settings, secrets, runs and its HTTP interface
├── cli/           # `x2loc` Typer entry point
└── ui/            # interactive review prompts
webui/             # the local app's pages (React, Vite, Tailwind)
packaging/         # Nuitka build of x2loc.exe
mods/x2locWeaponCatNames/   # weapon category names script mod
```

## License

MIT
