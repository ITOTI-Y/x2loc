# x2loc

English | [简体中文](README.zh-CN.md)

Most XCOM 2: War of the Chosen mods ship only English text, and community translations, often shared in comment threads, fall behind as the mods update. x2loc translates mods through a shared Weblate project: it parses the UE3 `.int` and `.chn` localization files, aligns source text with existing translations, translates the missing units with a large language model (LLM) behind a scoring quality gate, and installs each result as a standalone overlay mod. The same pipeline runs in three interfaces: a local app for players, a command line interface (CLI), and an HTTP service.

## Download

Players download `x2loc.exe` from [Releases](https://github.com/ITOTI-Y/x2loc/releases) and run it. The program opens a browser page that scans the subscribed mods, translates missing text with the player's own OpenAI compatible model, and installs one translation mod per original mod. Without a Weblate token, it reads the shared translations only and keeps its own results on the local computer. [Local app](#local-app) describes the program in detail.

## Requirements

- Python 3.13 and [uv](https://docs.astral.sh/uv/)
- A Weblate project with three glossary components: base game, mods, and custom
- An OpenAI compatible LLM endpoint, OpenRouter by default
- SteamCMD and a Steam account, needed only when the HTTP service downloads Workshop items

## Setup

```bash
uv sync
cp configs/weblate.toml configs/weblate.local.toml   # gitignored; holds the credentials
```

`configs/weblate.local.toml` is the single configuration file of the CLI and the HTTP service:

| Table | Contents |
|---|---|
| top level | `service_token` (bearer token of the HTTP API), `data_root`, `bind_port` |
| `[weblate]` | API URL, token, project slug, source language |
| `[agent]` | LLM API key, endpoint URL, model names, batch size, `units_per_request`, auto-approve threshold, `llm_timeout_seconds` |
| `[steam]` | SteamCMD executable, download root, Steam credentials |
| `[glossary]` | slugs of the base game, mods, and custom glossary components |

In `[agent]`, `retry_translation_model_name` translates the units that fail the quality gate and falls back to the translation model when empty; `units_per_request` packs several units into one LLM request; `llm_timeout_seconds` limits one request and defaults to 45 s, which flex tier endpoints exceed because they queue requests for minutes.

Every key can be overridden by an environment variable with the `X2LOC_` prefix, with nested keys joined by `__`, for example `X2LOC_WEBLATE__TOKEN`.

## Offline CLI

```bash
uv run x2loc parse XComGame.int -f json -o out/XComGame.json
uv run x2loc align XComGame.int XComGame.chn -f csv -o out/corpus.csv
uv run x2loc extract out/corpora/ -f csv -o out/glossary.csv
```

`parse` reads one file. `align` pairs a source file with its translation into a bilingual corpus. `extract` mines glossary terms from directories of corpus JSON files, giving priority to the directories listed first.

## Interactive translation agent

```bash
uv run x2loc agent run --config configs/weblate.local.toml --batch-size 20
```

The agent pulls the untranslated units of the mods glossary from Weblate, translates them, validates their markup tags, and scores each candidate. Approved translations are uploaded only after a person confirms each batch.

## HTTP service

```bash
uv run x2loc-api
```

Every route requires the header `Authorization: Bearer <service_token>`.

| Route | Purpose |
|---|---|
| `POST /v1/jobs` | Submit `{"workshop_url": "https://steamcommunity.com/sharedfiles/filedetails/?id=..."}`; optional LLM fields override `[agent]` |
| `GET /v1/jobs/{id}` | Job status and stage |
| `GET /v1/jobs/{id}/events` | Progress as server-sent events (SSE) |
| `POST /v1/jobs/{id}/cancel` | Cancel a running job |
| `GET /v1/jobs/{id}/artifact` | Download the overlay zip, with its digest in `X-Artifact-SHA256` |

A job downloads the mod with SteamCMD and syncs its localization into Weblate components. It then translates the missing units with automatic threshold review, writes the Chinese overlay files, adds newly mined terms to the custom glossary, and packages the result. Jobs and artifacts live in memory and under `data_root`, which is reset at every start.

The quality gate decides what reaches Weblate. A mod's own `.chn` translation is never imported directly: it is scored as the first candidate of its unit, and a rejected one becomes a reference for the retry. A unit that still fails after the last attempt is written as "needs editing" with its best candidate, and the scorer's reasons become a Weblate comment. A unit without any candidate whose tags match the source stays empty, and the overlay keeps its source text; the job progress counts these units as `units_untranslated`. Existing translations whose tags disagree with the source are cleared and translated again.

### Batch translation of a collection

```bash
docker compose exec x2loc x2loc batch \
  "https://steamcommunity.com/sharedfiles/filedetails/?id=<collection id>" \
  --max-size-mb 5 --limit 30
```

`x2loc batch` runs the job pipeline in process for every item of a public Workshop collection, one item at a time, and writes each overlay zip and `summary.json` to `output/batch/<UTC timestamp>/`. It differs from the service in two respects:

- Glossaries are cached on disk in `glossary_cache_dir`, the `glossary-cache` volume in Docker. The first run reads them in full, later runs pull only the units changed since the last sync, and a full read happens once a day.
- New custom glossary terms are queued locally and published when the run ends, while later jobs of the same run already use them. A run that stops early leaves the queue on disk, and the next run publishes it.

Run it inside the container, which holds SteamCMD and its login cache, and submit no service jobs meanwhile, because both would drive the same SteamCMD installation.

### Translating the mods installed on this machine

```bash
uv run x2loc local                 # every mod in [local] workshop_dir
uv run x2loc local 1122974240 667104300
```

`x2loc local` reads mods from the local Workshop content directory and runs the same pipeline as `batch`. It needs Weblate and the LLM endpoint, but neither SteamCMD, a Steam account, nor `service_token`. The two paths are set once in the `[local]` table or passed with `--workshop-dir` and `--mods-dir`. Mods without `.int` files under `Localization` are skipped.

Each translated mod is installed as the standalone mod `XComGame/Mods/x2loc_<workshop id>_zh/`, which holds the generated `.chn` files and its own `.XComMod`. A Steam update of the original mod therefore cannot overwrite the translation, and a new run replaces the previous one. The overlay mods are enabled in the mod launcher. Overlay zips and `summary.json` are also written to `output/local/<UTC timestamp>/`. Run this command on the computer where the game is installed, not in the container.

#### Without a Weblate token

With an empty `[weblate] token`, Weblate is only read. Translations already on Weblate are used as they are, the rest are translated with the configured LLM endpoint, and the results are kept in a local SQLite file, `store_path`, which defaults to `data/store.sqlite`. With a token set, `--store local` keeps the results locally as well, and `--store weblate` forces writing to Weblate. A token with write permission uploads the local results later:

```bash
uv run x2loc local 1122974240 --store local
uv run x2loc upload                # every queued translation
uv run x2loc upload <component slug>
```

`upload` never overwrites a unit that someone changed on Weblate after it was translated locally; it reports such a unit as a conflict.

When Weblate, the local store, and the mod itself each hold a translation of a unit, the overlay uses the first one available in `priority` order: `manual` (edited locally by hand), `upstream` (translated on Weblate), `machine` (translated locally), `needs_editing`, and `shipped` (shipped with the mod). The source text is the last resort.

### Docker

```bash
docker compose up -d --build
```

The container mounts `configs/weblate.local.toml` read only, listens on `127.0.0.1:8180`, and keeps SteamCMD, its login cache, and the Workshop download cache in named volumes.

## Local app

`x2loc-app` offers the same pipeline through a browser page on the local computer. The page covers settings (model, Weblate, game paths, and translation priority), the subscribed mods with their Weblate progress, runs, installed translations, review, and the upload queue. The app listens on a free port of `127.0.0.1` and opens the browser with a session token generated at each launch, which every API request must carry.

```bash
cd webui && pnpm install && pnpm build && cd ..   # builds into src/webui/dist
uv run x2loc-app
```

Settings and the local translation store live in `%LOCALAPPDATA%\x2loc\`, which `X2LOC_HOME` can override, and the LLM API key and the Weblate token are kept in the Windows Credential Manager. For interface development, run `uv run x2loc-app --port 8765 --token dev --no-browser` together with `pnpm dev` in `webui/`, then open `http://localhost:5173/#token=dev`.

Each release packages the app as one Windows executable, `x2loc.exe`, which needs neither Python nor Node.js. Its console window stays open while the page is in use, and closing the window stops the program. Pushing a `v*` tag runs `.github/workflows/release.yml`, which builds the pages, runs the tests, compiles the executable with Nuitka through `packaging/build.py`, and attaches it to a GitHub release. A local build runs as follows:

```bash
cd webui && pnpm build && cd ..
uv run --group build python packaging/build.py -- --zig   # without Visual Studio; omit --zig with MSVC
```

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
├── config.py      # layered service and CLI configuration (env > TOML > defaults)
├── core/          # UE3 parsing, alignment, term extraction, overlay writing, packaging
├── models/        # Pydantic schemas
├── export/        # CSV and JSON serializers for the offline CLI
├── services/      # Weblate and Steam clients, glossaries, translation stores
├── agent/         # LangGraph translation agent and its nodes
├── jobs/          # job manager and Workshop pipeline of the service
├── api/           # FastAPI service
├── app/           # local app: settings, secrets, runs, and its HTTP interface
├── cli/           # `x2loc` Typer entry point
└── ui/            # interactive review prompts
webui/             # pages of the local app (React, Vite, Tailwind)
packaging/         # Nuitka build of x2loc.exe
```

## License

MIT
