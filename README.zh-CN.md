# x2loc

[English](README.md) | 简体中文

XCOM 2 天选者之战（War of the Chosen）本地化工具。它解析 UE3 的 `.int`/`.chn` 文件，对齐原文与译文，并借助 Weblate 和大语言模型（LLM）翻译 mod；既可以在命令行中交互使用，也可以作为 HTTP 服务运行：输入 Steam 创意工坊链接，返回翻译好的汉化覆盖包。

## 下载

玩家可以从 [Releases](https://github.com/ITOTI-Y/x2loc/releases) 下载 `x2loc.exe` 并直接运行。程序会在浏览器中打开一个页面，扫描已订阅的 mod，用你自己的 OpenAI 兼容模型翻译缺失的文本，并把每个 mod 的汉化安装为独立的 mod。没有 Weblate token 时，程序只读取共享译文，自己翻译的结果保存在本机。详见[本地应用](#本地应用)。

## 环境要求

- Python 3.13 与 [uv](https://docs.astral.sh/uv/)
- 一个包含三个术语表组件（原版游戏、mod、自定义）的 Weblate 项目
- 一个 OpenAI 兼容的 LLM 接口（默认 OpenRouter）
- SteamCMD 与 Steam 账号，仅 HTTP 服务下载创意工坊内容时需要

## 配置

```bash
uv sync
cp configs/weblate.toml configs/weblate.local.toml   # 已加入 gitignore，在其中填写凭据
```

`configs/weblate.local.toml` 是命令行与 HTTP 服务共用的唯一配置文件：

| 表 | 用途 |
|---|---|
| 顶层 | `service_token`（API 的 Bearer 令牌）、`data_root`、`bind_port` |
| `[weblate]` | API 地址、token、项目标识、原文语言 |
| `[agent]` | LLM 的 API key、接口地址、各模型名（`retry_translation_model_name` 用于质量检查后的重译，留空则沿用翻译模型）、批大小、`units_per_request`（每次 LLM 请求打包的条目数）、自动通过阈值、`llm_timeout_seconds`（单次请求超时，默认 45 秒；弹性计费的接口会排队数分钟，需要调大） |
| `[steam]` | SteamCMD 可执行文件、下载目录、Steam 凭据 |
| `[glossary]` | 原版游戏、mod 与自定义三个术语表组件的标识 |

每个配置项都可以用带 `X2LOC_` 前缀的环境变量覆盖，嵌套层级用 `__` 连接，例如 `X2LOC_WEBLATE__TOKEN`。

## 离线命令行

```bash
uv run x2loc parse XComGame.int -f json -o out/XComGame.json
uv run x2loc align XComGame.int XComGame.chn -f csv -o out/corpus.csv
uv run x2loc extract out/corpora/ -f csv -o out/glossary.csv
```

`parse` 读取单个文件；`align` 把原文文件与其译文配对成双语语料；`extract` 从若干语料 JSON 目录中提取术语，排在前面的目录优先。

## 交互式翻译代理

```bash
uv run x2loc agent run --config configs/weblate.local.toml --batch-size 20
```

代理从 Weblate 拉取 mod 术语表中未翻译的条目，依次翻译、校验标签、为每个候选打分，每一批都请人工确认后再上传通过的译文。

## HTTP 服务

```bash
uv run x2loc-api
```

所有接口都要求请求头 `Authorization: Bearer <service_token>`。

| 接口 | 用途 |
|---|---|
| `POST /v1/jobs` | 提交 `{"workshop_url": "https://steamcommunity.com/sharedfiles/filedetails/?id=..."}`；LLM 相关字段可选，用于覆盖 `[agent]` |
| `GET /v1/jobs/{id}` | 任务状态与阶段 |
| `GET /v1/jobs/{id}/events` | 以服务器推送事件（SSE）发送进度 |
| `POST /v1/jobs/{id}/cancel` | 取消运行中的任务 |
| `GET /v1/jobs/{id}/artifact` | 下载汉化覆盖压缩包（响应头 `X-Artifact-SHA256`） |

一个任务会用 SteamCMD 下载 mod，把其本地化文本同步为 Weblate 组件，按阈值自动审核并翻译缺失条目，生成中文覆盖文件，把新提取的术语加入自定义术语表，最后打包结果。mod 自带的 `.chn` 译文不会直接导入：每条都作为该条目的第一个候选参与评分，只有通过质量检查才会进入 Weblate；未通过的会作为参考提供给重译。最后一次尝试后仍未通过的条目，以最佳候选写入并标记为“需要编辑”，评分器给出的理由作为 Weblate 评论；没有任何候选的标签与原文一致时，该条目保持为空，覆盖文件中保留原文（任务进度中的 `units_untranslated`）。标签与原文不一致的已有译文会被清空并重新翻译。任务与产物保存在内存和 `data_root` 中，每次启动都会重置。

### 批量翻译一个合集

```bash
docker compose exec x2loc x2loc batch \
  "https://steamcommunity.com/sharedfiles/filedetails/?id=<合集 id>" \
  --max-size-mb 5 --limit 30
```

`x2loc batch` 在进程内对一个公开创意工坊合集中的每个条目依次运行任务流水线，把每个覆盖压缩包和 `summary.json` 写入 `output/batch/<UTC 时间戳>/`。它与 HTTP 服务有两点不同：

- 术语表缓存在磁盘上（`glossary_cache_dir`，Docker 中对应 `glossary-cache` 卷）：第一次运行完整读取，之后只拉取上次同步后变化的条目，每天完整重读一次。
- 新的自定义术语先在本地排队，整次运行结束后统一发布；同一次运行中后面的任务已经可以使用它们。运行中途停止时，队列保留在磁盘上，由下一次运行发布。

请在容器内运行，那里有 SteamCMD 及其登录缓存；运行期间不要再提交服务任务，否则两者会同时操作同一个 SteamCMD。

### 翻译本机已安装的 mod

```bash
uv run x2loc local                 # [local] workshop_dir 中的全部 mod
uv run x2loc local 1122974240 667104300
```

`x2loc local` 直接从本机的创意工坊内容目录读取 mod，运行与 `batch` 相同的流水线，因此需要 Weblate 和 LLM 接口，但不需要 SteamCMD、Steam 账号和 `service_token`。在 `[local]` 表中设置一次两个路径即可，也可以用 `--workshop-dir` 和 `--mods-dir` 传入。`Localization` 下没有 `.int` 文件的 mod 会被跳过。

#### 没有 Weblate token 时

`[weblate] token` 为空时只读取 Weblate：已有的译文直接使用，其余部分用你自己的 LLM 接口翻译，结果保存在本地 SQLite 文件中（`store_path`，默认 `data/store.sqlite`），不写回 Weblate。设置了 token 时，`--store local` 可以达到同样效果；`--store weblate` 则强制写入 Weblate。之后可以用有写权限的 token 上传本地保存的译文：

```bash
uv run x2loc local 1122974240 --store local
uv run x2loc upload                # 上传全部待上传译文
uv run x2loc upload <组件标识>
```

某条目在本地翻译之后，如果 Weblate 上的译文已被他人修改，就不会被覆盖，`upload` 会把它列为冲突。

同一条目在 Weblate、本地和 mod 自身各有一份译文时，覆盖文件按 `priority` 的顺序取第一个可用的：`manual`（本地人工修改）、`upstream`（Weblate 已翻译）、`machine`（本地机器翻译）、`needs_editing`、`shipped`（mod 自带）；都没有时使用原文。

每个翻译好的 mod 都安装为独立的 mod `XComGame/Mods/x2loc_<创意工坊 id>_zh/`，其中包含生成的 `.chn` 文件和独立的 `.XComMod`，因此原 mod 的 Steam 更新不会覆盖汉化；重新运行会替换旧版本。请在 mod 启动器中启用这些汉化 mod。覆盖压缩包和 `summary.json` 同时写入 `output/local/<UTC 时间戳>/`。请在安装了游戏的电脑上运行，不要在容器中运行。

### Docker

```bash
docker compose up -d --build
```

容器以只读方式挂载 `configs/weblate.local.toml`，监听 `127.0.0.1:8180`，并把 SteamCMD、其登录缓存和创意工坊下载缓存保存在命名卷中。

## 本地应用

`x2loc-app` 通过本机浏览器页面提供同一套流水线：设置（模型、Weblate、游戏路径、译文优先级）、已订阅的 mod 及其 Weblate 进度、任务、已安装的汉化、审核以及待上传队列。它监听 `127.0.0.1` 上的空闲端口，并用每次启动时生成的会话令牌打开浏览器；所有 API 请求都必须携带该令牌。

```bash
cd webui && pnpm install && pnpm build && cd ..   # 构建到 src/webui/dist
uv run x2loc-app
```

设置保存在 `%LOCALAPPDATA%\x2loc\settings.json`（可用 `X2LOC_HOME` 指定其他目录），本地译文库也在同一目录；LLM 的 API key 与 Weblate token 保存在 Windows 凭据管理器中。开发界面时，运行 `uv run x2loc-app --port 8765 --token dev --no-browser`，并在 `webui/` 中运行 `pnpm dev`，然后打开 `http://localhost:5173/#token=dev`。

发布版把本地应用打包为单个 Windows 可执行文件 `x2loc.exe`，无需安装 Python 或 Node.js：运行后在打开的页面中使用，期间保持其控制台窗口打开，关闭窗口即退出。推送 `v*` 标签会运行 `.github/workflows/release.yml`：构建页面、运行测试、用 Nuitka 编译可执行文件（`packaging/build.py`），并附加到 GitHub Release。本地构建：

```bash
cd webui && pnpm build && cd ..
uv run --group build python packaging/build.py -- --zig   # 没有 Visual Studio 时使用；有 MSVC 时省略 --zig
```

## 武器类别名称 mod

`mods/x2locWeaponCatNames` 是一个小型脚本 mod。[WOTCMusashiModFixes](https://steamcommunity.com/sharedfiles/filedetails/?id=2569592723) 会在技能说明末尾追加一行“可用武器”，其中由 mod 新增的武器类别显示为内部代号，例如 `combatknife`；本 mod 把每个代号替换为该类别武器自身的本地化名称，跟随游戏当前语言。它需要天选者之战、[X2WOTCCommunityHighlander](https://steamcommunity.com/sharedfiles/filedetails/?id=1134256495) 和 WOTCMusashiModFixes，本地应用可以在“已安装汉化”页中安装它。

```bash
uv run python mods/x2locWeaponCatNames/build.py --sdk <WOTC SDK> --mods-dir <XComGame/Mods>
uv run python mods/x2locWeaponCatNames/publish.py --steamcmd <steamcmd.exe> --username <Steam 账号>
```

`build.py` 用天选者之战 SDK 编译并安装它；`publish.py` 用 [SteamCMD](https://developer.valvesoftware.com/wiki/SteamCMD) 把它上传到 Steam 创意工坊，简介取自 `workshop/description.bbcode`，预览图为 `ModPreview.jpg`。标签和必需物品需要在条目的创意工坊页面上设置。

## 开发

```bash
uv run pytest
uv run ruff check --fix . && uv run ruff format .
uv run ty check src tests
cd webui && pnpm lint && pnpm check:i18n && pnpm build
```

## 目录结构

```
src/
├── config.py      # 分层的服务与命令行配置（环境变量 > TOML > 默认值）
├── core/          # UE3 解析、对齐、术语提取、覆盖文件写入、打包
├── models/        # Pydantic 数据模型
├── export/        # 离线命令行的 CSV/JSON 序列化
├── services/      # Weblate 与 Steam 客户端、术语表、译文存储
├── agent/         # LangGraph 翻译代理及其节点
├── jobs/          # 服务的任务管理器与创意工坊流水线
├── api/           # FastAPI 服务
├── app/           # 本地应用：设置、密钥、任务及其 HTTP 接口
├── cli/           # `x2loc` Typer 入口
└── ui/            # 交互式审核提示
webui/             # 本地应用的页面（React、Vite、Tailwind）
packaging/         # 用 Nuitka 构建 x2loc.exe
mods/x2locWeaponCatNames/   # 武器类别名称脚本 mod
```

## 许可证

MIT
