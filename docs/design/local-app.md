# x2loc 本地应用设计

- **状态**：已审阅，按第 9 节的决定实施
- **日期**：2026-10-12
- **范围**：面向普通玩家的本地应用，包括存储抽象、本地接口、网页界面、打包与发布

## 1. 背景与目标

x2loc 目前只能通过命令行使用，运行前要准备 TOML 配置文件、Weblate token、模型接口和本机路径，翻译结果直接写回 Weblate。这套流程适合维护者本人，其他玩家很难上手：没有 Weblate token 的玩家完全无法运行，因为同步原文和写回译文都需要写权限。

本设计的目标是让任何玩家下载一个 exe 后，在浏览器界面里完成配置、扫描已订阅的 mod、翻译并安装中文覆盖。没有 token 时以只读方式使用 Weblate 上的公共译文，缺少的部分用玩家自己的模型在本机补译，结果保存在本机；有可写 token 时可以把本机结果推送到 Weblate。所有文件读写和模型调用都在玩家本机进行。

## 2. 已确认的需求

| 项目 | 决定 |
|---|---|
| 运行形态 | 本地程序自带网页：启动 exe 后在浏览器打开 `http://127.0.0.1:<端口>` |
| 无 token | 读取 Weblate 已有译文，缺失部分在本机用玩家配置的模型翻译并经过质量门，结果存本机 |
| 有可写 token | 支持两种上传方式：任务完成后自动推送；或本机积累，之后在“待上传”列表中挑选上传 |
| 审核 | 有 token 的玩家修改后提交到 Weblate；无 token 的玩家修改只作用于本机覆盖，同时进入待上传列表 |
| 密钥 | 模型 API key 和 Weblate token 存入 Windows 凭据管理器，配置文件中不出现明文 |
| 前端 | React、Vite、Tailwind CSS；构建产物打包进 exe，玩家无需安装 Node.js |
| 风格 | 简洁工具风格，浅色和深色随系统自动切换 |
| 语言 | 中文和英文，使用 i18n JSON 语言包，可扩展到其他语言 |
| 分发 | Nuitka 打包为 Windows 单文件 exe，GitHub Actions 在打标签时构建并发布 Release |

## 3. 总体架构

本地应用是一个进程：FastAPI 服务同时提供接口和网页静态文件，只监听 `127.0.0.1`。exe 启动后选择空闲端口，打开浏览器，并在系统托盘或控制台窗口中保持运行；关闭窗口即退出。

```text
浏览器（React 界面）
   │  HTTP + SSE，附带启动时生成的会话令牌
   ▼
本地服务（FastAPI，127.0.0.1）
   ├─ 配置与密钥：config.toml + Windows 凭据管理器
   ├─ 路径探测：Steam 注册表 + libraryfolders.vdf
   ├─ 任务管理：JobManager（沿用），流水线 WorkshopPipeline（沿用）
   ├─ 译文存储：TranslationStore（新增抽象）
   │    ├─ WeblateStore：有可写 token，读写 Weblate（现有逻辑）
   │    └─ LocalStore：无 token，只读 Weblate + 本机 SQLite
   └─ 覆盖管理：install_overlay（沿用）+ 已安装覆盖的扫描与卸载
```

本机数据的存放位置如下。程序本体是单文件 exe，不在安装目录写入任何数据。

| 数据 | 位置 |
|---|---|
| 配置（路径、模型名、阈值、界面语言） | `%APPDATA%\x2loc\config.toml` |
| API key、Weblate token | Windows 凭据管理器，服务名 `x2loc` |
| 本机译文库、待上传队列 | `%LOCALAPPDATA%\x2loc\store.sqlite` |
| 术语表缓存（沿用 `LocalGlossaryStore`） | `%LOCALAPPDATA%\x2loc\glossaries\` |
| 任务日志与运行汇总 | `%LOCALAPPDATA%\x2loc\runs\<时间戳>\` |

### 3.1 本地接口的安全

本地服务只监听回环地址，但浏览器中任何网页都可以向 `127.0.0.1` 发请求，恶意网页可能借此读取译文、触发任务，甚至改写游戏目录。为此采取三项措施：

- 启动时生成随机会话令牌，只通过打开浏览器的地址传给界面；所有接口要求请求头携带该令牌，缺失时返回 401。
- 校验 `Host` 头必须是 `127.0.0.1:<端口>` 或 `localhost:<端口>`，防止 DNS 重绑定攻击。
- 不开放跨域访问（不返回任何 CORS 允许头）。

接口返回的配置中，密钥字段只给出“已设置”或“未设置”，不回传明文。

## 4. 译文存储抽象

### 4.1 现有代码对 Weblate 的依赖

现有流水线把 Weblate 当作唯一的存储，读写调用分布如下：

| 位置 | 调用 | 用途 |
|---|---|---|
| `jobs/pipeline.py` `_sync` | `sync_corpus`、`download_units` | 创建组件、追加原文、读取快照 |
| `jobs/pipeline.py` | `list_units`、`patch_unit` | 原文变更处理、清除标签不匹配的译文 |
| `jobs/pipeline.py` 回读 | `download_units` | 生成覆盖所用的权威译文 |
| `agent/nodes/fetch_empty.py` | `list_units(q="state:empty")` | 取待翻译单元 |
| `agent/nodes/uploader.py` | `patch_unit`、`add_unit_comment` | 写回译文和拒绝原因 |
| `services/context_index.py` | `list_component_slugs`、`download_units` | 跨 mod 上下文索引 |
| `services/glossary*.py` | `list_units`、`create_unit`、`upload_targets` | 术语表读取与新术语发布 |

其中写操作在无 token 时全部不可用。读操作在 Weblate 项目开放匿名下载后全部可用：2026-10-12 实测匿名下载 `glossary-custom` 的译文文件（2.6 MB）耗时 2.2 s。

### 4.2 接口定义

新增 `TranslationStore` 协议，流水线和图节点只通过它读写译文。方法按上表的用途划分：

| 方法 | WeblateStore | LocalStore |
|---|---|---|
| `prepare(work) -> snapshot` | 现有 `_sync` 全部逻辑 | 读取 Weblate 上该组件的译文（组件不存在时为空），与本机库合并 |
| `pending(slug) -> units` | `list_units(q="state:empty")` | 原文在本机、且上游与本机都没有译文的单元 |
| `write(decisions)` | `patch_unit` + `add_unit_comment` | 写入本机库，标记来源为“机器翻译”，加入待上传队列 |
| `authoritative(work) -> dict` | 现有回读逻辑 | 按 4.3 节的优先级合并上游与本机 |
| `corpus() -> snapshot` | 现有上下文索引来源 | 上游快照与本机译文合并 |

单元标识的处理：Weblate 单元 id 只在上游存在，本机库以 `(组件 slug, key)` 作为主键，图节点内部的 `WeblateUnitSchema.id` 在本地模式下使用本机库的自增 id。

术语表沿用 `LocalGlossaryStore`：它已经实现了本机缓存和“待发布”队列，无 token 时新术语只进入本机队列，界面上和译文一起出现在待上传列表中。

### 4.3 合并优先级

同一个单元在上游和本机都有译文时，生成本机覆盖按优先级顺序取第一个存在的值。顺序保存在配置中，玩家可以在设置页拖动调整；默认顺序如下：

1. 本机人工修改的译文；
2. 上游状态为“已翻译”或“已批准”的译文；
3. 本机机器翻译并通过质量门的译文；
4. 上游或本机状态为“需要编辑”的最佳候选；
5. mod 自带的译文（沿用 `_with_shipped`，标签不合法的不用）；
6. 原文。

默认顺序的依据是：玩家亲手改过的内容最能代表本人意图；上游已翻译的内容经过了维护者流程，可信度高于本机未经他人审核的机器结果。“原文”固定在末位，不参与拖动。

### 4.4 上传

待上传队列中的每条记录保存：组件 slug、key、原文、译文、来源（机器翻译或人工修改）、质量门分数与评论，以及读取时的上游译文。上传时逐条处理：

- 上游组件不存在：先执行 `sync_corpus` 创建组件，这要求 token 具有创建组件的权限，否则该 mod 的记录整体标为失败并显示原因。
- 上游译文与读取时相同：写入，状态按来源设置（人工修改为“已翻译”，未达阈值的机器结果为“需要编辑”并附评论）。
- 上游译文已被他人改动：不覆盖，标为冲突，在界面上并排显示双方译文，由玩家选择保留哪一份。

“任务完成后自动推送”等价于任务结束时对本次产生的记录自动执行一次上传。

### 4.5 写权限的判定

填入 token 后，程序调用 Weblate 的用户信息接口确认 token 有效，并检查对项目的权限；无效或只读时按无 token 处理，界面显示当前模式。上传途中遇到 403，当前批次停止，剩余记录保留在队列中。

## 5. 本地接口

所有接口以 `/api` 为前缀，返回 JSON；错误统一返回 `{"code": "...", "detail": {...}}`，界面按 `code` 查语言包显示，后端不返回成句的提示文字。

| 方法与路径 | 用途 |
|---|---|
| `GET /api/config`、`PUT /api/config` | 读取和保存配置；密钥字段只返回是否已设置 |
| `PUT /api/secrets/{name}`、`DELETE /api/secrets/{name}` | 写入或清除 API key、Weblate token |
| `POST /api/config/test` | 测试模型接口连通性和 Weblate token 权限 |
| `GET /api/paths/detect` | 自动探测创意工坊目录和游戏 Mods 目录 |
| `GET /api/mods` | 扫描已订阅 mod，返回名称、可翻译文件数、上游覆盖率、本机覆盖安装状态 |
| `POST /api/jobs`、`GET /api/jobs`、`GET /api/jobs/{id}/events` | 提交翻译任务、查看列表、通过 SSE 接收实时进度 |
| `POST /api/jobs/{id}/cancel` | 取消任务 |
| `GET /api/overlays`、`DELETE /api/overlays/{id}` | 已安装覆盖的列表与卸载 |
| `GET /api/review`、`PATCH /api/review/{unit}` | “需要编辑”单元及拒绝原因；提交修改 |
| `GET /api/uploads`、`POST /api/uploads`、`POST /api/uploads/{id}/resolve` | 待上传列表、执行上传、处理冲突 |

路径探测读取注册表 `HKCU\Software\Valve\Steam` 的 `SteamPath`，再解析 `steamapps\libraryfolders.vdf` 找到所有 Steam 库，在其中查找 `workshop\content\268500` 和 `common\XCOM 2\XCom2-WarOfTheChosen\XComGame\Mods`。

## 6. 网页界面

### 6.1 技术选型

| 部分 | 选择 | 理由 |
|---|---|---|
| 框架 | React + Vite | 实时进度、批量勾选和审核编辑需要较多客户端交互 |
| 样式 | Tailwind CSS v4 | 浅色、深色主题用 `dark:` 变体统一处理 |
| 组件 | shadcn/ui，按项目风格调整圆角与配色 | 组件代码归项目所有，可访问性完整 |
| 数据请求 | TanStack Query | 处理缓存、重试和任务状态轮询 |
| 多语言 | react-i18next，`locales/zh-CN.json`、`locales/en.json` | 新增语言只需添加一个 JSON 文件 |
| 图标 | Phosphor Icons | 线条风格统一 |

视觉上采用中性灰底色加一个强调色，等宽字体只用于 mod ID、key 和译文中的标签；不使用渐变、发光和大面积卡片阴影。

### 6.2 页面结构

左侧固定导航，五个页面：模组、任务、审核、待上传、设置。首次启动若配置不完整，直接进入设置页的引导流程。

```text
┌──────────┬────────────────────────────────────────────────────────┐
│ x2loc    │ 模组                         [搜索…]   [翻译所选 (12)] │
│          ├────────────────────────────────────────────────────────┤
│ ▣ 模组   │ ☐  名称                 ID           上游覆盖   本机覆盖 │
│ ▷ 任务   │ ☑  Long War of the …    2663990965   98.6%     已安装  │
│ ✎ 审核 3 │ ☑  Expanded Playable…   3240087035   99.2%     已安装  │
│ ⇡ 待上传 │ ☐  Rider Class          2042813766   100%      未安装  │
│ ⚙ 设置   │ …                                                      │
│          │                                                        │
│ 只读模式 │                                                        │
└──────────┴────────────────────────────────────────────────────────┘
```

```text
任务详情
┌──────────────────────────────────────────────────────────────────┐
│ Long War of the Chosen          翻译中   1 284 / 1 590 单元       │
│ ███████████████████████░░░░░                                     │
│ 同步 ✓   翻译 …   写入覆盖   提取术语   安装                       │
├──────────────────────────────────────────────────────────────────┤
│ 日志（可折叠，只显示警告以上，可切换为全部）                       │
└──────────────────────────────────────────────────────────────────┘
```

```text
审核
┌───────────────────────────────┬──────────────────────────────────┐
│ 单元列表（按 mod 分组）        │ 原文                              │
│ ▸ Musashi's Mods Fixes   2    │ Show valid weapon categories …    │
│ ▾ Long War of the Chosen 1    │ 当前译文（可编辑）                │
│   PoisonChanceLabel           │ [中毒几率                       ] │
│                               │ 质量门评论                        │
│                               │ score 85：术语与官方表不一致 …    │
│                               │ [保存到本机]  [提交到 Weblate]    │
└───────────────────────────────┴──────────────────────────────────┘
```

设置页分为五组：模型（base URL、翻译模型、评分模型、API key、并发数、质量阈值）、Weblate（地址、token、当前模式）、路径（创意工坊目录、游戏 Mods 目录，各带“自动探测”和“浏览”）、译文优先级（4.3 节各来源的可拖动列表，附“恢复默认”）、界面（语言）。每组带“测试连接”按钮，结果就地显示。

### 6.3 状态与反馈

每个列表都提供加载、空和错误三种状态：加载用与最终布局同形的骨架，空状态说明下一步操作（例如“尚未设置创意工坊目录，前往设置”），错误就地显示并给出重试入口。长任务的进度通过 SSE 推送，浏览器刷新后可以重新连接到进行中的任务。

## 7. 打包与发布

| 步骤 | 做法 |
|---|---|
| 前端构建 | `npm ci && npm run build`，产物输出到 `src/webui/dist/` |
| 打包 | Nuitka `--onefile --standalone`，用 `--include-data-dir` 带上前端产物和武器类别 mod 的预编译文件 |
| 入口 | `x2loc-app`：选端口、启动服务、打开浏览器；保留现有 `x2loc` 命令行 |
| 发布 | GitHub Actions，推送 `v*` 标签时在 `windows-latest` 上构建，把 exe 上传到 Release |

未做代码签名时，Windows SmartScreen 会提示“未知发布者”，玩家需要点“仍要运行”。签名证书需要另行购买，本阶段不包含。

武器类别名 mod（`mods/x2locWeaponCatNames`）的编译需要 XCOM 2 SDK，无法在 GitHub Actions 中完成。做法是在本机编译后把 `.u` 文件提交到仓库，打包时带入 exe，玩家在覆盖管理页一键安装。

## 8. 实施阶段与验收

| 阶段 | 内容 | 验收标准 |
|---|---|---|
| 1 | `TranslationStore` 抽象、`LocalStore`（SQLite）、待上传队列与上传冲突处理 | 无 token 时对一个 mod 完成翻译并生成覆盖，上游无任何写入；填入可写 token 后上传成功，冲突记录被正确拦下 |
| 2 | 本地接口、会话令牌与 Host 校验、路径探测、密钥存储 | 接口测试覆盖主要路径；缺少令牌或错误 Host 的请求被拒绝 |
| 3 | 前端五个页面与中英文语言包 | 浅色和深色下各页面完整可用；切换语言后无遗漏的硬编码文字 |
| 4 | Nuitka 打包、GitHub Actions 发布 | 在未安装 Python 和 Node.js 的 Windows 上运行 exe，完成一次完整翻译 |

## 9. 已确认的决定

1. **匿名下载**：Weblate 项目已开放匿名下载译文文件，只读模式与可写模式使用同一套读取路径（`download_units`）。
2. **合并优先级**：默认按 4.3 节的顺序，玩家可在设置页拖动调整。
3. **上传冲突**：上游译文在读取之后被他人改动时，不覆盖，标为冲突，由玩家选择保留哪一份。
4. **命令行**：现有 `x2loc local`、`x2loc batch` 等命令保留，改为使用同一套 `TranslationStore`。

## 10. 实施清单

每完成一项即勾选并在括号中注明提交号。清单之外的改动先补进清单再做。

### 阶段 1：存储抽象与本机译文库

接口分两层。`UnitStore` 供图节点使用，只有 `pending`（取待翻译单元）和 `write`（写回一条决定）；`TranslationStore` 在其上增加 `prepare`（同步前准备，返回快照、旧译文和是否需要翻译）与 `candidates`（回读时按来源分类给出候选译文）。合并优先级在流水线中统一执行，两种存储给出同样分类的候选。

- [ ] 1.1 配置：`WeblateConfigSchema.token` 可为空，token 为空时客户端不发送 `Authorization` 头；新增 `store`（`auto`、`weblate`、`local`）与 `priority`（来源顺序，校验为合法排列）配置项
- [ ] 1.2 新建 `src/services/stores/`：定义 `UnitStore`、`TranslationStore` 协议与候选分类 `SourceKind`（manual、upstream、machine、needs_editing、shipped）
- [ ] 1.3 `WeblateStore`：把流水线中的 `_sync`、`_true_sources`、`_clear_changed_sources`、`_clear_tag_mismatches` 和回读逻辑原样迁入；`pending` 为 `state:empty` 查询，`write` 为 `patch_unit` 加评论
- [ ] 1.4 “需要编辑”单元的识别：按项目一次查询 `state:needs-editing` 单元，得到 `(组件, key)` 集合，供两种存储区分 upstream 与 needs_editing
- [ ] 1.5 图节点改用 `UnitStore`：`UnitIterator`、`BackgroundUploader`、`WorkflowNodes`、`build_graph` 接收 store；交互式命令行默认使用 `WeblateStore`
- [ ] 1.6 流水线改用 `TranslationStore`，按 `priority` 合并候选生成覆盖；`_with_shipped` 并入 shipped 来源
- [ ] 1.7 `LocalStore`：SQLite 表 `units`（id、slug、key）、`sources`（slug、key、原文、注释、组件名）、`translations`（来源、状态、译文、评论、读取时的上游译文、上传状态）；实现 `prepare`、`pending`、`write`、`candidates`；上游组件不存在时按空处理
- [ ] 1.8 原文变更：上游或本机译文对应的原文与当前原文不一致时，不作为候选，改作旧译文参考
- [ ] 1.9 上传：按组件分组；上游组件不存在时先 `sync_corpus`；读取时的上游译文与当前一致则写入，不一致则标为冲突；提供冲突处理方法（保留本机或保留上游）
- [ ] 1.10 命令行：`x2loc local` 与 `x2loc batch` 增加 `--store`；新增 `x2loc upload`；只读模式下术语只入本机队列不发布
- [ ] 1.11 HTTP 服务（`src/api/app.py`）按配置选择存储
- [ ] 1.12 测试：两种存储的单元测试、优先级合并、原文变更、上传成功与冲突；全部测试、ruff、ty 通过
- [ ] 1.13 验收：无 token 对一个 mod 完成翻译并生成覆盖，上游无写入；填入可写 token 后上传成功，冲突被拦下

### 阶段 2：本地应用接口

- [ ] 2.1 配置读写与密钥存储（keyring，服务名 `x2loc`），接口不回传密钥明文
- [ ] 2.2 会话令牌、`Host` 校验、仅监听回环地址
- [ ] 2.3 `POST /api/config/test`：模型连通性、Weblate token 有效性与写权限
- [ ] 2.4 路径探测：注册表 `SteamPath` + `libraryfolders.vdf`
- [ ] 2.5 `GET /api/mods`：扫描已订阅 mod，给出可翻译文件数、上游覆盖率、本机覆盖安装状态
- [ ] 2.6 任务接口：提交、列表、SSE 进度、取消；任务完成后安装覆盖，可选自动推送
- [ ] 2.7 覆盖管理：列表、卸载、安装武器类别名 mod
- [ ] 2.8 审核接口：上游与本机“需要编辑”单元及评论；有可写 token 时提交到 Weblate，否则保存为本机人工修改
- [ ] 2.9 待上传接口：列表、上传、冲突处理
- [ ] 2.10 错误统一返回错误码；接口测试覆盖主要路径与拒绝场景

### 阶段 3：网页界面

- [ ] 3.1 工程搭建：Vite、React、Tailwind v4、shadcn/ui、TanStack Query、react-i18next、Phosphor Icons；构建输出到 `src/webui/dist/`
- [ ] 3.2 布局与主题：左侧导航，浅色与深色随系统切换
- [ ] 3.3 设置页：模型、Weblate、路径、译文优先级拖动排序、界面语言；首次启动引导
- [ ] 3.4 模组页：列表、搜索、勾选、提交翻译
- [ ] 3.5 任务页：实时进度、分阶段状态、日志
- [ ] 3.6 审核页：分组列表、原文与译文对照编辑、质量门评论
- [ ] 3.7 待上传页：列表、勾选上传、冲突并排对比
- [ ] 3.8 加载、空、错误三种状态；`zh-CN.json`、`en.json` 无遗漏文字

### 阶段 4：打包与发布

- [ ] 4.1 入口 `x2loc-app`：选端口、启动服务、打开浏览器
- [ ] 4.2 Nuitka 单文件打包，带入前端产物与武器类别名 mod 的 `.u`
- [ ] 4.3 GitHub Actions：推送 `v*` 标签时构建前端、打包 exe、发布 Release
- [ ] 4.4 在未安装 Python 与 Node.js 的 Windows 上完成一次完整翻译
