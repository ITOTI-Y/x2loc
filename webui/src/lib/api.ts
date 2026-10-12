// Typed client of the local app's HTTP interface (src/app/api.py).

export type SourceKind = "manual" | "upstream" | "machine" | "needs_editing" | "shipped"
export type StoreMode = "auto" | "weblate" | "local"
export type UiLanguage = "zh-CN" | "en"
export type SecretName = "llm_api_key" | "weblate_token"

export interface ModelSettings {
  base_url: string
  translation_model: string
  retry_translation_model: string
  validation_model: string
  scoring_model: string
  batch_size: number
  units_per_request: number
  auto_approve_threshold: number
  timeout_seconds: number
  concurrency: number
}

export interface AppSettings {
  model: ModelSettings
  weblate: { url: string; project_slug: string }
  paths: { workshop_dir: string | null; mods_dir: string | null }
  store: StoreMode
  priority: SourceKind[]
  jobs: number
  language: UiLanguage
}

export interface SettingsView {
  settings: AppSettings
  secrets: Record<SecretName, boolean>
}

export interface Check {
  ok: boolean
  code: string | null
  detail: string
}

export interface SettingsCheck {
  llm: Check
  weblate: Check & { username: string | null; can_write: boolean }
}

export interface Coverage {
  translated: number
  total: number
  components_found: number
}

export interface ModView {
  workshop_id: string
  title: string
  slugs: string[]
  overlay_installed: boolean
  problem: "no_localization" | "invalid_mod" | null
  coverage: Coverage | null
}

export interface ModsView {
  mods: ModView[]
  coverage_state: "loading" | "ready" | "failed"
}

export type JobStatus = "queued" | "running" | "succeeded" | "cancelled" | "failed"
export type JobStage =
  | "downloading"
  | "discovering"
  | "syncing_weblate"
  | "translating"
  | "writing"
  | "extracting_terms"
  | "packaging"

export interface Job {
  id: string
  workshop_id: string
  title: string
  status: JobStatus
  stage: JobStage | null
  progress: {
    files_total: number
    files_completed: number
    units_total: number
    units_translated: number
    units_untranslated: number
    terms_added: number
    terms_skipped: number
  }
  error_code: string | null
  created_at: string
  finished_at: string | null
}

export type RunStatus = "running" | "succeeded" | "failed" | "cancelled"

export interface Run {
  id: string
  status: RunStatus
  workshop_ids: string[]
  upload_after: boolean
  started_at: string
  finished_at: string | null
  skipped: Record<string, string>
  jobs: Record<string, Job>
  installed: Record<string, string>
  upload: UploadReport | null
  error_code: string | null
}

export interface UploadReport {
  uploaded: number
  conflicts: number
}

export interface OverlaysView {
  overlays: { workshop_id: string; title: string }[]
  weapon_names_installed: boolean
}

export interface ReviewItem {
  where: "weblate" | "local"
  slug: string
  context: string
  source: string
  target: string
  upstream: string
  comment: string
  unit_id: number | null
  web_url: string
}

export interface QueuedTranslation {
  slug: string
  language: string
  context: string
  origin: "machine" | "manual"
  source: string
  target: string
  state: number
  comment: string
  upload: "pending" | "uploaded" | "conflict"
  upstream_at_read: string
  upstream_now: string
}

export class ApiError extends Error {
  readonly code: string
  readonly detail: string
  readonly status: number

  constructor(code: string, detail: string, status: number) {
    super(code)
    this.code = code
    this.detail = detail
    this.status = status
  }
}

const TOKEN_KEY = "x2loc.session"

// The launcher opens the app at /#token=…; the token is moved out of the
// address bar and kept for reloads of this tab only.
function sessionToken(): string {
  const match = /token=([^&]+)/.exec(window.location.hash)
  if (match) {
    sessionStorage.setItem(TOKEN_KEY, match[1])
    history.replaceState(null, "", window.location.pathname)
    return match[1]
  }
  return sessionStorage.getItem(TOKEN_KEY) ?? ""
}

const token = sessionToken()

export function hasSession(): boolean {
  return token !== ""
}

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  let response: Response
  try {
    response = await fetch(path, {
      method,
      headers: {
        "X-Session-Token": token,
        ...(body === undefined ? {} : { "Content-Type": "application/json" }),
      },
      body: body === undefined ? undefined : JSON.stringify(body),
    })
  } catch {
    throw new ApiError("app_unreachable", "", 0)
  }
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}))
    throw new ApiError(payload.code ?? "internal_error", payload.detail ?? "", response.status)
  }
  return response.status === 204 ? (undefined as T) : ((await response.json()) as T)
}

export const api = {
  settings: () => request<SettingsView>("GET", "/api/settings"),
  saveSettings: (settings: AppSettings) => request<SettingsView>("PUT", "/api/settings", settings),
  setSecret: (name: SecretName, value: string) =>
    request<void>("PUT", `/api/secrets/${name}`, { value }),
  deleteSecret: (name: SecretName) => request<void>("DELETE", `/api/secrets/${name}`),
  testSettings: () => request<SettingsCheck>("POST", "/api/settings/test"),
  detectPaths: () =>
    request<AppSettings["paths"]>("GET", "/api/paths/detect"),
  mods: (refresh = false) => request<ModsView>("GET", `/api/mods${refresh ? "?refresh=true" : ""}`),
  startRun: (workshopIds: string[], uploadAfter: boolean) =>
    request<Run>("POST", "/api/runs", { workshop_ids: workshopIds, upload_after: uploadAfter }),
  runs: () => request<Run[]>("GET", "/api/runs"),
  cancelRun: (id: string) => request<Run>("POST", `/api/runs/${id}/cancel`),
  cancelJob: (id: string) => request<Job>("POST", `/api/jobs/${id}/cancel`),
  overlays: () => request<OverlaysView>("GET", "/api/overlays"),
  uninstallOverlay: (workshopId: string) =>
    request<void>("DELETE", `/api/overlays/${workshopId}`),
  installWeaponNames: () => request<void>("PUT", "/api/weapon-names"),
  uninstallWeaponNames: () => request<void>("DELETE", "/api/weapon-names"),
  review: () => request<ReviewItem[]>("GET", "/api/review"),
  saveReview: (item: ReviewItem, target: string) =>
    request<void>("PUT", "/api/review", {
      slug: item.slug,
      context: item.context,
      source: item.source,
      target,
      upstream: item.upstream,
      unit_id: item.unit_id,
    }),
  uploads: () => request<QueuedTranslation[]>("GET", "/api/uploads"),
  upload: (slugs: string[] | null) => request<UploadReport>("POST", "/api/uploads", { slugs }),
  resolve: (row: QueuedTranslation, keep: "local" | "upstream") =>
    request<void>("POST", "/api/uploads/resolve", {
      slug: row.slug,
      context: row.context,
      origin: row.origin,
      keep,
    }),
}
