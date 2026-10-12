import {
  CheckCircleIcon,
  CircleNotchIcon,
  KeyIcon,
  MagnifyingGlassIcon,
  PlugsConnectedIcon,
  XCircleIcon,
} from "@phosphor-icons/react"
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { useState, type ReactNode } from "react"
import { useTranslation } from "react-i18next"
import { toast } from "sonner"

import { ErrorNotice, PageHeader, useCodeText, useErrorText } from "@/components/common"
import { PriorityList } from "@/components/PriorityList"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Skeleton } from "@/components/ui/skeleton"
import {
  api,
  type AppSettings,
  type Check,
  type SecretName,
  type SettingsCheck,
  type SettingsView,
  type StoreMode,
} from "@/lib/api"
import { keys, useSettings } from "@/lib/queries"
import { useTheme, type ThemeMode } from "@/lib/theme"

function Section({
  title,
  description,
  children,
}: {
  title: string
  description: string
  children: ReactNode
}) {
  return (
    <section className="grid gap-x-10 gap-y-4 border-b py-8 last:border-b-0 lg:grid-cols-[16rem_1fr]">
      <div>
        <h2 className="font-medium">{title}</h2>
        <p className="mt-1 text-sm text-muted-foreground">{description}</p>
      </div>
      <div className="grid max-w-2xl gap-5">{children}</div>
    </section>
  )
}

function Field({
  id,
  label,
  hint,
  children,
}: {
  id: string
  label: string
  hint?: string
  children: ReactNode
}) {
  return (
    <div className="grid gap-2">
      <Label htmlFor={id}>{label}</Label>
      {children}
      {hint && <p className="text-xs text-muted-foreground">{hint}</p>}
    </div>
  )
}

function NumberInput({
  id,
  value,
  onChange,
  min,
  max,
  step,
}: {
  id: string
  value: number
  onChange: (value: number) => void
  min?: number
  max?: number
  step?: number
}) {
  return (
    <Input
      id={id}
      type="number"
      inputMode="numeric"
      className="w-32 tabular-nums"
      value={value}
      min={min}
      max={max}
      step={step}
      onChange={(event) => {
        const next = event.target.valueAsNumber
        if (!Number.isNaN(next)) onChange(next)
      }}
    />
  )
}

function SecretField({
  name,
  label,
  hint,
  isSet,
}: {
  name: SecretName
  label: string
  hint: string
  isSet: boolean
}) {
  const { t } = useTranslation()
  const errorText = useErrorText()
  const queryClient = useQueryClient()
  const [value, setValue] = useState("")
  const refresh = () => queryClient.invalidateQueries({ queryKey: keys.settings })
  const save = useMutation({
    mutationFn: () => api.setSecret(name, value),
    onSuccess: () => {
      setValue("")
      toast.success(t("settings.secret.saved"))
      void refresh()
    },
    onError: (error) => toast.error(errorText(error)),
  })
  const remove = useMutation({
    mutationFn: () => api.deleteSecret(name),
    onSuccess: () => void refresh(),
    onError: (error) => toast.error(errorText(error)),
  })
  return (
    <div className="grid gap-2">
      <div className="flex items-center gap-2">
        <Label htmlFor={name}>{label}</Label>
        {isSet ? (
          <Badge variant="secondary" className="gap-1">
            <KeyIcon />
            {t("settings.secret.stored")}
          </Badge>
        ) : (
          <Badge variant="outline">{t("settings.secret.not_set")}</Badge>
        )}
      </div>
      <form
        className="flex gap-2"
        onSubmit={(event) => {
          event.preventDefault()
          if (value) save.mutate()
        }}
      >
        <Input
          id={name}
          type="password"
          autoComplete="off"
          placeholder={isSet ? t("settings.secret.replace") : ""}
          value={value}
          onChange={(event) => setValue(event.target.value)}
        />
        <Button type="submit" variant="outline" disabled={!value || save.isPending}>
          {t("common.save")}
        </Button>
        {isSet && (
          <Button
            type="button"
            variant="ghost"
            disabled={remove.isPending}
            onClick={() => remove.mutate()}
          >
            {t("settings.secret.clear")}
          </Button>
        )}
      </form>
      <p className="text-xs text-muted-foreground">{hint}</p>
    </div>
  )
}

function CheckLine({ label, check, extra }: { label: string; check: Check; extra?: string }) {
  const codeText = useCodeText()
  const { t } = useTranslation()
  return (
    <div className="flex items-start gap-2 text-sm">
      {check.ok ? (
        <CheckCircleIcon className="mt-0.5 size-4 shrink-0 text-success" weight="fill" />
      ) : (
        <XCircleIcon className="mt-0.5 size-4 shrink-0 text-destructive" weight="fill" />
      )}
      <span className="font-medium">{label}</span>
      <span className="text-muted-foreground">
        {check.ok
          ? (extra ?? t("settings.test.ok"))
          : codeText(check.code ?? "unknown", check.detail)}
      </span>
    </div>
  )
}

function TestResult({ result }: { result: SettingsCheck }) {
  const { t } = useTranslation()
  const weblate = result.weblate
  const account = weblate.username
    ? t(weblate.can_write ? "settings.test.writer" : "settings.test.reader", {
        name: weblate.username,
      })
    : t("settings.test.anonymous")
  return (
    <div className="grid gap-2 rounded-lg border bg-muted/40 p-4">
      <CheckLine label={t("settings.test.llm")} check={result.llm} />
      <CheckLine label={t("settings.test.weblate")} check={weblate} extra={account} />
    </div>
  )
}

export function SettingsPage() {
  const settings = useSettings()
  if (settings.isError) return <ErrorNotice error={settings.error} />
  if (!settings.data) {
    return (
      <div className="grid gap-4">
        <Skeleton className="h-8 w-48" />
        <Skeleton className="h-64 w-full" />
      </div>
    )
  }
  return <SettingsForm view={settings.data} />
}

function SettingsForm({ view }: { view: SettingsView }) {
  const { t, i18n } = useTranslation()
  const errorText = useErrorText()
  const queryClient = useQueryClient()
  const { mode, setMode } = useTheme()
  const [draft, setDraft] = useState<AppSettings>(view.settings)

  const save = useMutation({
    mutationFn: (next: AppSettings) => api.saveSettings(next),
    onSuccess: (view) => {
      queryClient.setQueryData(keys.settings, view)
      void queryClient.invalidateQueries({ queryKey: keys.mods })
      setDraft(view.settings)
      void i18n.changeLanguage(view.settings.language)
      toast.success(t("settings.saved"))
    },
    onError: (error) => toast.error(errorText(error)),
  })
  const test = useMutation({ mutationFn: api.testSettings })
  const detect = useMutation({
    mutationFn: api.detectPaths,
    onSuccess: (found) => {
      if (!found.workshop_dir && !found.mods_dir) {
        toast.warning(t("settings.paths.not_found"))
        return
      }
      setDraft((current) => ({
          ...current,
          paths: {
            workshop_dir: found.workshop_dir ?? current.paths.workshop_dir,
            mods_dir: found.mods_dir ?? current.paths.mods_dir,
          },
        }))
    },
    onError: (error) => toast.error(errorText(error)),
  })

  const secrets = view.secrets
  const dirty = JSON.stringify(draft) !== JSON.stringify(view.settings)
  const set = (patch: Partial<AppSettings>) => setDraft({ ...draft, ...patch })
  const setModel = (patch: Partial<AppSettings["model"]>) =>
    set({ model: { ...draft.model, ...patch } })

  return (
    <div className="pb-24">
      <PageHeader title={t("settings.title")} description={t("settings.description")} />

      <Section title={t("settings.model.title")} description={t("settings.model.description")}>
        <Field id="base_url" label={t("settings.model.base_url")} hint={t("settings.model.base_url_hint")}>
          <Input
            id="base_url"
            value={draft.model.base_url}
            onChange={(event) => setModel({ base_url: event.target.value })}
          />
        </Field>
        <Field id="translation_model" label={t("settings.model.translation_model")}>
          <Input
            id="translation_model"
            placeholder="google/gemini-3.8-flash"
            value={draft.model.translation_model}
            onChange={(event) => setModel({ translation_model: event.target.value })}
          />
        </Field>
        <SecretField
          name="llm_api_key"
          label={t("settings.model.api_key")}
          hint={t("settings.model.api_key_hint")}
          isSet={secrets.llm_api_key}
        />
        <details className="group rounded-lg border px-4 py-3">
          <summary className="cursor-pointer text-sm font-medium select-none">
            {t("settings.model.advanced")}
          </summary>
          <div className="mt-4 grid gap-5">
            {(["retry_translation_model", "validation_model", "scoring_model"] as const).map(
              (key) => (
                <Field key={key} id={key} label={t(`settings.model.${key}`)} hint={t("settings.model.same_as_translation")}>
                  <Input
                    id={key}
                    value={draft.model[key]}
                    onChange={(event) => setModel({ [key]: event.target.value })}
                  />
                </Field>
              ),
            )}
            <div className="grid gap-5 sm:grid-cols-2">
              <Field id="concurrency" label={t("settings.model.concurrency")} hint={t("settings.model.concurrency_hint")}>
                <NumberInput id="concurrency" min={1} value={draft.model.concurrency} onChange={(concurrency) => setModel({ concurrency })} />
              </Field>
              <Field id="timeout_seconds" label={t("settings.model.timeout")}>
                <NumberInput id="timeout_seconds" min={1} value={draft.model.timeout_seconds} onChange={(timeout_seconds) => setModel({ timeout_seconds })} />
              </Field>
              <Field id="batch_size" label={t("settings.model.batch_size")} hint={t("settings.model.batch_size_hint")}>
                <NumberInput id="batch_size" min={1} value={draft.model.batch_size} onChange={(batch_size) => setModel({ batch_size })} />
              </Field>
              <Field id="units_per_request" label={t("settings.model.units_per_request")} hint={t("settings.model.units_per_request_hint")}>
                <NumberInput id="units_per_request" min={1} value={draft.model.units_per_request} onChange={(units_per_request) => setModel({ units_per_request })} />
              </Field>
              <Field id="auto_approve_threshold" label={t("settings.model.threshold")} hint={t("settings.model.threshold_hint")}>
                <NumberInput id="auto_approve_threshold" min={0} max={100} value={draft.model.auto_approve_threshold} onChange={(auto_approve_threshold) => setModel({ auto_approve_threshold })} />
              </Field>
            </div>
          </div>
        </details>
      </Section>

      <Section title={t("settings.weblate.title")} description={t("settings.weblate.description")}>
        <Field id="weblate_url" label={t("settings.weblate.url")}>
          <Input
            id="weblate_url"
            value={draft.weblate.url}
            onChange={(event) => set({ weblate: { ...draft.weblate, url: event.target.value } })}
          />
        </Field>
        <Field id="project_slug" label={t("settings.weblate.project")}>
          <Input
            id="project_slug"
            className="max-w-60"
            value={draft.weblate.project_slug}
            onChange={(event) =>
              set({ weblate: { ...draft.weblate, project_slug: event.target.value } })
            }
          />
        </Field>
        <SecretField
          name="weblate_token"
          label={t("settings.weblate.token")}
          hint={t("settings.weblate.token_hint")}
          isSet={secrets.weblate_token}
        />
        <Field id="store" label={t("settings.weblate.store")} hint={t(`settings.weblate.store_${draft.store}`)}>
          <Select value={draft.store} onValueChange={(store) => set({ store: store as StoreMode })}>
            <SelectTrigger id="store" className="w-60">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {(["auto", "local", "weblate"] as const).map((store) => (
                <SelectItem key={store} value={store}>
                  {t(`settings.weblate.store_name_${store}`)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </Field>
      </Section>

      <Section title={t("settings.paths.title")} description={t("settings.paths.description")}>
        <Field id="workshop_dir" label={t("settings.paths.workshop_dir")} hint={t("settings.paths.workshop_hint")}>
          <Input
            id="workshop_dir"
            className="font-mono text-xs"
            placeholder="…\steamapps\workshop\content\268500"
            value={draft.paths.workshop_dir ?? ""}
            onChange={(event) =>
              set({ paths: { ...draft.paths, workshop_dir: event.target.value || null } })
            }
          />
        </Field>
        <Field id="mods_dir" label={t("settings.paths.mods_dir")} hint={t("settings.paths.mods_hint")}>
          <Input
            id="mods_dir"
            className="font-mono text-xs"
            placeholder="…\XCom2-WarOfTheChosen\XComGame\Mods"
            value={draft.paths.mods_dir ?? ""}
            onChange={(event) =>
              set({ paths: { ...draft.paths, mods_dir: event.target.value || null } })
            }
          />
        </Field>
        <div>
          <Button variant="outline" disabled={detect.isPending} onClick={() => detect.mutate()}>
            {detect.isPending ? <CircleNotchIcon className="animate-spin" /> : <MagnifyingGlassIcon />}
            {t("settings.paths.detect")}
          </Button>
        </div>
      </Section>

      <Section title={t("settings.priority.title")} description={t("settings.priority.description")}>
        <PriorityList value={draft.priority} onChange={(priority) => set({ priority })} />
      </Section>

      <Section title={t("settings.general.title")} description={t("settings.general.description")}>
        <Field id="jobs" label={t("settings.general.jobs")} hint={t("settings.general.jobs_hint")}>
          <NumberInput id="jobs" min={1} value={draft.jobs} onChange={(jobs) => set({ jobs })} />
        </Field>
        <div className="grid gap-5 sm:grid-cols-2">
          <Field id="language" label={t("settings.general.language")}>
            <Select
              value={draft.language}
              onValueChange={(language) => set({ language: language as AppSettings["language"] })}
            >
              <SelectTrigger id="language" className="w-44">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="zh-CN">简体中文</SelectItem>
                <SelectItem value="en">English</SelectItem>
              </SelectContent>
            </Select>
          </Field>
          <Field id="theme" label={t("settings.general.theme")} hint={t("settings.general.theme_hint")}>
            <Select value={mode} onValueChange={(next) => setMode(next as ThemeMode)}>
              <SelectTrigger id="theme" className="w-44">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {(["system", "light", "dark"] as const).map((option) => (
                  <SelectItem key={option} value={option}>
                    {t(`settings.general.theme_${option}`)}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </Field>
        </div>
      </Section>

      {test.data && (
        <div className="max-w-2xl lg:ml-[calc(16rem+2.5rem)]">
          <TestResult result={test.data} />
        </div>
      )}
      {test.isError && <ErrorNotice error={test.error} className="max-w-2xl" />}

      <div className="fixed inset-x-0 bottom-0 z-20 border-t bg-background/90 backdrop-blur md:left-56">
        <div className="mx-auto flex max-w-6xl items-center justify-end gap-2 px-6 py-3">
          {dirty && <span className="mr-auto text-sm text-muted-foreground">{t("settings.unsaved")}</span>}
          <Button variant="outline" disabled={test.isPending || dirty} onClick={() => test.mutate()}>
            {test.isPending ? <CircleNotchIcon className="animate-spin" /> : <PlugsConnectedIcon />}
            {t("settings.test.run")}
          </Button>
          <Button disabled={!dirty || save.isPending} onClick={() => save.mutate(draft)}>
            {t("common.save")}
          </Button>
        </div>
      </div>
    </div>
  )
}
