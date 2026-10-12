import { CloudArrowUpIcon, TrayIcon, WarningIcon } from "@phosphor-icons/react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { useMemo, useState } from "react"
import { useTranslation } from "react-i18next"
import { toast } from "sonner"

import { EmptyState, ErrorNotice, Mono, PageHeader, useErrorText } from "@/components/common"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import { Skeleton } from "@/components/ui/skeleton"
import { api, type QueuedTranslation } from "@/lib/api"
import { keys, useSettings } from "@/lib/queries"
import type { Page } from "@/pages"

function rowKey(row: QueuedTranslation): string {
  return `${row.slug}:${row.context}:${row.origin}`
}

function Conflict({ row }: { row: QueuedTranslation }) {
  const { t } = useTranslation()
  const errorText = useErrorText()
  const queryClient = useQueryClient()
  const resolve = useMutation({
    mutationFn: (keep: "local" | "upstream") => api.resolve(row, keep),
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: keys.uploads }),
    onError: (error) => toast.error(errorText(error)),
  })
  return (
    <article className="grid gap-3 rounded-lg border border-warning/50 p-4">
      <header className="flex flex-wrap items-center gap-2">
        <WarningIcon className="size-4 text-warning" weight="fill" />
        <Mono className="min-w-0 truncate">
          {row.slug} · {row.context}
        </Mono>
      </header>
      <p className="text-sm text-muted-foreground">{row.source}</p>
      <div className="grid gap-3 sm:grid-cols-2">
        <div className="grid content-start gap-2 rounded-md bg-muted/50 p-3">
          <span className="text-xs font-medium text-muted-foreground">{t("uploads.conflict.local")}</span>
          <p className="text-sm whitespace-pre-wrap">{row.target}</p>
          <Button size="sm" variant="outline" className="justify-self-start" disabled={resolve.isPending} onClick={() => resolve.mutate("local")}>
            {t("uploads.conflict.keep_local")}
          </Button>
        </div>
        <div className="grid content-start gap-2 rounded-md bg-muted/50 p-3">
          <span className="text-xs font-medium text-muted-foreground">{t("uploads.conflict.upstream")}</span>
          <p className="text-sm whitespace-pre-wrap">{row.upstream_now || t("uploads.conflict.empty")}</p>
          <Button size="sm" variant="outline" className="justify-self-start" disabled={resolve.isPending} onClick={() => resolve.mutate("upstream")}>
            {t("uploads.conflict.keep_upstream")}
          </Button>
        </div>
      </div>
    </article>
  )
}

export function UploadsPage({ onNavigate }: { onNavigate: (page: Page) => void }) {
  const { t } = useTranslation()
  const errorText = useErrorText()
  const queryClient = useQueryClient()
  const settings = useSettings()
  const uploads = useQuery({ queryKey: keys.uploads, queryFn: api.uploads })
  const [selected, setSelected] = useState<Set<string>>(new Set())

  const { pending, conflicts, components } = useMemo(() => {
    const rows = uploads.data ?? []
    const pending = rows.filter((row) => row.upload === "pending")
    const counts = new Map<string, number>()
    for (const row of pending) counts.set(row.slug, (counts.get(row.slug) ?? 0) + 1)
    return {
      pending,
      conflicts: rows.filter((row) => row.upload === "conflict"),
      components: [...counts.entries()].sort(([a], [b]) => a.localeCompare(b)),
    }
  }, [uploads.data])

  const upload = useMutation({
    mutationFn: () => api.upload(selected.size > 0 ? [...selected] : null),
    onSuccess: (report) => {
      setSelected(new Set())
      toast.success(t("uploads.done", { ...report }))
      void queryClient.invalidateQueries({ queryKey: keys.uploads })
      void queryClient.invalidateQueries({ queryKey: keys.mods })
    },
    onError: (error) => toast.error(errorText(error)),
  })

  const hasToken = settings.data?.secrets.weblate_token ?? false
  const toggle = (slug: string, on: boolean) => {
    const next = new Set(selected)
    if (on) next.add(slug)
    else next.delete(slug)
    setSelected(next)
  }

  return (
    <div className="grid gap-6">
      <PageHeader
        title={t("uploads.title")}
        description={t("uploads.description")}
        actions={
          <Button disabled={!hasToken || pending.length === 0 || upload.isPending} onClick={() => upload.mutate()}>
            <CloudArrowUpIcon />
            {selected.size > 0 ? t("uploads.upload_selected", { count: selected.size }) : t("uploads.upload_all")}
          </Button>
        }
      />
      {!hasToken && (
        <p className="text-sm text-muted-foreground">
          {t("uploads.no_token")}{" "}
          <button type="button" className="text-primary hover:underline" onClick={() => onNavigate("settings")}>
            {t("uploads.go_settings")}
          </button>
        </p>
      )}
      {uploads.isError && <ErrorNotice error={uploads.error} />}

      {uploads.isLoading ? (
        <Skeleton className="h-48 w-full" />
      ) : pending.length === 0 && conflicts.length === 0 ? (
        <EmptyState icon={TrayIcon} title={t("uploads.empty.title")} description={t("uploads.empty.description")} />
      ) : (
        <>
          {conflicts.length > 0 && (
            <section className="grid gap-3">
              <h2 className="font-medium">{t("uploads.conflicts", { count: conflicts.length })}</h2>
              <p className="text-sm text-muted-foreground">{t("uploads.conflicts_hint")}</p>
              {conflicts.map((row) => (
                <Conflict key={rowKey(row)} row={row} />
              ))}
            </section>
          )}
          {components.length > 0 && (
            <section className="grid gap-3">
              <h2 className="font-medium">{t("uploads.pending", { count: pending.length })}</h2>
              <ul className="divide-y rounded-lg border">
                {components.map(([slug, count]) => (
                  <li key={slug} className="flex items-center gap-3 px-4 py-2.5">
                    <Checkbox
                      id={`upload-${slug}`}
                      checked={selected.has(slug)}
                      onCheckedChange={(on) => toggle(slug, on === true)}
                    />
                    <label htmlFor={`upload-${slug}`} className="min-w-0 flex-1 truncate font-mono text-sm">
                      {slug}
                    </label>
                    <Badge variant="secondary" className="tabular-nums">
                      {t("uploads.units", { count })}
                    </Badge>
                  </li>
                ))}
              </ul>
            </section>
          )}
        </>
      )}
    </div>
  )
}
