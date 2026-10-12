import {
  CheckCircleIcon,
  CircleNotchIcon,
  ClockIcon,
  ListChecksIcon,
  ProhibitIcon,
  StopIcon,
  XCircleIcon,
} from "@phosphor-icons/react"
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { useTranslation } from "react-i18next"
import { toast } from "sonner"

import { EmptyState, ErrorNotice, Mono, PageHeader } from "@/components/common"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Skeleton } from "@/components/ui/skeleton"
import { useCodeText, useErrorText } from "@/lib/errors"
import { api, type Job, type JobStage, type JobStatus, type Run } from "@/lib/api"
import { keys, useRuns } from "@/lib/queries"
import type { Page } from "@/pages"
import { cn } from "@/lib/utils"

const STAGES: JobStage[] = [
  "downloading",
  "discovering",
  "syncing_weblate",
  "translating",
  "writing",
  "extracting_terms",
  "packaging",
]

function StatusIcon({ status }: { status: JobStatus | Run["status"] }) {
  switch (status) {
    case "queued":
      return <ClockIcon className="size-4 text-muted-foreground" />
    case "running":
      return <CircleNotchIcon className="size-4 animate-spin text-primary" />
    case "succeeded":
      return <CheckCircleIcon className="size-4 text-success" weight="fill" />
    case "cancelled":
      return <ProhibitIcon className="size-4 text-muted-foreground" />
    case "failed":
      return <XCircleIcon className="size-4 text-destructive" weight="fill" />
  }
}

function JobRow({ job, live }: { job: Job; live: boolean }) {
  const { t } = useTranslation()
  const codeText = useCodeText()
  const errorText = useErrorText()
  const queryClient = useQueryClient()
  const cancel = useMutation({
    mutationFn: () => api.cancelJob(job.id),
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: keys.runs }),
    onError: (error) => toast.error(errorText(error)),
  })
  const { units_total, units_translated, files_total, files_completed } = job.progress
  // Progress is reported per stage, so the bar follows the stages.
  const ratio =
    job.status === "succeeded"
      ? 1
      : job.stage === null
        ? 0
        : (STAGES.indexOf(job.stage) + 0.5) / STAGES.length
  const active = job.status === "queued" || job.status === "running"
  return (
    <li className="grid gap-2 px-4 py-3 sm:grid-cols-[1fr_14rem_auto] sm:items-center sm:gap-6">
      <div className="flex min-w-0 items-center gap-3">
        <StatusIcon status={job.status} />
        <div className="min-w-0">
          <p className="truncate text-sm font-medium">{job.title}</p>
          <p className="text-xs text-muted-foreground">
            {job.status === "failed" && job.error_code
              ? codeText(job.error_code)
              : job.stage && active
                ? t(`stages.${job.stage}`)
                : t(`status.${job.status}`)}
          </p>
        </div>
      </div>
      <div className="grid gap-1">
        <div className="h-1.5 overflow-hidden rounded-full bg-muted">
          <div
            className={cn(
              "h-full rounded-full transition-[width] duration-500",
              job.status === "failed" ? "bg-destructive" : job.status === "succeeded" ? "bg-success" : "bg-primary",
            )}
            style={{ width: `${Math.round(ratio * 100)}%` }}
          />
        </div>
        <span className="text-xs text-muted-foreground tabular-nums">
          {t("runs.progress", {
            units: units_translated,
            unitsTotal: units_total,
            files: files_completed,
            filesTotal: files_total,
          })}
        </span>
      </div>
      <div className="flex justify-end">
        {live && active && (
          <Button size="sm" variant="ghost" disabled={cancel.isPending} onClick={() => cancel.mutate()}>
            {t("common.cancel")}
          </Button>
        )}
      </div>
    </li>
  )
}

function RunCard({ run }: { run: Run }) {
  const { t, i18n } = useTranslation()
  const codeText = useCodeText()
  const errorText = useErrorText()
  const queryClient = useQueryClient()
  const cancel = useMutation({
    mutationFn: () => api.cancelRun(run.id),
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: keys.runs }),
    onError: (error) => toast.error(errorText(error)),
  })
  const live = run.status === "running"
  const jobs = Object.values(run.jobs)
  const skipped = Object.entries(run.skipped)
  const started = new Date(run.started_at).toLocaleString(i18n.language)
  return (
    <article className={cn("overflow-hidden rounded-lg border", live && "ring-1 ring-primary/40")}>
      <header className="flex flex-wrap items-center gap-3 border-b bg-muted/40 px-4 py-3">
        <StatusIcon status={run.status} />
        <span className="text-sm font-medium">{t(`runs.status.${run.status}`)}</span>
        <Mono>{started}</Mono>
        <Badge variant="outline">{t("runs.mods", { count: run.workshop_ids.length })}</Badge>
        {Object.keys(run.installed).length > 0 && (
          <Badge variant="secondary">{t("runs.installed", { count: Object.keys(run.installed).length })}</Badge>
        )}
        {run.upload && (
          <Badge variant="secondary">
            {t("runs.uploaded", { uploaded: run.upload.uploaded, conflicts: run.upload.conflicts })}
          </Badge>
        )}
        {live && (
          <Button size="sm" variant="outline" className="ml-auto" disabled={cancel.isPending} onClick={() => cancel.mutate()}>
            <StopIcon weight="fill" />
            {t("runs.cancel")}
          </Button>
        )}
      </header>
      {run.error_code && <p className="px-4 pt-3 text-sm text-destructive">{codeText(run.error_code)}</p>}
      {jobs.length === 0 && live ? (
        <p className="px-4 py-4 text-sm text-muted-foreground">{t("runs.preparing")}</p>
      ) : (
        <ul className="divide-y">
          {jobs.map((job) => (
            <JobRow key={job.id} job={job} live={live} />
          ))}
        </ul>
      )}
      {skipped.length > 0 && (
        <div className="border-t px-4 py-3 text-sm">
          <p className="font-medium">{t("runs.skipped")}</p>
          <ul className="mt-1 grid gap-0.5 text-muted-foreground">
            {skipped.map(([id, reason]) => (
              <li key={id}>
                <Mono>{id}</Mono> {reason}
              </li>
            ))}
          </ul>
        </div>
      )}
    </article>
  )
}

export function RunsPage({ onNavigate }: { onNavigate: (page: Page) => void }) {
  const { t } = useTranslation()
  const runs = useRuns()
  return (
    <div className="grid gap-6">
      <PageHeader title={t("runs.title")} description={t("runs.description")} />
      {runs.isError && <ErrorNotice error={runs.error} />}
      {runs.isLoading ? (
        <Skeleton className="h-40 w-full" />
      ) : runs.data && runs.data.length > 0 ? (
        <div className="grid gap-4">
          {runs.data.map((run) => (
            <RunCard key={run.id} run={run} />
          ))}
        </div>
      ) : (
        <EmptyState
          icon={ListChecksIcon}
          title={t("runs.empty.title")}
          description={t("runs.empty.description")}
          action={<Button onClick={() => onNavigate("mods")}>{t("runs.empty.action")}</Button>}
        />
      )}
    </div>
  )
}
