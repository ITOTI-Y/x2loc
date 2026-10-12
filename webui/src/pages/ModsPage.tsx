import {
  ArrowClockwiseIcon,
  FolderSimpleDashedIcon,
  MagnifyingGlassIcon,
  PlayIcon,
  PuzzlePieceIcon,
} from "@phosphor-icons/react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { useMemo, useState } from "react"
import { useTranslation } from "react-i18next"
import { toast } from "sonner"

import { EmptyState, ErrorNotice, Mono, PageHeader } from "@/components/common"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Skeleton } from "@/components/ui/skeleton"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { useErrorText } from "@/lib/errors"
import { ApiError, api, type ModView } from "@/lib/api"
import { keys, useActiveRun, useSettings } from "@/lib/queries"
import type { Page } from "@/pages"
import { cn } from "@/lib/utils"

type Filter = "all" | "incomplete" | "installed" | "problems"

function complete(mod: ModView): boolean {
  const coverage = mod.coverage
  return (
    coverage !== null &&
    coverage.components_found === mod.slugs.length &&
    coverage.translated === coverage.total
  )
}

function matches(mod: ModView, filter: Filter): boolean {
  switch (filter) {
    case "all":
      return true
    case "incomplete":
      return mod.problem === null && (!complete(mod) || !mod.overlay_installed)
    case "installed":
      return mod.overlay_installed
    case "problems":
      return mod.problem !== null
  }
}

function CoverageCell({ mod, loading }: { mod: ModView; loading: boolean }) {
  const { t } = useTranslation()
  if (mod.problem !== null) {
    return <span className="text-sm text-muted-foreground">—</span>
  }
  if (mod.coverage === null) {
    return loading ? <Skeleton className="h-2 w-40" /> : <span className="text-sm text-muted-foreground">—</span>
  }
  const { translated, total, components_found } = mod.coverage
  const missing = mod.slugs.length - components_found
  const ratio = total === 0 ? 0 : translated / total
  return (
    <div className="grid w-44 gap-1">
      <div className="h-1.5 overflow-hidden rounded-full bg-muted">
        <div
          className={cn("h-full rounded-full", ratio === 1 && missing === 0 ? "bg-success" : "bg-primary")}
          style={{ width: `${Math.round(ratio * 100)}%` }}
        />
      </div>
      <span className="text-xs text-muted-foreground tabular-nums">
        {components_found === 0
          ? t("mods.coverage.not_synced")
          : [
              total === 0 ? t("mods.coverage.none") : `${translated} / ${total}`,
              ...(missing > 0 ? [t("mods.coverage.unsynced", { count: missing })] : []),
            ].join(" · ")}
      </span>
    </div>
  )
}

export function ModsPage({ onNavigate }: { onNavigate: (page: Page) => void }) {
  const { t } = useTranslation()
  const errorText = useErrorText()
  const queryClient = useQueryClient()
  const settings = useSettings()
  const activeRun = useActiveRun()
  const [filter, setFilter] = useState<Filter>("all")
  const [search, setSearch] = useState("")
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [uploadAfter, setUploadAfter] = useState(false)

  const mods = useQuery({
    queryKey: keys.mods,
    queryFn: () => api.mods(),
    refetchInterval: (query) => (query.state.data?.coverage_state === "loading" ? 2000 : false),
    retry: false,
  })
  const refresh = useMutation({
    mutationFn: () => api.mods(true),
    onSuccess: (data) => queryClient.setQueryData(keys.mods, data),
    onError: (error) => toast.error(errorText(error)),
  })
  const start = useMutation({
    mutationFn: () => api.startRun([...selected], uploadAfter),
    onSuccess: () => {
      setSelected(new Set())
      void queryClient.invalidateQueries({ queryKey: keys.runs })
      onNavigate("runs")
    },
    onError: (error) => toast.error(errorText(error)),
  })

  const visible = useMemo(() => {
    const needle = search.trim().toLowerCase()
    return (mods.data?.mods ?? []).filter(
      (mod) =>
        matches(mod, filter) &&
        (!needle || mod.title.toLowerCase().includes(needle) || mod.workshop_id.includes(needle)),
    )
  }, [mods.data, filter, search])

  if (mods.error instanceof ApiError && mods.error.code === "paths_not_set") {
    return (
      <div className="grid gap-6">
        <PageHeader title={t("mods.title")} description={t("mods.description")} />
        <EmptyState
          icon={FolderSimpleDashedIcon}
          title={t("mods.no_paths.title")}
          description={t("mods.no_paths.description")}
          action={<Button onClick={() => onNavigate("settings")}>{t("mods.no_paths.action")}</Button>}
        />
      </div>
    )
  }

  const selectable = visible.filter((mod) => mod.problem === null)
  const allSelected = selectable.length > 0 && selectable.every((mod) => selected.has(mod.workshop_id))
  const toggle = (id: string, on: boolean) => {
    const next = new Set(selected)
    if (on) next.add(id)
    else next.delete(id)
    setSelected(next)
  }
  const hasToken = settings.data?.secrets.weblate_token ?? false
  const loadingCoverage = mods.data?.coverage_state === "loading"

  return (
    <div className="grid gap-6">
      <PageHeader
        title={t("mods.title")}
        description={t("mods.description")}
        actions={
          <Button variant="outline" disabled={refresh.isPending || mods.isLoading} onClick={() => refresh.mutate()}>
            <ArrowClockwiseIcon className={cn(refresh.isPending && "animate-spin")} />
            {t("mods.rescan")}
          </Button>
        }
      />

      {mods.isError && <ErrorNotice error={mods.error} />}
      {mods.data?.coverage_state === "failed" && (
        <p className="text-sm text-warning">{t("mods.coverage.failed")}</p>
      )}

      <div className="flex flex-wrap items-center gap-3">
        <Tabs value={filter} onValueChange={(value) => setFilter(value as Filter)}>
          <TabsList>
            {(["all", "incomplete", "installed", "problems"] as const).map((value) => (
              <TabsTrigger key={value} value={value}>
                {t(`mods.filter.${value}`)}
              </TabsTrigger>
            ))}
          </TabsList>
        </Tabs>
        <div className="relative ml-auto w-full sm:w-72">
          <MagnifyingGlassIcon className="pointer-events-none absolute top-1/2 left-2.5 size-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            className="pl-8"
            placeholder={t("mods.search")}
            value={search}
            onChange={(event) => setSearch(event.target.value)}
          />
        </div>
      </div>

      {mods.isLoading ? (
        <div className="grid gap-2">
          {Array.from({ length: 8 }, (_, index) => (
            <Skeleton key={index} className="h-11 w-full" />
          ))}
          <p className="text-sm text-muted-foreground">{t("mods.scanning")}</p>
        </div>
      ) : visible.length === 0 ? (
        <EmptyState icon={PuzzlePieceIcon} title={t("mods.empty")} />
      ) : (
        <div className="overflow-hidden rounded-lg border">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-10">
                  <Checkbox
                    aria-label={t("mods.select_all")}
                    checked={allSelected}
                    onCheckedChange={(on) =>
                      setSelected(on ? new Set(selectable.map((mod) => mod.workshop_id)) : new Set())
                    }
                  />
                </TableHead>
                <TableHead className="w-full">{t("mods.column.mod")}</TableHead>
                <TableHead className="hidden sm:table-cell">{t("mods.column.coverage")}</TableHead>
                <TableHead className="text-right">{t("mods.column.overlay")}</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {visible.map((mod) => (
                <TableRow
                  key={mod.workshop_id}
                  data-state={selected.has(mod.workshop_id) ? "selected" : undefined}
                  className={cn(mod.problem !== null && "text-muted-foreground")}
                >
                  <TableCell>
                    <Checkbox
                      aria-label={mod.title}
                      disabled={mod.problem !== null}
                      checked={selected.has(mod.workshop_id)}
                      onCheckedChange={(on) => toggle(mod.workshop_id, on === true)}
                    />
                  </TableCell>
                  <TableCell className="max-w-0 whitespace-normal">
                    <p className="truncate font-medium">{mod.title}</p>
                    <Mono>{mod.workshop_id}</Mono>
                    <div className="mt-2 sm:hidden">
                      <CoverageCell mod={mod} loading={loadingCoverage} />
                    </div>
                  </TableCell>
                  <TableCell className="hidden sm:table-cell">
                    <CoverageCell mod={mod} loading={loadingCoverage} />
                  </TableCell>
                  <TableCell className="text-right">
                    {mod.problem !== null ? (
                      <Badge variant="outline">{t(`mods.problem.${mod.problem}`)}</Badge>
                    ) : mod.overlay_installed ? (
                      <Badge variant="secondary">{t("mods.installed")}</Badge>
                    ) : (
                      <span className="text-sm text-muted-foreground">{t("mods.not_installed")}</span>
                    )}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      )}

      {selected.size > 0 && (
        <div className="sticky bottom-4 z-10 mx-auto flex w-full max-w-3xl flex-wrap items-center gap-4 rounded-xl border bg-background/95 px-4 py-3 shadow-lg backdrop-blur">
          <span className="text-sm font-medium">{t("mods.selected", { count: selected.size })}</span>
          {hasToken && (
            <div className="flex items-center gap-2">
              <Checkbox
                id="upload_after"
                checked={uploadAfter}
                onCheckedChange={(on) => setUploadAfter(on === true)}
              />
              <Label htmlFor="upload_after" className="text-sm font-normal">
                {t("mods.upload_after")}
              </Label>
            </div>
          )}
          <div className="ml-auto flex gap-2">
            <Button variant="ghost" onClick={() => setSelected(new Set())}>
              {t("common.clear")}
            </Button>
            <Button disabled={start.isPending || activeRun !== null} onClick={() => start.mutate()}>
              <PlayIcon weight="fill" />
              {activeRun !== null ? t("mods.run_busy") : t("mods.translate")}
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}
