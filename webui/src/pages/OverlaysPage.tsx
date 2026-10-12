import { FolderSimpleDashedIcon, PackageIcon, TrashIcon } from "@phosphor-icons/react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { useMemo, useState } from "react"
import { useTranslation } from "react-i18next"
import { toast } from "sonner"

import { EmptyState, ErrorNotice, Mono, PageHeader } from "@/components/common"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Skeleton } from "@/components/ui/skeleton"
import { useErrorText } from "@/lib/errors"
import { ApiError, api } from "@/lib/api"
import { keys } from "@/lib/queries"
import type { Page } from "@/pages"

export function OverlaysPage({ onNavigate }: { onNavigate: (page: Page) => void }) {
  const { t } = useTranslation()
  const errorText = useErrorText()
  const queryClient = useQueryClient()
  const [search, setSearch] = useState("")
  const [pending, setPending] = useState<{ workshop_id: string; title: string } | null>(null)

  const overlays = useQuery({ queryKey: keys.overlays, queryFn: api.overlays, retry: false })
  const changed = () => {
    void queryClient.invalidateQueries({ queryKey: keys.overlays })
    void queryClient.invalidateQueries({ queryKey: keys.mods })
  }
  const uninstall = useMutation({
    mutationFn: (workshopId: string) => api.uninstallOverlay(workshopId),
    onSuccess: () => {
      toast.success(t("overlays.removed"))
      changed()
    },
    onError: (error) => toast.error(errorText(error)),
  })

  const visible = useMemo(() => {
    const needle = search.trim().toLowerCase()
    return (overlays.data?.overlays ?? []).filter(
      (overlay) =>
        !needle || overlay.title.toLowerCase().includes(needle) || overlay.workshop_id.includes(needle),
    )
  }, [overlays.data, search])

  if (overlays.error instanceof ApiError && overlays.error.code === "paths_not_set") {
    return (
      <div className="grid gap-6">
        <PageHeader title={t("overlays.title")} description={t("overlays.description")} />
        <EmptyState
          icon={FolderSimpleDashedIcon}
          title={t("mods.no_paths.title")}
          description={t("mods.no_paths.description")}
          action={<Button onClick={() => onNavigate("settings")}>{t("mods.no_paths.action")}</Button>}
        />
      </div>
    )
  }

  return (
    <div className="grid gap-6">
      <PageHeader title={t("overlays.title")} description={t("overlays.description")} />
      {overlays.isError && <ErrorNotice error={overlays.error} />}

      <div className="flex flex-wrap items-center justify-between gap-3">
        <h2 className="font-medium">
          {t("overlays.installed", { count: overlays.data?.overlays.length ?? 0 })}
        </h2>
        <Input
          className="w-full sm:w-72"
          placeholder={t("mods.search")}
          value={search}
          onChange={(event) => setSearch(event.target.value)}
        />
      </div>

      {overlays.isLoading ? (
        <Skeleton className="h-60 w-full" />
      ) : visible.length === 0 ? (
        <EmptyState icon={PackageIcon} title={t("overlays.empty.title")} description={t("overlays.empty.description")} />
      ) : (
        <ul className="divide-y rounded-lg border">
          {visible.map((overlay) => (
            <li key={overlay.workshop_id} className="flex items-center gap-4 px-4 py-2.5">
              <div className="min-w-0 flex-1">
                <p className="truncate text-sm font-medium">{overlay.title}</p>
                <Mono>x2loc_{overlay.workshop_id}_zh</Mono>
              </div>
              <Button
                size="icon-sm"
                variant="ghost"
                aria-label={t("overlays.remove")}
                onClick={() => setPending(overlay)}
              >
                <TrashIcon />
              </Button>
            </li>
          ))}
        </ul>
      )}

      <AlertDialog open={pending !== null} onOpenChange={(open) => !open && setPending(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{t("overlays.confirm.title")}</AlertDialogTitle>
            <AlertDialogDescription>
              {t("overlays.confirm.description", { title: pending?.title })}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>{t("common.cancel")}</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              onClick={() => pending && uninstall.mutate(pending.workshop_id)}
            >
              {t("overlays.remove")}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  )
}
