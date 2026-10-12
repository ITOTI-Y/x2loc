import {
  CircleNotchIcon,
  CloudArrowUpIcon,
  GearSixIcon,
  ListChecksIcon,
  NotePencilIcon,
  PackageIcon,
  PlugsIcon,
  PuzzlePieceIcon,
  type Icon,
} from "@phosphor-icons/react"
import { useQuery } from "@tanstack/react-query"
import { useEffect, useState } from "react"
import { useTranslation } from "react-i18next"

import { EmptyState } from "@/components/common"
import { api, hasSession } from "@/lib/api"
import { keys, useActiveRun, useSettings } from "@/lib/queries"
import { cn } from "@/lib/utils"
import { PAGES, type Page } from "@/pages"
import { ModsPage } from "@/pages/ModsPage"
import { OverlaysPage } from "@/pages/OverlaysPage"
import { ReviewPage } from "@/pages/ReviewPage"
import { RunsPage } from "@/pages/RunsPage"
import { SettingsPage } from "@/pages/SettingsPage"
import { UploadsPage } from "@/pages/UploadsPage"

const PAGE_KEY = "x2loc.page"

const ICONS: Record<Page, Icon> = {
  mods: PuzzlePieceIcon,
  runs: ListChecksIcon,
  overlays: PackageIcon,
  review: NotePencilIcon,
  uploads: CloudArrowUpIcon,
  settings: GearSixIcon,
}

function storedPage(): Page | null {
  try {
    const value = localStorage.getItem(PAGE_KEY)
    return PAGES.find((page) => page === value) ?? null
  } catch {
    return null
  }
}

function NavItem({
  page,
  current,
  onSelect,
  trailing,
}: {
  page: Page
  current: boolean
  onSelect: () => void
  trailing?: React.ReactNode
}) {
  const { t } = useTranslation()
  const IconComponent = ICONS[page]
  return (
    <button
      type="button"
      onClick={onSelect}
      aria-current={current ? "page" : undefined}
      className={cn(
        "flex shrink-0 items-center gap-2.5 rounded-md px-2.5 py-1.5 text-sm transition-colors md:w-full",
        current
          ? "bg-primary/10 font-medium text-primary"
          : "text-muted-foreground hover:bg-muted hover:text-foreground",
      )}
    >
      <IconComponent className="size-4" weight={current ? "fill" : "regular"} />
      <span>{t(`nav.${page}`)}</span>
      {trailing && <span className="ml-auto">{trailing}</span>}
    </button>
  )
}

function Shell() {
  const { t, i18n } = useTranslation()
  const settings = useSettings()
  const activeRun = useActiveRun()
  const uploads = useQuery({ queryKey: keys.uploads, queryFn: api.uploads })
  const [page, setPageState] = useState<Page | null>(storedPage)

  // Paths are all a run needs; a model and a key only matter once a mod has
  // text Weblate lacks, and the run then says so.
  const configured =
    settings.data !== undefined &&
    settings.data.settings.paths.workshop_dir !== null &&
    settings.data.settings.paths.mods_dir !== null

  useEffect(() => {
    if (settings.data) void i18n.changeLanguage(settings.data.settings.language)
  }, [settings.data, i18n])

  // A first start goes to the settings; later ones resume where they left.
  // Decided once, so saving the paths does not pull the player off the page.
  if (page === null && settings.data) setPageState(configured ? "mods" : "settings")
  // Null until the settings load; nothing is shown before the page is known.
  const current = page

  const setPage = (next: Page) => {
    setPageState(next)
    window.scrollTo({ top: 0 })
    try {
      localStorage.setItem(PAGE_KEY, next)
    } catch {
      // Only the next visit's starting page is lost.
    }
  }

  const pendingUploads = uploads.data?.filter((row) => row.upload !== "uploaded").length ?? 0

  return (
    <div className="min-h-dvh bg-background md:grid md:grid-cols-[14rem_1fr]">
      <aside className="sticky top-0 z-30 border-b bg-background md:h-dvh md:border-r md:border-b-0">
        <div className="flex items-center gap-2 px-4 pt-4 pb-3 md:px-5 md:pt-6 md:pb-6">
          <span className="font-mono text-sm font-semibold tracking-tight">x2loc</span>
          <span className="text-xs text-muted-foreground">{t("app.tagline")}</span>
        </div>
        <nav className="flex gap-1 overflow-x-auto px-3 pb-3 md:grid md:overflow-visible md:pb-0">
          {PAGES.map((item) => (
            <NavItem
              key={item}
              page={item}
              current={item === current}
              onSelect={() => setPage(item)}
              trailing={
                item === "runs" && activeRun ? (
                  <CircleNotchIcon className="size-3.5 animate-spin text-primary" />
                ) : item === "uploads" && pendingUploads > 0 ? (
                  <span className="rounded-full bg-muted px-1.5 text-xs tabular-nums">{pendingUploads}</span>
                ) : null
              }
            />
          ))}
        </nav>
      </aside>
      <main className="mx-auto w-full max-w-6xl px-4 py-6 md:px-8 md:py-8">
        {settings.data && !configured && current !== "settings" && (
          <button
            type="button"
            onClick={() => setPage("settings")}
            className="mb-6 w-full rounded-lg border border-primary/30 bg-primary/5 px-4 py-3 text-left text-sm"
          >
            <span className="font-medium text-primary">{t("app.setup.title")}</span>{" "}
            <span className="text-muted-foreground">{t("app.setup.description")}</span>
          </button>
        )}
        {current === "mods" && <ModsPage onNavigate={setPage} />}
        {current === "runs" && <RunsPage onNavigate={setPage} />}
        {current === "overlays" && <OverlaysPage onNavigate={setPage} />}
        {current === "review" && <ReviewPage />}
        {current === "uploads" && <UploadsPage onNavigate={setPage} />}
        {current === "settings" && <SettingsPage />}
      </main>
    </div>
  )
}

export default function App() {
  const { t } = useTranslation()
  if (!hasSession()) {
    return (
      <main className="mx-auto grid min-h-dvh max-w-lg place-items-center px-4">
        <EmptyState icon={PlugsIcon} title={t("app.no_session.title")} description={t("app.no_session.description")} />
      </main>
    )
  }
  return <Shell />
}
