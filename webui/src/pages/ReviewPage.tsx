import { ArrowSquareOutIcon, CheckIcon, SealCheckIcon } from "@phosphor-icons/react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { useMemo, useState } from "react"
import { useTranslation } from "react-i18next"
import { toast } from "sonner"

import { EmptyState, ErrorNotice, Mono, PageHeader, useErrorText } from "@/components/common"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Skeleton } from "@/components/ui/skeleton"
import { Textarea } from "@/components/ui/textarea"
import { api, type ReviewItem } from "@/lib/api"
import { keys } from "@/lib/queries"

const PAGE_SIZE = 40

function itemKey(item: ReviewItem): string {
  return `${item.where}:${item.slug}:${item.context}`
}

function ReviewCard({ item }: { item: ReviewItem }) {
  const { t } = useTranslation()
  const errorText = useErrorText()
  const queryClient = useQueryClient()
  const [target, setTarget] = useState(item.target)
  const save = useMutation({
    mutationFn: () => api.saveReview(item, target),
    onSuccess: () => {
      toast.success(t("review.saved"))
      void queryClient.invalidateQueries({ queryKey: keys.review })
      void queryClient.invalidateQueries({ queryKey: keys.uploads })
    },
    onError: (error) => toast.error(errorText(error)),
  })
  return (
    <article className="grid gap-3 rounded-lg border p-4">
      <header className="flex flex-wrap items-center gap-2">
        <Badge variant={item.where === "weblate" ? "secondary" : "outline"}>
          {t(`review.where.${item.where}`)}
        </Badge>
        <Mono className="min-w-0 truncate">
          {item.slug} · {item.context}
        </Mono>
        {item.web_url && (
          <a
            href={item.web_url}
            target="_blank"
            rel="noreferrer"
            className="ml-auto inline-flex items-center gap-1 text-xs text-primary hover:underline"
          >
            {t("review.open_weblate")}
            <ArrowSquareOutIcon />
          </a>
        )}
      </header>
      <div className="grid gap-3 lg:grid-cols-2">
        <div className="grid content-start gap-1">
          <span className="text-xs font-medium text-muted-foreground">{t("review.source")}</span>
          <p className="rounded-md bg-muted/50 px-3 py-2 text-sm whitespace-pre-wrap">{item.source}</p>
        </div>
        <div className="grid content-start gap-1">
          <span className="text-xs font-medium text-muted-foreground">{t("review.target")}</span>
          <Textarea
            aria-label={t("review.target")}
            value={target}
            className="min-h-20"
            onChange={(event) => setTarget(event.target.value)}
          />
        </div>
      </div>
      {item.comment && (
        <p className="border-l-2 border-warning pl-3 text-sm text-muted-foreground whitespace-pre-wrap">
          {item.comment}
        </p>
      )}
      <div className="flex justify-end">
        <Button size="sm" disabled={!target.trim() || save.isPending} onClick={() => save.mutate()}>
          <CheckIcon />
          {target === item.target ? t("review.approve") : t("review.save")}
        </Button>
      </div>
    </article>
  )
}

export function ReviewPage() {
  const { t } = useTranslation()
  const [search, setSearch] = useState("")
  const [limit, setLimit] = useState(PAGE_SIZE)
  const review = useQuery({ queryKey: keys.review, queryFn: api.review })

  const visible = useMemo(() => {
    const needle = search.trim().toLowerCase()
    return (review.data ?? []).filter(
      (item) =>
        !needle ||
        item.slug.toLowerCase().includes(needle) ||
        item.source.toLowerCase().includes(needle) ||
        item.target.toLowerCase().includes(needle),
    )
  }, [review.data, search])

  return (
    <div className="grid gap-6">
      <PageHeader title={t("review.title")} description={t("review.description")} />
      {review.isError && <ErrorNotice error={review.error} />}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h2 className="font-medium">{t("review.count", { count: visible.length })}</h2>
        <Input
          className="w-full sm:w-72"
          placeholder={t("review.search")}
          value={search}
          onChange={(event) => setSearch(event.target.value)}
        />
      </div>
      {review.isLoading ? (
        <div className="grid gap-3">
          <Skeleton className="h-36 w-full" />
          <Skeleton className="h-36 w-full" />
        </div>
      ) : visible.length === 0 ? (
        <EmptyState icon={SealCheckIcon} title={t("review.empty.title")} description={t("review.empty.description")} />
      ) : (
        <div className="grid gap-3">
          {visible.slice(0, limit).map((item) => (
            <ReviewCard key={itemKey(item)} item={item} />
          ))}
          {visible.length > limit && (
            <Button variant="outline" className="justify-self-center" onClick={() => setLimit(limit + PAGE_SIZE)}>
              {t("common.show_more", { count: visible.length - limit })}
            </Button>
          )}
        </div>
      )}
    </div>
  )
}
