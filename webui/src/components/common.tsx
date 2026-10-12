import type { Icon } from "@phosphor-icons/react"
import { WarningCircleIcon } from "@phosphor-icons/react"
import type { ReactNode } from "react"
import { useTranslation } from "react-i18next"

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { useErrorText } from "@/lib/errors"
import { cn } from "@/lib/utils"

export function PageHeader({
  title,
  description,
  actions,
}: {
  title: string
  description?: string
  actions?: ReactNode
}) {
  return (
    <header className="flex flex-wrap items-end justify-between gap-4 border-b pb-5">
      <div className="min-w-0">
        <h1 className="text-xl font-semibold tracking-tight">{title}</h1>
        {description && (
          <p className="mt-1 max-w-[70ch] text-sm text-muted-foreground">{description}</p>
        )}
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </header>
  )
}

export function ErrorNotice({ error, className }: { error: unknown; className?: string }) {
  const { t } = useTranslation()
  const text = useErrorText()
  return (
    <Alert variant="destructive" className={className}>
      <WarningCircleIcon />
      <AlertTitle>{t("common.error")}</AlertTitle>
      <AlertDescription>{text(error)}</AlertDescription>
    </Alert>
  )
}

export function EmptyState({
  icon: IconComponent,
  title,
  description,
  action,
  className,
}: {
  icon: Icon
  title: string
  description?: string
  action?: ReactNode
  className?: string
}) {
  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center gap-3 rounded-lg border border-dashed px-6 py-14 text-center",
        className,
      )}
    >
      <IconComponent className="size-8 text-muted-foreground" weight="duotone" />
      <div>
        <p className="font-medium">{title}</p>
        {description && (
          <p className="mx-auto mt-1 max-w-[50ch] text-sm text-muted-foreground">{description}</p>
        )}
      </div>
      {action}
    </div>
  )
}

export function Mono({ children, className }: { children: ReactNode; className?: string }) {
  return <span className={cn("font-mono text-xs text-muted-foreground", className)}>{children}</span>
}
