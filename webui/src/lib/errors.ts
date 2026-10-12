import { useTranslation } from "react-i18next"

import { ApiError } from "@/lib/api"

export function useCodeText() {
  const { t } = useTranslation()
  return (code: string, detail = "") => {
    const text = t(`errors.${code}`, { defaultValue: t("errors.unknown", { code }) })
    return detail ? `${text} (${detail})` : text
  }
}

export function useErrorText() {
  const codeText = useCodeText()
  return (error: unknown) =>
    error instanceof ApiError ? codeText(error.code, error.detail) : codeText("unknown", String(error))
}
