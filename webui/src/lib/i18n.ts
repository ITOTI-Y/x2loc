import i18n from "i18next"
import { initReactI18next } from "react-i18next"

import en from "@/locales/en.json"
import zhCN from "@/locales/zh-CN.json"

export const LANGUAGES = ["zh-CN", "en"] as const

// The saved setting takes over once loaded; until then the browser decides.
const initial = navigator.language.toLowerCase().startsWith("zh") ? "zh-CN" : "en"

void i18n.use(initReactI18next).init({
  resources: { "zh-CN": { translation: zhCN }, en: { translation: en } },
  lng: initial,
  fallbackLng: "en",
  interpolation: { escapeValue: false },
})

document.documentElement.lang = initial
i18n.on("languageChanged", (language) => {
  document.documentElement.lang = language
})

export default i18n
