import { useEffect, useState, type ReactNode } from "react"

import { ThemeContext, type ThemeMode } from "@/lib/theme-context"

const STORAGE_KEY = "x2loc.theme"
const DARK_QUERY = "(prefers-color-scheme: dark)"

function storedMode(): ThemeMode {
  try {
    const value = localStorage.getItem(STORAGE_KEY)
    return value === "light" || value === "dark" ? value : "system"
  } catch {
    return "system"
  }
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [mode, setModeState] = useState<ThemeMode>(storedMode)
  const [systemDark, setSystemDark] = useState(() => matchMedia(DARK_QUERY).matches)

  useEffect(() => {
    const query = matchMedia(DARK_QUERY)
    const listener = (event: MediaQueryListEvent) => setSystemDark(event.matches)
    query.addEventListener("change", listener)
    return () => query.removeEventListener("change", listener)
  }, [])

  const resolved = mode === "system" ? (systemDark ? "dark" : "light") : mode

  useEffect(() => {
    document.documentElement.classList.toggle("dark", resolved === "dark")
  }, [resolved])

  const setMode = (next: ThemeMode) => {
    setModeState(next)
    try {
      localStorage.setItem(STORAGE_KEY, next)
    } catch {
      // The choice still applies to this visit.
    }
  }

  return (
    <ThemeContext.Provider value={{ mode, resolved, setMode }}>{children}</ThemeContext.Provider>
  )
}
