import { createContext, useContext } from "react"

export type ThemeMode = "system" | "light" | "dark"

export interface ThemeState {
  mode: ThemeMode
  resolved: "light" | "dark"
  setMode: (mode: ThemeMode) => void
}

export const ThemeContext = createContext<ThemeState | null>(null)

export function useTheme(): ThemeState {
  const state = useContext(ThemeContext)
  if (state === null) throw new Error("useTheme outside ThemeProvider")
  return state
}
