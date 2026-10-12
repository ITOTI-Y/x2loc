export const PAGES = ["mods", "runs", "overlays", "review", "uploads", "settings"] as const

export type Page = (typeof PAGES)[number]
