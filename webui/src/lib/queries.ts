import { useQuery } from "@tanstack/react-query"

import { api } from "@/lib/api"

export const keys = {
  settings: ["settings"],
  mods: ["mods"],
  runs: ["runs"],
  overlays: ["overlays"],
  review: ["review"],
  uploads: ["uploads"],
} as const

export function useSettings() {
  return useQuery({ queryKey: keys.settings, queryFn: api.settings })
}

export function useRuns() {
  return useQuery({
    queryKey: keys.runs,
    queryFn: api.runs,
    // Follow a running translation closely, otherwise stay quiet.
    refetchInterval: (query) =>
      query.state.data?.some((run) => run.status === "running") ? 1500 : false,
  })
}

export function useActiveRun() {
  const runs = useRuns()
  return runs.data?.find((run) => run.status === "running") ?? null
}
