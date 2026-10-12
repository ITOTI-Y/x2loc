// Every key the code names exists in each locale, and the locales hold the
// same keys. Keys built at runtime (t(`nav.${page}`)) are covered by parity.
import { readFileSync, readdirSync } from "node:fs"
import { join } from "node:path"

const PLURAL = /_(zero|one|two|few|many|other)$/

function flatten(tree, prefix = "") {
  return Object.entries(tree).flatMap(([key, value]) =>
    typeof value === "object" ? flatten(value, `${prefix}${key}.`) : [`${prefix}${key}`],
  )
}

function sources(dir) {
  return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const path = join(dir, entry.name)
    if (entry.isDirectory()) return sources(path)
    return /\.tsx?$/.test(entry.name) ? [readFileSync(path, "utf8")] : []
  })
}

const locales = Object.fromEntries(
  ["zh-CN", "en"].map((name) => [
    name,
    new Set(flatten(JSON.parse(readFileSync(`src/locales/${name}.json`, "utf8"))).map((k) => k.replace(PLURAL, ""))),
  ]),
)
const used = new Set(
  sources("src").flatMap((text) => [...text.matchAll(/\bt\(\s*"([a-z0-9_.]+)"/gi)].map((m) => m[1])),
)

const problems = []
for (const [name, keys] of Object.entries(locales)) {
  for (const key of used) if (!keys.has(key)) problems.push(`${name}: missing ${key}`)
  for (const [other, otherKeys] of Object.entries(locales)) {
    if (other === name) continue
    for (const key of keys) if (!otherKeys.has(key)) problems.push(`${other}: missing ${key} (in ${name})`)
  }
}
if (problems.length > 0) {
  console.error([...new Set(problems)].join("\n"))
  process.exit(1)
}
console.log(`i18n: ${used.size} keys used, locales match`)
