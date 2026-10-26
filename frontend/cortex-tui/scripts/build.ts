import { mkdirSync } from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"
import solidPlugin from "@opentui/solid/bun-plugin"

const scriptDir = path.dirname(fileURLToPath(import.meta.url))
const projectDir = path.resolve(scriptDir, "..")
const repoRoot = path.resolve(projectDir, "..", "..")
const defaultOutDir = path.resolve(repoRoot, "cortex", "ui_runtime", "bin")
const outBinary = process.env.CORTEX_TUI_OUTFILE
  ? path.resolve(process.cwd(), process.env.CORTEX_TUI_OUTFILE)
  : path.resolve(defaultOutDir, "cortex-tui")

// The sidecar is built for the machine it is built on: OpenTUI's native core
// comes from the platform package `bun install` fetched for this host.
// Keep in step with cortex/ui_runtime/platforms.py (tests/test_packaging.py).
const HOST_TARGETS: Record<string, string> = {
  "darwin-arm64": "bun-darwin-arm64",
  "darwin-x64": "bun-darwin-x64-baseline",
  "linux-x64": "bun-linux-x64-baseline",
  "linux-arm64": "bun-linux-arm64",
}
const target = process.env.CORTEX_TUI_TARGET ?? HOST_TARGETS[`${process.platform}-${process.arch}`]
if (!target) {
  console.error(`No Cortex build for ${process.platform}-${process.arch}; supported: ${Object.keys(HOST_TARGETS).join(", ")}`)
  process.exit(1)
}

mkdirSync(path.dirname(outBinary), { recursive: true })

const result = await Bun.build({
  entrypoints: [path.resolve(projectDir, "src", "index.tsx")],
  target: "bun",
  outdir: path.resolve(projectDir, "dist"),
  plugins: [solidPlugin],
  conditions: ["browser"],
  compile: {
    target: target as Bun.Build.Target,
    outfile: outBinary,
  },
})

if (!result.success) {
  for (const log of result.logs) {
    console.error(log)
  }
  process.exit(1)
}

console.log(`Built cortex-tui (${target}) at ${outBinary}`)
