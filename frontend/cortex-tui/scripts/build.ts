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

mkdirSync(path.dirname(outBinary), { recursive: true })

const result = await Bun.build({
  entrypoints: [path.resolve(projectDir, "src", "index.tsx")],
  target: "bun",
  outdir: path.resolve(projectDir, "dist"),
  plugins: [solidPlugin],
  conditions: ["browser"],
  compile: {
    target: "bun-darwin-arm64",
    outfile: outBinary,
  },
})

if (!result.success) {
  for (const log of result.logs) {
    console.error(log)
  }
  process.exit(1)
}

console.log(`Built cortex-tui binary at ${outBinary}`)
