// Copy the measured summary dataset into the static site. Refuses to invent data: if the
// summary is missing, the build fails with instructions instead of shipping placeholders.
import { copyFileSync, existsSync, mkdirSync, statSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const src = resolve(here, "../../results/summary.json");
const dst = resolve(here, "../public/data/summary.json");
if (!existsSync(src)) {
  console.error(`missing ${src}\nrun: gpulab analyze   (from the repository root)`);
  process.exit(1);
}
mkdirSync(dirname(dst), { recursive: true });
copyFileSync(src, dst);
console.log(`synced summary.json (${(statSync(dst).size / 1024).toFixed(0)} KiB)`);
