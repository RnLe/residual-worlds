// The writing rule for this site is plain language with no em-dashes.
// Prose lives in the page, scripts, and styles alike, so the guard walks
// every authored file.

import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

const SITE_ROOT = fileURLToPath(new URL("../..", import.meta.url));

const SKIP_DIRECTORIES = new Set(["node_modules", "dist", "public"]);

const AUTHORED_EXTENSIONS = [".html", ".ts", ".mts", ".css", ".mjs", ".json", ".md"];

function walk(dir: string, found: string[]): void {
  for (const name of readdirSync(dir)) {
    const path = join(dir, name);
    if (statSync(path).isDirectory()) {
      if (!SKIP_DIRECTORIES.has(name)) walk(path, found);
      continue;
    }
    if (!AUTHORED_EXTENSIONS.some((ext) => name.endsWith(ext))) continue;
    found.push(path);
  }
}

describe("em-dash guard", () => {
  it("no authored file contains an em-dash", () => {
    const files: string[] = [];
    walk(SITE_ROOT, files);
    expect(files.length).toBeGreaterThan(8);

    const offenders: string[] = [];
    for (const file of files) {
      const lines = readFileSync(file, "utf8").split("\n");
      lines.forEach((line: string, index: number) => {
        if (line.includes("\u2014")) {
          offenders.push(`${relative(SITE_ROOT, file)}:${index + 1}`);
        }
      });
    }
    expect(offenders, `em-dashes found in:\n${offenders.join("\n")}`).toEqual([]);
  });
});
