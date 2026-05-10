/**
 * Regression tests for security_review_1.md M1 + M2 (Medium batch
 * shipping in 1.11.0).
 *
 * - M1: the inline-attachment iframe is sandboxed so a malicious or
 *   misconfigured attachment endpoint can't run scripts in the UI's
 *   origin.
 * - M2: every ``target="_blank"`` link carries ``rel="noopener
 *   noreferrer"`` to defeat reverse tabnabbing.
 *
 * AttachmentTile is module-internal and AttachmentGallery fetches its
 * own data, so the cleanest regression check is to assert the source
 * carries the right attributes — the property we actually ship.
 */
import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const MODAL_TSX = readFileSync(
  path.join(__dirname, "AttachmentModal.tsx"),
  "utf-8"
);
const GALLERY_TSX = readFileSync(
  path.join(__dirname, "AttachmentGallery.tsx"),
  "utf-8"
);

describe("M1 — attachment iframe is sandboxed", () => {
  it("AttachmentModal.tsx renders the iframe with sandbox attribute", () => {
    // The iframe block must declare a sandbox.
    expect(MODAL_TSX).toMatch(/<iframe[\s\S]*?sandbox=/);
  });
  it("sandbox does NOT enable scripts or top-navigation", () => {
    const sandboxMatch = MODAL_TSX.match(/sandbox="([^"]+)"/);
    expect(sandboxMatch).not.toBeNull();
    const value = sandboxMatch?.[1] ?? "";
    expect(value).not.toContain("allow-scripts");
    expect(value).not.toContain("allow-top-navigation");
    expect(value).not.toContain("allow-modals");
    expect(value).not.toContain("allow-forms");
  });
  it("iframe sets referrerPolicy='no-referrer'", () => {
    expect(MODAL_TSX).toMatch(/<iframe[\s\S]*?referrerPolicy="no-referrer"/);
  });
});

describe("M2 — target='_blank' carries rel='noopener noreferrer'", () => {
  it("every target=_blank in AttachmentGallery has noopener+noreferrer", () => {
    const blankAnchors = [
      ...GALLERY_TSX.matchAll(/<a\b[\s\S]*?target="_blank"[\s\S]*?>/g),
    ];
    // Sanity: there is at least one such anchor in the file.
    expect(blankAnchors.length).toBeGreaterThan(0);
    for (const m of blankAnchors) {
      const tag = m[0];
      const relMatch = tag.match(/rel="([^"]+)"/);
      expect(relMatch, `<a target=_blank> missing rel: ${tag}`).not.toBeNull();
      const rel = relMatch?.[1] ?? "";
      expect(rel).toContain("noopener");
      expect(rel).toContain("noreferrer");
    }
  });
});
