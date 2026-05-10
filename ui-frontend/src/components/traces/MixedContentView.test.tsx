/**
 * MixedContentView walks input/output JSON and inlines images & PDFs in
 * place of raw base64. The fall-through path delegates to JsonViewer when
 * no multimodal parts are present, so existing trace surfaces look
 * identical for plain text/JSON traces.
 */
import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { MixedContentView } from "./MixedContentView";

describe("MixedContentView", () => {
  it("falls through to JsonViewer for plain JSON (no multimodal parts)", () => {
    const { container } = render(
      <MixedContentView value={{ "agent.input": "hello" }} />
    );
    // No mixed-content marker means the JSON path was taken.
    expect(container.querySelector('[data-multimodal="true"]')).toBeNull();
    // JsonViewer renders the literal value somewhere in the DOM.
    expect(screen.getAllByText(/hello/).length).toBeGreaterThan(0);
  });

  it("falls through to JsonViewer for plain string input", () => {
    const { container } = render(<MixedContentView value="just a string" />);
    expect(container.querySelector('[data-multimodal="true"]')).toBeNull();
    expect(screen.getByText("just a string")).toBeInTheDocument();
  });

  it("renders inline image thumbnails when content parts include images", () => {
    const value = {
      "gen_ai.request.messages": [
        {
          role: "user",
          content: [
            { type: "text", text: "What is in this picture?" },
            {
              type: "image_url",
              image_url: { url: "https://example.com/cat.jpg" },
              media_type: "image/jpeg",
            },
          ],
        },
      ],
    };
    const { container } = render(
      <MixedContentView value={value} traceId="t1" spanId="s1" />
    );
    expect(container.querySelector('[data-multimodal="true"]')).toBeInTheDocument();
    const img = screen.getByRole("img");
    expect(img).toHaveAttribute("src", "https://example.com/cat.jpg");
    // Text part renders too.
    expect(screen.getByText("What is in this picture?")).toBeInTheDocument();
  });

  it("renders a PdfCard for PDF content parts with a page-count badge", () => {
    const value = {
      "agent.input": [
        { type: "text", text: "Summarize this contract." },
        {
          type: "input_pdf",
          filename: "contract.pdf",
          page_count: 12,
          size_bytes: 340_000,
        },
      ],
    };
    const { container } = render(<MixedContentView value={value} />);
    expect(container.querySelector('[data-multimodal="true"]')).toBeInTheDocument();
    expect(screen.getByText("contract.pdf")).toBeInTheDocument();
    expect(screen.getByText("12 pages")).toBeInTheDocument();
  });

  it("shows the empty label when value is null/undefined", () => {
    render(<MixedContentView value={null} emptyLabel="No input captured." />);
    expect(screen.getByText("No input captured.")).toBeInTheDocument();
  });

  it("preserves part order across messages", () => {
    const value = [
      {
        role: "user",
        content: [
          { type: "text", text: "first" },
          {
            type: "image",
            url: "https://example.com/a.png",
            media_type: "image/png",
          },
          { type: "text", text: "second" },
        ],
      },
    ];
    render(<MixedContentView value={value} />);
    const textNodes = screen.getAllByText(/first|second/);
    expect(textNodes.map((n) => n.textContent)).toEqual(["first", "second"]);
  });

  // ---------------------------------------------------------------------
  // security_review_1.md H6 — XSS sanitization in trace markdown
  // ---------------------------------------------------------------------

  it("strips javascript: protocol on links", () => {
    // A malicious LLM output could try to ship a clickable
    // ``javascript:`` link in trace text. ``rehype-sanitize`` must rewrite
    // or drop the dangerous href so the rendered anchor is harmless.
    const value = "[click me](javascript:alert('xss'))";
    const { container } = render(<MixedContentView value={value} />);
    const anchors = container.querySelectorAll("a");
    for (const a of anchors) {
      const href = a.getAttribute("href") ?? "";
      expect(href.toLowerCase()).not.toMatch(/^javascript:/);
    }
  });

  it("does not execute inline scripts from raw HTML", () => {
    // ``react-markdown`` already escapes raw HTML by default; this test
    // confirms our sanitizer keeps doing the right thing if a future
    // refactor adds ``rehype-raw`` or similar.
    const value = "Hello<script>window.__pwned = true</script>World";
    const { container } = render(<MixedContentView value={value} />);
    expect(container.querySelector("script")).toBeNull();
    // ``window.__pwned`` should never have been set.
    expect((window as unknown as { __pwned?: boolean }).__pwned).toBeUndefined();
  });

  it("strips dangerous event-handler attributes from img tags", () => {
    const value = '<img src="x" onerror="window.__xss=1">';
    const { container } = render(<MixedContentView value={value} />);
    const imgs = container.querySelectorAll("img");
    for (const img of imgs) {
      expect(img.getAttribute("onerror")).toBeNull();
    }
    expect((window as unknown as { __xss?: number }).__xss).toBeUndefined();
  });
});
