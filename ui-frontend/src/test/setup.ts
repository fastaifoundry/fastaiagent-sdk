import "@testing-library/jest-dom/vitest";

// jsdom doesn't implement matchMedia; ThemeProvider relies on it.
if (!window.matchMedia) {
  Object.defineProperty(window, "matchMedia", {
    writable: true,
    value: (query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addEventListener: () => {},
      removeEventListener: () => {},
      addListener: () => {},
      removeListener: () => {},
      dispatchEvent: () => false,
    }),
  });
}

// jsdom doesn't implement ResizeObserver; React Flow uses it to track the
// canvas size. A no-op polyfill is enough for non-visual component tests.
const g = globalThis as unknown as {
  ResizeObserver?: unknown;
  DOMRect?: unknown;
};
if (typeof g.ResizeObserver === "undefined") {
  class ResizeObserverPolyfill {
    observe(): void {}
    unobserve(): void {}
    disconnect(): void {}
  }
  g.ResizeObserver = ResizeObserverPolyfill;
}

// React Flow also calls these DOM APIs; jsdom omits them.
if (typeof Element !== "undefined" && !Element.prototype.scrollTo) {
  (Element.prototype as unknown as { scrollTo: () => void }).scrollTo = () => {};
}
if (typeof g.DOMRect === "undefined") {
  g.DOMRect = class {
    static fromRect() {
      return { x: 0, y: 0, width: 0, height: 0, top: 0, left: 0, right: 0, bottom: 0 };
    }
  };
}
