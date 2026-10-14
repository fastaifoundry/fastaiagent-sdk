import { describe, expect, it, vi } from "vitest";
import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { renderWithProviders } from "@/test/utils";
import { RunEvalDialog } from "./RunEvalDialog";
import type { RunEvalOptions, ScorerOption } from "@/lib/types";

/**
 * The Run eval dialog (1.89.0) builds the request from the options endpoint:
 * agent or dataset check, scorers as names or specs, warnings from what the
 * rows carry. The api.ts boundary (fetch) is the only mock surface.
 */

function scorer(
  name: string,
  family: ScorerOption["family"],
  extra: Partial<ScorerOption> = {}
): ScorerOption {
  return {
    name,
    family,
    description: `${name} description`,
    params: [],
    needs: [],
    judge: false,
    needs_agent: false,
    available: true,
    unavailable_reason: null,
    rows_missing: {},
    ...extra,
  };
}

const JUDGE_PARAMS = (scale: string): ScorerOption["params"] => [
  { name: "criteria", type: "string", required: true, default: null, minimum: null, maximum: null, description: "" },
  { name: "scale", type: "string", required: false, default: scale, minimum: null, maximum: null, description: "binary, 0-1, 1-5 or 1-10" },
  { name: "threshold", type: "number", required: false, default: 0.5, minimum: 0, maximum: 1, description: "" },
];

function options(overrides: Partial<RunEvalOptions> = {}): RunEvalOptions {
  return {
    agents: [
      { name: "support-bot", kind: "agent", supported: true, reason: null },
      {
        name: "pipeline",
        kind: "chain",
        supported: false,
        reason: "a Chain has no run(); evaluate its agents",
      },
    ],
    scorers: [
      scorer("exact_match", "deterministic", {
        needs: ["expected_output"],
        rows_missing: { expected_output: 1 },
      }),
      scorer("regex_match", "deterministic", {
        params: [
          {
            name: "pattern",
            type: "string",
            required: true,
            default: null,
            minimum: null,
            maximum: null,
            description: "Python regex, at most 500 characters.",
          },
        ],
      }),
      scorer("latency", "deterministic", {
        params: [
          { name: "max_ms", type: "integer", required: false, default: 5000, minimum: 1, maximum: null, description: "" },
        ],
      }),
      scorer("task_completion", "llm", { judge: true }),
      scorer("llm_judge", "llm", { judge: true, params: JUDGE_PARAMS("binary") }),
      scorer("geval", "llm", { judge: true, params: JUDGE_PARAMS("1-5") }),
      scorer("tool_usage_accuracy", "trajectory", {
        needs: ["expected_trajectory"],
        needs_agent: true,
        rows_missing: { expected_trajectory: 0 },
      }),
      scorer("faithfulness", "rag", {
        needs: ["context"],
        judge: true,
        available: false,
        unavailable_reason: "no row has context",
        rows_missing: { context: 3 },
      }),
    ],
    judge_models: [
      { provider: "anthropic", model: "claude-haiku" },
      { provider: "openai", model: "gpt-4o" },
      { provider: "openai", model: "gpt-4o-mini" },
    ],
    dataset_features: {
      rows: 3,
      with_expected_output: 2,
      with_expected_trajectory: 3,
      with_expected_tool_calls: 0,
      with_expected_steps: 0,
      with_context: 0,
      multimodal: 1,
    },
    echo_available: true,
    ...overrides,
  };
}

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

const OK_RESULT = {
  run_id: "r1",
  mode: "agent",
  kind: "ui",
  pass_rate: 0.5,
  pass_count: 1,
  fail_count: 1,
  errored_count: 1,
  total_cases: 3,
  cases: { passed: 1, failed: 1, errored: 1, pass_rate: 0.5 },
  checks: { passed: 3, failed: 1, errored: 0, skipped: 2, pass_rate: 0.75 },
  note: null,
};

function stubApi(
  opts: RunEvalOptions,
  respond: () => Response | Promise<Response> = () => json(OK_RESULT)
) {
  const posts: Record<string, unknown>[] = [];
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = typeof input === "string" ? input : (input as Request).url;
    const method = (init?.method ?? "GET").toUpperCase();
    if (method === "GET" && url.endsWith("/api/datasets/qa/run-eval/options")) {
      return json(opts);
    }
    if (method === "POST" && url.endsWith("/api/datasets/qa/run-eval")) {
      posts.push(JSON.parse(String(init?.body)));
      return respond();
    }
    return json({});
  });
  return posts;
}

function renderDialog() {
  return renderWithProviders(
    <RunEvalDialog datasetName="qa" open onOpenChange={() => {}} />
  );
}

const box = (name: string) => screen.getByRole("checkbox", { name });
const run = () => fireEvent.click(screen.getByRole("button", { name: /^run$/i }));

describe("RunEvalDialog", () => {
  it("lists agents and scorers from the options; unsupported ones are disabled", async () => {
    stubApi(options());
    renderDialog();

    const agent = await screen.findByRole("combobox", { name: "Agent" });
    expect(agent).toHaveValue("support-bot");
    expect(within(agent).getByRole("option", { name: /support-bot/ })).toBeEnabled();
    expect(within(agent).getByRole("option", { name: /pipeline/ })).toBeDisabled();
    expect(screen.getByRole("radio", { name: "Agent" })).toBeChecked();

    // Grouped by family; exact_match on by default.
    expect(screen.getByText("Deterministic")).toBeInTheDocument();
    expect(screen.getByText("LLM judges")).toBeInTheDocument();
    expect(screen.getByText("Trajectory")).toBeInTheDocument();
    expect(screen.getByText("RAG")).toBeInTheDocument();
    expect(box("exact_match")).toBeChecked();
    expect(box("regex_match")).not.toBeChecked();
    // An unavailable scorer says why.
    expect(box("faithfulness")).toBeDisabled();
    expect(screen.getByText("no row has context")).toBeInTheDocument();
  });

  it("preselects the dataset check when no agent is registered", async () => {
    const posts = stubApi(options({ agents: [] }));
    renderDialog();

    expect(await screen.findByRole("radio", { name: /dataset check/i })).toBeChecked();
    expect(screen.getByRole("radio", { name: "Agent" })).toBeDisabled();
    expect(screen.getByText("fastaiagent ui --agent app.py:my_agent")).toBeInTheDocument();
    expect(screen.getByText(/tests the dataset, not a model/)).toBeInTheDocument();

    run();
    await waitFor(() => expect(posts).toHaveLength(1));
    expect(posts[0]).toEqual({ mode: "echo", scorers: ["exact_match"] });
  });

  it("sends regex_match as a builtin spec with its pattern", async () => {
    const posts = stubApi(options());
    renderDialog();

    fireEvent.click(await screen.findByRole("checkbox", { name: "regex_match" }));
    const pattern = screen.getByLabelText("regex_match pattern");
    fireEvent.change(pattern, { target: { value: "^\\d+$" } });
    fireEvent.change(screen.getByLabelText("Run name (optional)"), {
      target: { value: "nightly" },
    });
    run();

    await waitFor(() => expect(posts).toHaveLength(1));
    expect(posts[0]).toEqual({
      agent_name: "support-bot",
      mode: "agent",
      run_name: "nightly",
      scorers: [
        "exact_match",
        { type: "builtin", name: "regex_match", params: { pattern: "^\\d+$" } },
      ],
    });
  });

  it("refuses to run regex_match without a pattern, without calling the server", async () => {
    const posts = stubApi(options());
    renderDialog();

    fireEvent.click(await screen.findByRole("checkbox", { name: "regex_match" }));
    run();
    expect(await screen.findByRole("alert")).toHaveTextContent("regex_match needs pattern.");
    expect(posts).toHaveLength(0);
  });

  it("sends llm_judge as its own spec type with criteria, scale and model", async () => {
    const posts = stubApi(options());
    renderDialog();

    fireEvent.click(await screen.findByRole("checkbox", { name: "exact_match" }));
    fireEvent.click(box("llm_judge"));
    // gpt-4o-mini is the default judge.
    const model = screen.getByLabelText("llm_judge judge model");
    expect(within(model).getByRole("option", { selected: true })).toHaveTextContent(
      "openai · gpt-4o-mini"
    );
    fireEvent.change(screen.getByLabelText("llm_judge criteria"), {
      target: { value: "Polite and correct" },
    });
    fireEvent.change(screen.getByLabelText("llm_judge scale"), { target: { value: "1-5" } });
    fireEvent.change(model, { target: { value: "0" } });
    run();

    await waitFor(() => expect(posts).toHaveLength(1));
    expect(posts[0]!.scorers).toEqual([
      {
        type: "llm_judge",
        criteria: "Polite and correct",
        scale: "1-5",
        threshold: 0.5,
        model: { provider: "anthropic", model: "claude-haiku" },
      },
    ]);
  });

  it("sends a judge built-in with its model, and geval steps and rubric", async () => {
    const posts = stubApi(options());
    renderDialog();

    fireEvent.click(await screen.findByRole("checkbox", { name: "exact_match" }));
    fireEvent.click(box("task_completion"));
    fireEvent.click(box("geval"));
    fireEvent.change(screen.getByLabelText("geval criteria"), {
      target: { value: "Answers the question" },
    });
    fireEvent.click(screen.getByRole("button", { name: /add step/i }));
    fireEvent.click(screen.getByRole("button", { name: /add step/i }));
    fireEvent.change(screen.getByLabelText("geval step 1"), { target: { value: "Read it" } });
    fireEvent.change(screen.getByLabelText("geval step 2"), { target: { value: "Check it" } });
    fireEvent.click(screen.getByRole("button", { name: /add rubric row/i }));
    fireEvent.click(screen.getByRole("button", { name: /add rubric row/i }));
    fireEvent.change(screen.getByLabelText("geval rubric 1 score"), { target: { value: "1" } });
    fireEvent.change(screen.getByLabelText("geval rubric 1 description"), {
      target: { value: "wrong" },
    });
    fireEvent.change(screen.getByLabelText("geval rubric 2 score"), { target: { value: "5" } });
    fireEvent.change(screen.getByLabelText("geval rubric 2 description"), {
      target: { value: "right" },
    });
    run();

    await waitFor(() => expect(posts).toHaveLength(1));
    const mini = { provider: "openai", model: "gpt-4o-mini" };
    expect(posts[0]!.scorers).toEqual([
      { type: "builtin", name: "task_completion", model: mini },
      {
        type: "geval",
        criteria: "Answers the question",
        scale: "1-5",
        threshold: 0.5,
        model: mini,
        evaluation_steps: ["Read it", "Check it"],
        rubric: [
          [1, "wrong"],
          [5, "right"],
        ],
      },
    ]);
  });

  it("warns about rows missing a field, images, and the judge-call count", async () => {
    stubApi(options());
    renderDialog();

    const warnings = await screen.findByTestId("run-eval-warnings");
    expect(warnings).toHaveTextContent(
      "exact_match needs expected_output; 1 of 3 rows lack it: those cases are n/a"
    );
    expect(warnings).toHaveTextContent(
      "1 row has images or PDFs; text scorers may not apply to them"
    );
    expect(warnings).not.toHaveTextContent(/judge calls/);

    fireEvent.click(box("task_completion"));
    fireEvent.click(box("llm_judge"));
    expect(screen.getByTestId("run-eval-warnings")).toHaveTextContent("~6 judge calls");

    // latency reads the run, not its text: no image warning on its own.
    fireEvent.click(box("exact_match"));
    fireEvent.click(box("task_completion"));
    fireEvent.click(box("llm_judge"));
    fireEvent.click(box("latency"));
    expect(screen.queryByTestId("run-eval-warnings")).not.toBeInTheDocument();
  });

  it("a dataset check disables needs_agent scorers and sends mode echo without an agent", async () => {
    const posts = stubApi(options());
    renderDialog();

    fireEvent.click(await screen.findByRole("checkbox", { name: "tool_usage_accuracy" }));
    expect(box("tool_usage_accuracy")).toBeChecked();

    fireEvent.click(screen.getByRole("radio", { name: /dataset check/i }));
    expect(screen.queryByRole("combobox", { name: "Agent" })).not.toBeInTheDocument();
    expect(box("tool_usage_accuracy")).toBeDisabled();
    expect(box("tool_usage_accuracy")).not.toBeChecked();
    expect(
      screen.getByText("needs an agent: it measures the agent's run")
    ).toBeInTheDocument();
    expect(screen.getByTestId("run-eval-warnings")).toHaveTextContent(
      "Dataset check: inputs are echoed back; the result does not measure a model"
    );

    run();
    await waitFor(() => expect(posts).toHaveLength(1));
    expect(posts[0]).toEqual({ mode: "echo", scorers: ["exact_match"] });
  });

  it("shows a 422 detail verbatim and keeps the form editable", async () => {
    const detail = "Scorer 'regex_match': the pattern does not compile: unterminated character set";
    stubApi(options(), () => json({ detail }, 422));
    renderDialog();

    fireEvent.click(await screen.findByRole("checkbox", { name: "regex_match" }));
    fireEvent.change(screen.getByLabelText("regex_match pattern"), {
      target: { value: "[a-" },
    });
    run();

    expect(await screen.findByRole("alert")).toHaveTextContent(detail);
    expect(screen.getByRole("button", { name: /^run$/i })).toBeEnabled();
    expect(screen.getByLabelText("regex_match pattern")).toBeEnabled();
    expect(screen.getByLabelText("regex_match pattern")).toHaveValue("[a-");
  });

  it("shows pending, then the case and check numbers and a link to the run", async () => {
    let release: (r: Response) => void = () => {};
    const pendingResponse = new Promise<Response>((resolve) => {
      release = resolve;
    });
    stubApi(options(), () => pendingResponse);
    renderDialog();

    await screen.findByRole("checkbox", { name: "exact_match" });
    run();
    const busy = await screen.findByRole("button", { name: /running/i });
    expect(busy).toBeDisabled();

    release(
      json({
        ...OK_RESULT,
        mode: "echo",
        kind: "ui-echo",
        note: "A dataset check: each input was echoed back unchanged.",
      })
    );
    const result = await screen.findByTestId("run-eval-result");
    expect(result).toHaveTextContent("Cases 50%: 1 passed · 1 failed · 1 errored");
    expect(result).toHaveTextContent("Checks 75%: 3 passed · 1 failed · 2 n/a");
    expect(result).toHaveTextContent("A dataset check: each input was echoed back unchanged.");
    expect(within(result).getByRole("link", { name: "Open run" })).toHaveAttribute(
      "href",
      "/evals/r1"
    );
    expect(screen.getByRole("button", { name: /run again/i })).toBeEnabled();
  });
});
