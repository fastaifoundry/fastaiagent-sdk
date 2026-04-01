"""Example 07: Evaluate an agent with multiple scorers.

Shows how to run systematic evaluation with built-in
and custom scorers.
"""

from fastaiagent.eval import Dataset, Scorer, ScorerResult, evaluate
from fastaiagent.eval.builtins import Contains


# A simple agent function (replace with your real agent)
def my_agent(input_text: str) -> str:
    """Simple echo agent for demonstration."""
    responses = {
        "What is 2+2?": "The answer is 4.",
        "What color is the sky?": "The sky is blue.",
        "Who wrote Hamlet?": "Shakespeare wrote Hamlet.",
    }
    return responses.get(input_text, f"I don't know about: {input_text}")


# Custom scorer using the @Scorer.code decorator
@Scorer.code("has_answer")
def has_answer(input: str, output: str, expected: str | None = None) -> ScorerResult:
    """Check if the output contains a definitive answer (not 'I don't know')."""
    is_answer = "don't know" not in output.lower()
    return ScorerResult(score=1.0 if is_answer else 0.0, passed=is_answer)


if __name__ == "__main__":
    # Define test cases
    dataset = Dataset.from_list([
        {"input": "What is 2+2?", "expected": "4"},
        {"input": "What color is the sky?", "expected": "blue"},
        {"input": "Who wrote Hamlet?", "expected": "Shakespeare"},
        {"input": "What is quantum computing?", "expected": "quantum"},
    ])

    # Run evaluation with multiple scorers
    results = evaluate(
        agent_fn=my_agent,
        dataset=dataset,
        scorers=[Contains(), has_answer],
    )

    print(results.summary())

    # Export results
    results.export("/tmp/fastaiagent-eval-results.json")
    print("\nResults exported to /tmp/fastaiagent-eval-results.json")
