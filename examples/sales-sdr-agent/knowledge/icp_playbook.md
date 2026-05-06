# Ideal Customer Profile — Playbook

This playbook is loaded into a `LocalKB` at startup and consulted by the
**scorer** node to grade prospects. Edit it to match your real ICP.

## Firmographic fit

- **Company size**: 50–5,000 employees. Below 50 the deal size won't justify our white-glove onboarding; above 5,000 procurement cycles take longer than our typical pipeline.
- **Industry**: SaaS, fintech, e-commerce, AI/ML platform companies. Avoid heavily regulated verticals (defense, classified healthcare) — our compliance story isn't ready.
- **Geography**: North America, EU, ANZ. We don't have data residency for India / China yet.
- **Funding stage**: Series A through pre-IPO. Bootstrapped is fine if revenue is >$5M ARR. Post-IPO companies are workable but need exec sponsorship.

## Technographic fit

- Already runs LLM-driven features in production (any provider — OpenAI, Anthropic, Mistral, self-hosted).
- Has at least one Python or TypeScript service in their stack.
- Uses git, observable infra (Datadog / Grafana / OTel), and CI/CD. They'll value our trace + replay primitives.

## Persona fit

- Buyer: VP Engineering, Head of AI/ML, Head of Platform, CTO.
- Champion: Senior engineer or staff-eng who built the existing LLM features and is feeling the pain of un-observable agents.
- Anti-pattern: someone "exploring AI" without a concrete production use case. They'll spin for months without converting.

## Disqualifiers

Auto-reject if any of the following apply:

- The prospect is a competitor (LangChain, LlamaIndex, CrewAI, AgentOps, Helicone, Langfuse).
- The prospect is a personal-project / hobbyist (LinkedIn says "indie hacker", "consultant" with team_size=1).
- The prospect's product is *itself* a foundation model (we sell agent infrastructure, not foundation models).
- Recent fundraise/announcement that suggests they'll build everything in-house.

## Scoring rubric

Score is a float in `[0.0, 1.0]`. Return JSON: `{"score": <float>, "reasons": ["..."]}`.

| Bucket | Score | What it means |
|---|---|---|
| Strong fit | 0.85–1.0 | Hits ≥4 firmo + technographic checks. Clear champion persona. Outreach right away. |
| Workable | 0.7–0.85 | Hits ≥3 checks. Some ambiguity but worth a conversation. Outreach with light personalization. |
| Marginal | 0.5–0.7 | Hits 1–2 checks; gaps elsewhere. Park in nurture; do NOT reach out today. |
| Disqualified | <0.5 | Hits a disqualifier OR <1 check. Log and move on. |

Anything `>= 0.7` is "qualified" and proceeds to outreach. Below that, the chain disqualifies and logs the reason.
