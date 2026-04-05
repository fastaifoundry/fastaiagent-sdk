# TechCorp — Frequently Asked Questions

## Account & Login

### How do I reset my password?
Go to the login page and click "Forgot Password." Enter your email address and we'll send a reset link. The link expires after 24 hours. If you don't receive the email, check your spam folder or contact support.

### How do I enable two-factor authentication?
Navigate to Settings > Security > Two-Factor Authentication. We support authenticator apps (Google Authenticator, Authy) and SMS verification. Enterprise plans also support hardware security keys.

### Can I change my email address?
Yes. Go to Settings > Profile > Email. You'll need to verify the new email address before the change takes effect. Note: this does not change your login credentials if you use SSO.

### What happens if my account is locked?
Accounts are locked after 5 failed login attempts. Wait 30 minutes for automatic unlock, or contact support for immediate assistance. Enterprise customers can contact their Domain Admin to unlock.

## Billing & Payments

### What is your refund policy?
Full refund within 30 days of purchase, no questions asked. After 30 days, prorated refund based on usage. Refunds are processed within 5-7 business days to the original payment method.

### How do I upgrade my plan?
Navigate to Settings > Billing > Change Plan. Upgrades take effect immediately, prorated for the current billing cycle. You'll only be charged the difference for the remaining days.

### How do I cancel my subscription?
Go to Settings > Billing > Cancel Subscription. Your access continues until the end of the current billing period. Data is retained for 90 days after cancellation in case you want to reactivate.

### What payment methods do you accept?
We accept all major credit cards (Visa, Mastercard, American Express), bank transfers for annual plans, and invoice billing for Enterprise customers.

## Product Features

### What LLM providers do you support?
We support OpenAI (GPT-4o, GPT-4 Turbo), Anthropic (Claude), Azure OpenAI, Ollama (local models), AWS Bedrock, and any OpenAI-compatible endpoint via custom configuration.

### Do you support SSO login?
SSO is available on Enterprise plans. We support both SAML 2.0 and OpenID Connect (OIDC) protocols. JIT (Just-In-Time) user provisioning is enabled by default.

### What integrations do you support?
We offer 29+ pre-built connectors across databases (PostgreSQL, MySQL, MongoDB), cloud storage (S3, Azure Blob, Google Drive), CRM (Salesforce, Dynamics 365), messaging (Slack, Teams, Telegram), email (Gmail, Outlook), and more. We also support MCP (Model Context Protocol) for dynamic tool discovery.

### Can I export my data?
Yes. Data export is available from Settings > Data > Export. Supports CSV and JSON formats. You can export traces, eval results, and prompt versions. Enterprise customers can also use the Public API for programmatic export.

### How does the knowledge base work?
Upload documents (PDF, DOCX, TXT, Markdown, Excel) and they're automatically chunked, embedded, and indexed for semantic search. Agents can search the knowledge base as a tool during conversations.

## Technical

### What is Agent Replay?
Agent Replay lets you step through any past agent execution span-by-span — seeing the exact inputs, outputs, and decisions at each step. You can fork from any point and re-run with modified inputs to explore "what if?" scenarios.

### What are guardrails?
Guardrails are safety checks that run on agent inputs and outputs. We support 5 types: code-based, LLM-as-Judge, regex pattern matching, schema validation, and classification. 5 built-in guardrails are included (PII detection, toxicity, JSON validation, cost limits, domain allowlist).

### How does the eval framework work?
Create test datasets, define scoring dimensions (LLM-as-Judge, custom code, or built-in metrics like Ragas), run evaluations, and compare results. Online eval policies can score production traces automatically. Results integrate with CI/CD pipelines.

### Is there an API?
Yes. Full RESTful API at `/public/v1/` with scoped API keys, rate limiting, and SSE streaming. You can execute agents, submit feedback, manage approvals, and run evaluations programmatically.
