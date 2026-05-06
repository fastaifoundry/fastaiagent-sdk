# FAQ — shared across all three framework sub-examples

The same `LocalKB` is registered as `support-kb` and consumed by:

- the LangGraph agent via `lc_int.kb_as_retriever("support-kb")`
- the CrewAI crew via `ca_int.kb_as_tool("support-kb")`
- the PydanticAI agent via `pa_int.kb_as_tool("support-kb")`

## Common Questions

**Q: How do I reset my password?**
Visit Settings > Security > Reset Password. A reset link is emailed to your address on file. Links expire after 24 hours.

**Q: What's the refund window?**
Full refund within 30 days of purchase. After 30 days we offer prorated refunds based on usage.

**Q: Do you support SSO?**
Yes. SAML and OIDC are available on Enterprise plans. Configure under Settings > Authentication > SSO.

**Q: How do I export my data?**
Settings > Data > Export. Supports CSV and JSON. Exports up to 100k rows complete in ~30 seconds.

**Q: Where can I find API keys?**
Settings > Developer > API Keys. Click "Generate Key" to create one — keys are scoped per-environment.
