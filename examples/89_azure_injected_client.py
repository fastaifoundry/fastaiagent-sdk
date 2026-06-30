"""Azure OpenAI (classic API + Entra ID / managed identity) via an injected client.

Run on a host where ``DefaultAzureCredential`` resolves (e.g. your Azure ML
compute). It mirrors the standard ``AzureOpenAI`` quickstart, then hands that
exact client to fastaiagent's ``LLMClient`` — so the deployments URL,
``api_version``, and managed-identity token refresh are all reused, while
fastaiagent adds agents, tools, tracing, and eval on top.

    pip install --upgrade fastaiagent "openai>=1.0" azure-identity
    python examples/89_azure_injected_client.py
"""

from __future__ import annotations

import os

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI

from fastaiagent import Agent, LLMClient
from fastaiagent.llm.message import UserMessage

endpoint = os.getenv("ENDPOINT_URL", "https://fastaiagent.openai.azure.com/")
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-5.1")

# Entra ID auth — the provider mints (and refreshes) the Bearer token per call.
token_provider = get_bearer_token_provider(
    DefaultAzureCredential(),
    "https://cognitiveservices.azure.com/.default",
)

# Behind a corporate gateway with a private cert? add:
#   import httpx
#   http_client=httpx.Client(verify=False)   # or verify="/path/to/ca.pem"
client = AzureOpenAI(
    azure_endpoint=endpoint,
    azure_ad_token_provider=token_provider,
    api_version="2025-01-01-preview",
)

# Hand the working client to fastaiagent. base_url/api_key/verify are not needed
# here — the injected client owns transport, auth, api_version, and TLS.
llm = LLMClient(provider="azure", model=deployment, openai_client=client)

# 1) Direct completion
print("direct:", llm.complete([UserMessage("hi")]).content)

# 2) Same client, now backing an Agent (tools/guardrails/tracing all work)
agent = Agent(
    name="azure-bot",
    system_prompt="You are an AI assistant that helps people find information.",
    llm=llm,
)
result = agent.run("In one sentence, what is Azure OpenAI?")
print("agent:", result.output)
