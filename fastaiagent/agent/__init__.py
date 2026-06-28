"""Agent module — the central component of the SDK."""

from fastaiagent.agent.agent import Agent, AgentConfig, AgentResult
from fastaiagent.agent.context import RunContext
from fastaiagent.agent.memory import AgentMemory, ComposableMemory
from fastaiagent.agent.memory_blocks import (
    FactExtractionBlock,
    FewShotBlock,
    MemoryBlock,
    PersistentFactBlock,
    StaticBlock,
    SummaryBlock,
    VectorBlock,
)
from fastaiagent.agent.middleware import (
    AgentMiddleware,
    MiddlewareContext,
    RedactPII,
    Reflect,
    ToolBudget,
    TrimLongMessages,
)
from fastaiagent.agent.swarm import Swarm, SwarmError, SwarmState
from fastaiagent.agent.team import Supervisor, Worker

__all__ = [
    "Agent",
    "AgentConfig",
    "AgentMemory",
    "AgentMiddleware",
    "AgentResult",
    "ComposableMemory",
    "FactExtractionBlock",
    "FewShotBlock",
    "MemoryBlock",
    "MiddlewareContext",
    "PersistentFactBlock",
    "RedactPII",
    "Reflect",
    "RunContext",
    "StaticBlock",
    "SummaryBlock",
    "Supervisor",
    "Swarm",
    "SwarmError",
    "SwarmState",
    "ToolBudget",
    "TrimLongMessages",
    "VectorBlock",
    "Worker",
]
