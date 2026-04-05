"""
RunContext — Dependency injection for the support agent.

Dependencies are created once and passed to every tool call via RunContext[Deps].
This keeps tools pure — no global state, no import-time side effects.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TicketClient:
    """Mock ticket system client. Replace with your real API."""
    base_url: str = "https://tickets.example.com"
    _counter: int = field(default=1000, repr=False)

    async def create(self, subject: str, description: str, priority: str = "medium") -> dict:
        self._counter += 1
        return {
            "ticket_id": f"TKT-{self._counter}",
            "subject": subject,
            "description": description,
            "priority": priority,
            "status": "open",
        }


@dataclass
class CRMClient:
    """Mock CRM client. Replace with your real Salesforce/HubSpot connector."""
    base_url: str = "https://crm.example.com"

    async def lookup(self, email: str) -> Optional[dict]:
        # Mock data — replace with actual CRM lookup
        mock_accounts = {
            "alice@acme.com": {
                "name": "Alice Johnson",
                "company": "Acme Corp",
                "plan": "Enterprise",
                "account_id": "ACC-4821",
                "status": "active",
            },
            "bob@startup.io": {
                "name": "Bob Chen",
                "company": "Startup.io",
                "plan": "Pro",
                "account_id": "ACC-7293",
                "status": "active",
            },
        }
        return mock_accounts.get(email)


@dataclass
class OrderClient:
    """Mock order tracking client."""

    async def check_status(self, order_id: str) -> Optional[dict]:
        mock_orders = {
            "ORD-9912": {"status": "shipped", "tracking": "1Z999AA10123456784", "eta": "April 7, 2026"},
            "ORD-8834": {"status": "processing", "tracking": None, "eta": "April 10, 2026"},
            "ORD-7701": {"status": "delivered", "tracking": "1Z999AA10987654321", "eta": None},
        }
        return mock_orders.get(order_id)


@dataclass
class Deps:
    """All runtime dependencies for the support agent."""
    ticket_client: TicketClient
    crm_client: CRMClient
    order_client: OrderClient
    user_email: str = "unknown@example.com"


async def create_deps(user_email: str = "alice@acme.com") -> Deps:
    """Factory function to create dependencies."""
    return Deps(
        ticket_client=TicketClient(),
        crm_client=CRMClient(),
        order_client=OrderClient(),
        user_email=user_email,
    )
