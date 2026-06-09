"""Registered-runner daemon (Phase-2 task 2.6).

A long-lived daemon that registers with the platform's runner channel, pulls
``live_playground`` commands, runs the real agent in the customer's boundary
(with their local tools/keys, request-scoped per job), and reports results.
"""

from fastaiagent.runner.channel import RunnerAuthError, RunnerChannel
from fastaiagent.runner.daemon import RunnerDaemon

__all__ = ["RunnerAuthError", "RunnerChannel", "RunnerDaemon"]
