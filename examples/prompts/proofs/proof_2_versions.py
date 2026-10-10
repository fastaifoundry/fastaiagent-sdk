"""Proof for "2 · Between one version and the next" on docs/prompts/prompt-boundaries.md.

Offline: a registry on a scratch local.db, no model.

Every register() without a version adds a row; an alias is a pointer you move by
hand; diff() compares two rows. Then the edge: register() with an explicit
version= that already exists replaces that row in place.
"""

import _common
from _common import heading

from fastaiagent.prompt import PromptRegistry

reg = PromptRegistry(path=str(_common.SCRATCH))

heading("versions accumulate, the alias stays put")
reg.register("greeting", "Hello {{name}}!")  # v1
reg.register("greeting", "Hi there, {{name}}! Welcome.")  # v2
reg.set_alias("greeting", version=1, alias="production")
print("load()                 → v", reg.load("greeting").version)
print("load(alias=production) → v", reg.load("greeting", alias="production").version)
reg.register("greeting", "Hey {{name}}.")  # v3
print("after a third register : latest v", reg.load("greeting").version,
      "· production still v", reg.load("greeting", alias="production").version)
print("list()                 :", reg.list())

heading("diff(1, 2)")
print(reg.diff("greeting", 1, 2))

heading("register(version=1) on an existing v1")
reg.register("greeting", "REPLACED {{name}}", version=1)
print("v1 now reads           :", repr(reg.load("greeting", version=1).template))
print("production (→ v1) reads:", repr(reg.load("greeting", alias="production").template))
print("load() latest          : v", reg.load("greeting").version)
print("list()                 :", reg.list())
