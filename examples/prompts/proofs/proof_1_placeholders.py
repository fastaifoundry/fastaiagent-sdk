"""Proof for "1 · Between the template and the text" on docs/prompts/prompt-boundaries.md.

Offline: a registry on a scratch local.db, no model.

Two placeholder kinds resolve at two moments: ``{{@fragment}}`` when the prompt
is loaded, ``{{variable}}`` when it is formatted. A fragment can carry a variable
of its own. And nothing is checked for you: an unknown fragment and an unfilled
variable both stay in the text, verbatim.
"""

import re

import _common
from _common import heading

from fastaiagent.prompt import PromptRegistry

reg = PromptRegistry(path=str(_common.SCRATCH))
reg.register_fragment("tone", "Be {{tone}} and concise.")
stored = reg.register(
    name="greeting",
    template="Hello {{name}}, welcome to {{company}}. {{@tone}} {{@legal}}",
)

heading("what register() stored")
print("template :", repr(stored.template))
print("variables:", sorted(stored.variables))

loaded = reg.load("greeting")
heading("what load() returns — fragments resolved, variables still open")
print("template :", repr(loaded.template))
print("variables:", sorted(loaded.variables))

heading("what format() produces")
text = loaded.format(name="Dana", company="Acme")
print("format(name, company)      :", repr(text))
print("left in the text, verbatim :", re.findall(r"\{\{[^}]+\}\}", text))
text = loaded.format(name="Dana", company="Acme", tone="warm")
print("format(name, company, tone):", repr(text))
print("left in the text, verbatim :", re.findall(r"\{\{[^}]+\}\}", text))
