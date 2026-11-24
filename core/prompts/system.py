"""Builds the system prompt: a generic skeleton filled with the company's own persona.

It is composed once at start-up and never changes between requests. A stable system prompt
is what lets providers cache it, so per-request data goes into the messages instead.
"""

from core.config import BrainConfig
from core.prompts.store import PromptStore
from core.prompts.types import ComposedPrompt

SYSTEM = "system"
PERSONA = "persona"


def build_system_prompt(store: PromptStore, config: BrainConfig) -> ComposedPrompt:
    skeleton = store.get(SYSTEM)
    persona = store.get(PERSONA)
    text = skeleton.render(
        brain_name=config.name, description=config.description, persona=persona.template
    )
    return ComposedPrompt(text=text, refs=(skeleton.ref, persona.ref))
