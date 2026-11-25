"""Builds the guardrails from the pack's ``guardrails`` section."""

from core.config import BrainConfig, InjectionDetector, ModelTier
from core.guardrails.guards import Guardrails
from core.guardrails.injection import (
    INJECTION_PROMPT,
    HeuristicInjectionDetector,
    ModelInjectionDetector,
    NoInjectionDetector,
)
from core.guardrails.injection import InjectionDetector as Detector
from core.guardrails.leak import PromptLeakGuard
from core.guardrails.pii import PiiRedactor
from core.llm import ModelGateway
from core.prompts import PromptStore


def build_guardrails(
    config: BrainConfig, system_prompt: str, gateway: ModelGateway, prompts: PromptStore
) -> Guardrails:
    guards = config.guardrails
    detector: Detector
    match guards.injection.detector:
        case InjectionDetector.HEURISTIC:
            detector = HeuristicInjectionDetector()
        case InjectionDetector.MODEL:
            detector = ModelInjectionDetector(
                gateway.for_tier(ModelTier.SMALL), prompts.get(INJECTION_PROMPT)
            )
        case InjectionDetector.OFF:
            detector = NoInjectionDetector()
    leak = PromptLeakGuard(system_prompt, guards.prompt_leak.min_words)
    return Guardrails(guards, detector, PiiRedactor(guards.pii.kinds), leak)
