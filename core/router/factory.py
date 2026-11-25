"""Builds the router from the pack's ``router`` section."""

from core.config import ModelTier, RouterConfig, RouterKind
from core.llm import ModelGateway
from core.prompts import PromptStore
from core.router.router import ROUTER_PROMPT, FixedRouter, HeuristicRouter, ModelRouter, Router


def build_router(config: RouterConfig, gateway: ModelGateway, prompts: PromptStore) -> Router:
    match config.kind:
        case RouterKind.FIXED:
            return FixedRouter(config.default_tier)
        case RouterKind.HEURISTIC:
            return HeuristicRouter(config.default_tier)
        case RouterKind.MODEL:
            return ModelRouter(
                gateway.for_tier(ModelTier.SMALL), prompts.get(ROUTER_PROMPT), config.default_tier
            )
