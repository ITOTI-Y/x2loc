from __future__ import annotations

from functools import partial

from src.agent.config import AgentConfigSchema
from src.agent.nodes.context_collector import context_collector
from src.agent.nodes.decision_router import decision_router
from src.agent.nodes.fetch_empty import fetch_empty
from src.agent.nodes.glossary_loader import glossary_loader
from src.agent.nodes.pattern_extractor import pattern_extractor
from src.agent.nodes.scorer import scorer
from src.agent.nodes.tag_validator import tag_validator
from src.agent.nodes.translator import translator
from src.agent.nodes.uploader import auto_uploader, review_uploader
from src.agent.nodes.user_review import user_review
from src.services.weblate import WeblateClient


class WorkflowNodes:
    def __init__(
        self,
        client: WeblateClient,
        config: AgentConfigSchema,
    ) -> None:
        self.glossary_loader = partial(
            glossary_loader, client=client, agent_config=config
        )
        self.fetch_empty = partial(fetch_empty, client=client, agent_config=config)
        self.context_collector = partial(
            context_collector, client=client, agent_config=config
        )
        self.translator = partial(translator, agent_config=config)
        self.tag_validator = partial(tag_validator, agent_config=config)
        self.scorer = partial(scorer, agent_config=config)
        self.decision_router = partial(decision_router, agent_config=config)
        self.auto_uploader = partial(auto_uploader, client=client, agent_config=config)
        self.user_review = user_review
        self.review_uploader = partial(
            review_uploader, client=client, agent_config=config
        )
        self.pattern_extractor = pattern_extractor
