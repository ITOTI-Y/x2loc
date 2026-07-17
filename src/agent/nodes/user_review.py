from typing import TypedDict

from langgraph.types import interrupt

from src.models.agent import NewAgentStateSchema, ReviewDecisionSchema


class UserReviewOutputSchema(TypedDict):
    decisions: list[ReviewDecisionSchema]


def user_review(state: NewAgentStateSchema) -> dict:
    scores = state.scores

    decisions: list[ReviewDecisionSchema] = interrupt(scores)

    return {"decisions": decisions}
