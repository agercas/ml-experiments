# Pydantic models
from pydantic import BaseModel, Field


class FeasibilityCheck(BaseModel):
    """The result of the feasibility check"""

    feasible: bool = Field(description="Whether the question is feasible to answer with the available tools")
    reasoning: str = Field(description="The reasoning for the feasibility check")


class NextStep(BaseModel):
    """The next step in the plan"""

    step: str = Field(description="Description of the next step to take")
    tools: list[str] = Field(description="List of tool names to use for this step")
    is_final: bool = Field(description="Whether this is the final step")


class FinalConclusion(BaseModel):
    """A final conclusion from the executor"""

    conclusion: str = Field(description="The conclusion based on the work completed so far")
    partial_results: str = Field(description="Summary of partial results obtained")


class FinalAnswer(BaseModel):
    """The final answer to the question"""

    answer: str = Field(description="The comprehensive final answer to the question")
    reasoning: str = Field(description="The reasoning behind the final answer")
